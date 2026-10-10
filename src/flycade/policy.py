"""Pixel encoder -> directed fixed COO graph -> output-only actor/critic."""
from collections.abc import Callable
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class ConnectomePolicy(nn.Module):
    def __init__(self, nodes: int, edges: list[list[int]], weights: list[float],
                 inputs: list[int], outputs: list[int], state_dim: int = 8,
                 propagation_steps: int = 2, channels: int = 12):
        super().__init__()
        if not inputs or not outputs or set(inputs) & set(outputs):
            raise ValueError('Input/output groups must be nonempty and disjoint')
        if min(nodes, state_dim, propagation_steps, channels) <= 0:
            raise ValueError('Model dimensions and propagation steps must be positive')
        indices = torch.tensor(edges, dtype=torch.long).T
        values = torch.tensor(weights, dtype=torch.float32)
        if indices.shape != (2, len(weights)) or not len(weights):
            raise ValueError('Expected nonempty edge pairs and matching weights')
        if indices.min() < 0 or indices.max() >= nodes or not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Invalid graph indices or weights')
        if min(inputs + outputs) < 0 or max(inputs + outputs) >= nodes:
            raise ValueError('Invalid input/output indices')
        incoming = torch.zeros(nodes).scatter_add_(0, indices[1], values)
        values = values / incoming[indices[1]]
        self.register_buffer('adjacency', torch.sparse_coo_tensor(
            indices.flip(0), values, (nodes, nodes)).coalesce())
        self.register_buffer('inputs', torch.tensor(inputs))
        self.register_buffer('outputs', torch.tensor(outputs))
        self.nodes, self.state_dim, self.propagation_steps = nodes, state_dim, propagation_steps
        self.encoder = nn.Sequential(nn.Conv2d(channels, state_dim, 3, padding=1), nn.Tanh(),
                                     nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                     nn.Linear(state_dim * 16, len(inputs) * state_dim), nn.Tanh())
        self.actor = nn.Linear(len(outputs) * state_dim, 7)
        self.critic = nn.Linear(len(outputs) * state_dim, 1)

    def forward(self, pixels: Tensor, *, edge_scale: float = 1.,
                observe: Callable[[Tensor], None] | None = None) -> tuple[Categorical, Tensor]:
        """Batch of uint8 [B,stack,H,W,C]; optional uniform edge intervention."""
        if pixels.dtype != torch.uint8 or pixels.ndim != 5:
            raise ValueError('Expected uint8 [batch, stack, height, width, channels] pixels')
        batch, stack, height, width, channels = pixels.shape
        image = pixels.permute(0, 1, 4, 2, 3).reshape(batch, stack * channels, height, width).float() / 255.
        encoded = self.encoder(image).reshape(batch, -1, self.state_dim)
        drive = encoded.new_zeros((batch, self.nodes, self.state_dim)).index_copy(1, self.inputs, encoded)
        state = drive
        for _ in range(self.propagation_steps):
            dense = state.permute(1, 0, 2).reshape(self.nodes, -1)
            propagated = torch.sparse.mm(self.adjacency, dense) * edge_scale
            state = torch.tanh(propagated.reshape(self.nodes, batch, self.state_dim).permute(1, 0, 2) + drive)
        if observe is not None:
            observe(state)
        readout = state[:, self.outputs].reshape(batch, -1)
        return Categorical(logits=self.actor(readout)), self.critic(readout).squeeze(-1)


def policy_from_graph(graph: Path, state_dim: int, propagation_steps: int, channels: int) -> ConnectomePolicy:
    """Construct the training/evaluation policy from an already verified graph artifact."""
    nodes = json.loads((graph / 'nodes.json').read_text())
    return ConnectomePolicy(len(nodes), np.load(graph / 'edge_index.npy', allow_pickle=False).T.tolist(),
        np.load(graph / 'weight.npy', allow_pickle=False).tolist(),
        [node['index'] for node in nodes if node['input']], [node['index'] for node in nodes if node['output']],
        state_dim, propagation_steps, channels)


class CNNPolicy(nn.Module):
    """Small untrained convolutional control; no connectome or synthetic neuron activity."""
    def __init__(self, channels: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(channels, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(), nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(), nn.Linear(32 * 9, 64), nn.ReLU())
        self.actor = nn.Linear(64, 7)
        self.critic = nn.Linear(64, 1)

    def forward(self, pixels: Tensor, *, observe: Callable[[Tensor], None] | None = None) -> tuple[Categorical, Tensor]:
        if pixels.dtype != torch.uint8 or pixels.ndim != 5:
            raise ValueError('Expected uint8 [batch, stack, height, width, channels] pixels')
        batch, stack, height, width, channels = pixels.shape
        image = pixels.permute(0, 1, 4, 2, 3).reshape(batch, stack * channels, height, width).float() / 255.
        readout = self.encoder(image)
        # The common observer callback is intentionally not called: this model has no circuit.
        return Categorical(logits=self.actor(readout)), self.critic(readout).squeeze(-1)


Policy = ConnectomePolicy | CNNPolicy


def policy_for_kind(graph: Path, kind: str, state_dim: int, propagation_steps: int, channels: int) -> Policy:
    if kind == 'cnn':
        return CNNPolicy(channels)
    if kind == 'connectome':
        return policy_from_graph(graph, state_dim, propagation_steps, channels)
    raise ValueError(f'Unknown model kind: {kind}')
