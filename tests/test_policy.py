"""The training policy's agreed public pixel/distribution boundary."""
import pytest

torch = pytest.importorskip('torch')
from flycade.policy import ConnectomePolicy


def test_removing_directed_path_changes_distribution_and_preserves_pixel_gradient():
    torch.manual_seed(7)
    policy = ConnectomePolicy(3, [[0, 1], [1, 2]], [5., 7.], [0], [2], 4, 2, 1)
    pixels = torch.full((1, 1, 8, 8, 1), 180, dtype=torch.uint8)
    original, value = policy(pixels)
    removed, _ = policy(pixels, edge_scale=0.)
    assert not torch.allclose(original.probs, removed.probs, atol=1e-7, rtol=0)
    (-original.log_prob(torch.tensor([2])) + value.square()).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
               for p in policy.parameters())
    reversed_policy = ConnectomePolicy(3, [[1, 0], [2, 1]], [5., 7.], [0], [2], 4, 2, 1)
    reverse, _ = reversed_policy(pixels)
    reverse_cut, _ = reversed_policy(pixels, edge_scale=0.)
    torch.testing.assert_close(reverse.probs, reverse_cut.probs)
