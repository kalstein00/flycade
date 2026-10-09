# Domain Docs

This repo uses a single-context layout: `GLOSSARY.md` at the repo root and ADRs under `docs/adr/`.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root.
- **`docs/adr/`**: read ADRs that touch the area you are about to work in.

If either is missing, proceed silently. `/domain-modeling` creates domain docs lazily when terms or decisions get resolved.

## File structure

```text
/
├── GLOSSARY.md
└── docs/adr/
    ├── 0001-<decision>.md
    └── 0002-<decision>.md
```

## Use the glossary's vocabulary

When naming a domain concept in an issue title, refactor proposal, hypothesis, or test name, use the term defined in `GLOSSARY.md`.

If a needed concept is absent, reconsider whether it belongs to the project's language or note the gap for `/domain-modeling`.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface the conflict explicitly and explain why reopening the decision may be warranted.
