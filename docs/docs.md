# PaperZero Documentation

Reference documentation written by Codex.

Contact me by email or on Slack if you think there are mistakes. You are probably correct.

## At A Glance

| Topic | Where to Start |
| --- | --- |
| API reference by module | [Module References](#module-references) |
| Installation | [Setup](#setup) |
| Branch responsibilities | [Branch Structure](#branch-structure) |
| Running Python files correctly | [Execution Patterns](#execution-patterns) |

## Module References

- [`paper`](modules/paper.md)
- [`framework`](modules/framework.md)
- [`data.origami_sampler`](modules/data.origami_sampler.md)
- [`envs.origami_env`](modules/envs.origami_env.md)

## Setup

Use any Python environment manager (`venv`, Conda, Poetry, etc.).  
Run all commands from the repository root.

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Optional: verify imports

```bash
python -c "import paper; print('paper ok')"
python -c "from data.origami_sampler import OrigamiSampler; print('sampler ok')"
python -c "from envs.origami_env import OrigamiEnv; print('env ok')"
```

Expected output:

```text
paper ok
sampler ok
env ok
```

## Branch Structure

Current branch model:

| Branch | Purpose |
| --- | --- |
| `main` | Shared integration/docs surface and common infrastructure. |
| `mcst` | MCTS-focused development branch. |
| `diffusion` | Diffusion-focused development branch. |

Use this to choose where feature-specific work should happen.

## Execution Patterns

### Preferred way to run package files

When running a Python file that imports sibling packages, use module execution:

```bash
python -m package.module
```

Examples:

```bash
python -m baselines.train_sl
python -m data.origami_sampler
```

Why this helps:

- avoids relative import errors from direct path execution
- ensures package resolution is consistent across machines

### Direct file execution

Direct execution (`python path/to/file.py`) might work for standalone scripts, but is less reliable for package-based modules.

## Scope

> This guide documents the Python import surface and usage patterns.
> It does not document internal C++ implementation details.
