# AGENTS.md

## Project Rules for Agents

Use `uv` for Python dependency management and command execution in this repository.

## Environment

1. Create/sync environment with:
```bash
uv sync
```

2. Add libraries with:
```bash
uv add <package>
uv add --dev <package>
```

Do not use `pip install` directly unless explicitly requested.

## Running Scripts

Run all Python scripts through `uv run`:

```bash
uv run python app.py
uv run python collect_two_hand_sequence_data.py
uv run python train_keypoint_classifier.py
```

Run notebooks/tools through `uv run` as well:

```bash
uv run jupyter notebook
```

## Notes

1. Keep new dependencies in `pyproject.toml` via `uv add`.
2. Prefer reproducible commands that work from repo root.
3. If adding a new script command to docs, show the `uv run ...` form.

