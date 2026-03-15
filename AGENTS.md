# Repository Guidelines

## Project Structure & Module Organization

- `src/data_prep.py`: Builds a toy HURDAT2 dataset (48h history → 24h target).
- `src/forecaster.py`: LLM forecaster scaffold; outputs JSON for 24/48/72h.
- `src/payloads.py`: Converts a HURDAT2 sample row into LLM payload JSON.
- `src/evaluate.py`: Validates forecast JSON and scores 24h errors.
- `src/speciale_example.py`: Minimal V3.2-Speciale examples.
- `src/sim_adapter.py`: Builds LLM payloads + truth JSONL from simulated CSVs.
- `src/run_forecaster_jsonl.py`: Runs LLM over JSONL payloads.
- `src/evaluate_jsonl.py`: Evaluates JSONL predictions vs truth.
- `src/baselines.py`: Persistence/kinematic baselines from payload JSONL.
- `hurdat2_llm_toy/`: Generated artifacts (parquet + JSONL splits).
- `examples/`: Example payloads and model outputs (`forecast_*.json`).
- `docs/DISCUSSION.md`: Research plan + literature snapshot (keep updated when scope changes).

External data and foundation-model baselines live outside this repo at:
`/Users/mathinvariant/codes/funProjects/research/climate/hurricanes/hurricane_ft`.

## Build, Test, and Development Commands

Environment management (always use `uv`):
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If `uv` is missing, install it first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Generate HURDAT2 toy dataset:
```bash
python src/data_prep.py
```

Preview LLM prompt:
```bash
python src/forecaster.py --preview
```

Run LLM forecast (JSON mode):
```bash
python src/forecaster.py --json > examples/forecast_output.json
```

Validate/scoring:
```bash
python src/evaluate.py --response examples/forecast_output.json --truth truth_24h.json
```

## Coding Style & Naming Conventions

- Python, 4-space indentation, minimal boilerplate.
- Prefer short, descriptive variable names; avoid heavy guardrails or overly defensive code.
- Keep scripts readable and linear; add comments only for non-obvious logic.
- JSON outputs should follow the schema documented in `src/forecaster.py`.

## Testing Guidelines

- No formal test suite in this repo yet.
- Use the evaluator (`src/evaluate.py`) as a functional sanity check.
- When adding new scripts, include a quick “how to run” example in the README or `DISCUSSION.md`.

## Commit & Pull Request Guidelines

- No established commit convention detected; use clear, imperative messages (e.g., “Add sim adapter for CSV inputs”).
- PRs should include: purpose, key changes, and any sample outputs/paths touched.

## Configuration & Secrets

- Set `DEEPSEEK_API_KEY` via environment or a local `.env` file.
- Optional overrides: `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `DEEPSEEK_SPECIALE_BASE_URL`.
