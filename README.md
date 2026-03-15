# climate-scientist

Small sandbox for hurricane-focused LLM experiments:

- A workflow-driven CLI "climate scientist agent" MVP (`src/agent_cli.py`) that runs auditable research workflows and writes a `runs/<run_id>/` bundle.
- Data prep script that creates a supervised fine-tuning (SFT) toy dataset from the North Atlantic HURDAT2 archive using `tropycal`.
- A DeepSeek-V3.2 “hurricane forecaster” MVP (with optional V3.2-Speciale endpoint) that turns structured storm/environment data into 24h/48h/72h predictions.
- A tiny evaluator that validates model output JSON and can score it against truth for 24h lead.

## Environment setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

You’ll need a reasonably recent Python 3 (3.10+ recommended).

If you hit an `Operation not permitted` error under `~/.cache/uv/...`, set:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
```

---

## 0. Climate Scientist Agent CLI (MVP)

The agent CLI is a **workflow runner**: it executes a predefined workflow and writes an auditable run bundle under `runs/<run_id>/`:

- `plan.json`, `tool_calls.jsonl`, `env.lock`, `data_manifest.json`, `metrics.json`, `report.md`, `artifacts/`

Examples:

```bash
source .venv/bin/activate

# Full auto "scientist" run (updates docs/PAPER.md + docs/RUNS.md)
# Requires OPENAI_API_KEY; DeepSeek forecast experiments are optional (DEEPSEEK_API_KEY).
uv run --no-project python src/agent_cli.py \
  --goal "Improve the paper: add 1 hypothesis + results table" \
  --workflow auto \
  --data sim_outputs \
  --budget fast \
  --allow-web \
  --out runs

# Reproduce 24/48/72h metrics on an existing payloads/truth dataset
uv run --no-project python src/agent_cli.py \
  --goal "Reproduce baseline metrics on sim_outputs" \
  --workflow reproduce \
  --data sim_outputs \
  --out runs

# Reproduce HURDAT2 24h persistence baseline from the toy parquet
uv run --no-project python src/agent_cli.py \
  --goal "Reproduce HURDAT2 persistence baseline" \
  --workflow reproduce_hurdat2 \
  --data hurdat2_llm_toy \
  --out runs
```

Workflow definitions live in `src/workflows/`.

## 0b. Headless RI literature harvest (no LLM required)

```bash
uv run --no-project python src/harvest_ri_papers.py \
  --query "tropical cyclone rapid intensification" \
  --from-date 2020-01-01 \
  --max-results 10 \
  --include "rapid intensification,intensity forecast,ri probability" \
  --exclude "wildfire,impact,infrastructure,slowdown" \
  --download-pdf
```

## 1. Build the HURDAT2 SFT dataset (`src/data_prep.py`)

The data prep script pulls storm tracks from the HURDAT2 North Atlantic archive via `tropycal`, builds 48h-history → 24h-forecast samples, exports parquet/JSONL splits, and prints a simple 24h persistence baseline.

Run:

```bash
uv run --no-project python src/data_prep.py
```

`tropycal` will automatically download the HURDAT2 data on first run and cache it under `~/.tropycal`.

### Outputs

- `hurdat2_llm_toy/all_samples.parquet` – full sample table with features and targets
- `hurdat2_llm_toy/train_sft.jsonl`, `val_sft.jsonl`, `test_sft.jsonl` – prompts and targets suitable for SFT

### Notes

- Seasons are limited to 1980–2022; adjust `SEASON_MIN/SEASON_MAX` in `src/data_prep.py` if needed.
- The 24h persistence baseline uses great-circle distance (track) and last-known intensity (wind) as a simple reference metric.
- At the end, the script prints basic sample counts and baseline errors for quick sanity checks.

You will find the generated data under:

- `hurdat2_llm_toy/all_samples.parquet` – full table with numeric fields
- `hurdat2_llm_toy/train_sft.jsonl`, `val_sft.jsonl`, `test_sft.jsonl` – SFT-style prompt/target pairs

---

## 1b. Build simulated LLM payloads (`src/sim_adapter.py`)

Convert simulated hurricane long time series CSVs into LLM payloads + truth:

```bash
uv run --no-project python src/sim_adapter.py \
  --csv /Users/mathinvariant/codes/funProjects/research/climate/hurricanes/hurricane_ft/simulations/preprocessed_data/Galveston_AL_mpi6_ssp245cal/hurricane_long_time_series.csv \
  --out-dir sim_outputs \
  --history-hours 48 \
  --lead-hours 24,48,72 \
  --sample-minutes 60 \
  --stride-hours 6 \
  --guidance-step 3 \
  --guidance-mode summary
```

Outputs:
- `sim_outputs/payloads.jsonl`
- `sim_outputs/truth.jsonl`
- `sim_outputs/meta.json`

---

## 2. DeepSeek hurricane forecaster MVP (`src/forecaster.py`)

This script calls DeepSeek-V3.2 (via the `openai` Python client) as an autonomous hurricane forecaster, using a structured prompt based on storm state, local environment, large-scale pattern, and historical analogs. By default it uses the official API (`https://api.deepseek.com`); you can also target the V3.2-Speciale endpoint via a flag.

### API credentials

The script expects a DeepSeek API key:

- Set `DEEPSEEK_API_KEY` in your shell environment, **or**
- Create a local `.env` file in the project root with a line like:

  ```bash
  DEEPSEEK_API_KEY=sk-...
  ```

Optional overrides:

- `DEEPSEEK_BASE_URL` – override the base URL (defaults to `https://api.deepseek.com`)
- `DEEPSEEK_MODEL` – override the model name (defaults to `deepseek-reasoner`)

### Usage

Inspect the generated prompt without calling the API:

```bash
uv run --no-project python src/forecaster.py --preview
```

Call the model and ask it to return JSON (recommended if you want to use the evaluator below):

```bash
uv run --no-project python src/forecaster.py --json > examples/forecast_output.json
```

Use your own real data instead of the built-in demo by providing a payload JSON:

```bash
uv run --no-project python src/forecaster.py --json \
  --payload-json examples/my_storm_payload.json > examples/forecast_output.json
```

If you just want to reuse one of the HURDAT2 samples as input, you can first build a payload JSON from `all_samples.parquet`:

```bash
uv run --no-project python src/payloads.py --index 0 --out examples/my_storm_payload.json

uv run --no-project python src/forecaster.py --json \
  --payload-json examples/my_storm_payload.json > examples/forecast_output.json
```

To compare different DeepSeek-V3.2 variants:

- Non-thinking mode (chat): `--model deepseek-chat`
- Reasoning mode (similar to “Speciale”): `--model deepseek-reasoner`
 - V3.2-Speciale endpoint (thinking-only, no tool calls): add `--speciale` (forces the base URL to `https://api.deepseek.com/v3.2_speciale_expires_on_20251215`)

For example:

```bash
# Chat-style forecasts
uv run --no-project python src/forecaster.py --json \
  --model deepseek-chat \
  --payload-json examples/my_storm_payload.json > examples/forecast_chat.json

# Reasoning-style forecasts
uv run --no-project python src/forecaster.py --json \
  --model deepseek-reasoner \
  --payload-json examples/my_storm_payload.json > examples/forecast_reasoner.json
```

Call the V3.2-Speciale endpoint for comparison (server-side JSON mode is not supported there; the script enforces JSON only via the prompt and prints a warning):

```bash
uv run --no-project python src/forecaster.py --speciale --json \
  --model deepseek-reasoner \
  --max-tokens 12000 \
  --payload-json examples/my_storm_payload.json > examples/forecast_speciale.json
```

The payload file should look like:

```json
{
  "storm": {
    "id": "AL09",
    "basin": "Atlantic",
    "time": "2021-09-10 12Z",
    "lat": 24.3,
    "lon": -68.2,
    "wind": 95,
    "pressure": 965,
    "motion": "305 deg at 9 kt"
  },
  "environment": {
    "SST": "29.5 C",
    "Vertical wind shear (200-850 hPa)": "8 kt from WSW"
  },
  "large_scale": {
    "Subtropical ridge": "axis SW-NE, centered NE of storm"
  },
  "analogs": [
    {"name": "Analog 1", "summary": "Short description of a similar historical storm."}
  ],
  "guidance": [
    "Any external guidance or notes you want the model to see."
  ]
}
```

Useful flags:

- `--temperature` – sampling temperature (default `0.4`)
- `--max-tokens` – max tokens for the response (default `4096`; for `deepseek-chat` the valid range is [1, 8192]) 
- `--reasoning-limit` – soft limit on chain-of-thought tokens, enforced via the system prompt

The current script uses a built-in demo payload; wiring it to real-time data is left to downstream applications.

---

## 2b. Batch LLM forecasts (`src/run_forecaster_jsonl.py`)

Run LLM over a JSONL payload file:

```bash
uv run --no-project python src/run_forecaster_jsonl.py \
  --payloads sim_outputs/payloads.jsonl \
  --out sim_outputs/predictions.jsonl \
  --json \
  --limit 20
```

---

## 2c. Simple baselines (`src/baselines.py`)

Persistence and kinematic baselines from the payload JSONL:

```bash
uv run --no-project python src/baselines.py \
  --payloads sim_outputs/payloads.jsonl \
  --out sim_outputs/baseline_persistence.jsonl \
  --method persistence

uv run --no-project python src/baselines.py \
  --payloads sim_outputs/payloads.jsonl \
  --out sim_outputs/baseline_kinematic.jsonl \
  --method kinematic
```

---

## 3. Evaluating forecast JSON (`src/evaluate.py`)

This script validates DeepSeek’s forecast JSON (24/48/72h entries) and can compute simple errors versus a provided 24h truth.

### Expected response format

The response JSON should look like:

```json
{
  "forecast": [
    {"lead_hours": 24, "lat": 25.0, "lon": -70.0, "wind": 95.0},
    {"lead_hours": 48, "lat": 27.0, "lon": -72.0, "wind": 100.0},
    {"lead_hours": 72, "lat": 29.0, "lon": -73.5, "wind": 95.0}
  ],
  "reasoning": "Short explanation of track and intensity evolution."
}
```

### Running the evaluator

Validate only:

```bash
uv run --no-project python src/evaluate.py --response examples/forecast_output.json
```

Validate and compute 24h errors against truth (track MAE in km and wind MAE in kt):

```bash
uv run --no-project python src/evaluate.py --response examples/forecast_output.json --truth truth_24h.json
```

The truth file must contain:

```json
{"lat_24h": 25.3, "lon_24h": -69.8, "wind_24h": 100}
```

You can derive such truth values from the HURDAT-based dataset produced by `src/data_prep.py` (e.g., by selecting a row and exporting the 24h verifying position/intensity).

---

## 3b. Evaluating JSONL batches (`src/evaluate_jsonl.py`)

Compare JSONL predictions against JSONL truth (line-aligned):

```bash
uv run --no-project python src/evaluate_jsonl.py \
  --predictions sim_outputs/predictions.jsonl \
  --truth sim_outputs/truth.jsonl
```

---

## 3c. Rapid Intensification (RI) metrics (`src/ri_metrics.py`)

Compute RI classification metrics from deterministic wind forecasts:

```bash
uv run --no-project python src/ri_metrics.py \
  --payloads sim_outputs/payloads.jsonl \
  --truth sim_outputs/truth.jsonl \
  --predictions sim_outputs/predictions.jsonl \
  --lead-hours 24 \
  --threshold-kt 30 \
  --out-json results/ri_metrics.json
```

---

## 4. V3.2-Speciale examples (`src/speciale_example.py`)

For minimal, focused examples of the V3.2-Speciale endpoint (outside the full hurricane pipeline), you can use:

- Simple QA demo:

  ```bash
  uv run --no-project python src/speciale_example.py simple \
    "Explain why hurricanes weaken after landfall."
  ```

- Hurricane-style demo using the built-in payload:

  ```bash
  uv run --no-project python src/speciale_example.py hurricane-demo
  ```
