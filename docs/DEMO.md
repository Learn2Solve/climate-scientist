# Demo: ERA5 Idalia (2023) — Aurora vs LLM

Goal: show a working end-to-end pipeline comparing a climate foundation model (Aurora) vs an LLM baseline on the same **ERA5-derived truth**.

## What this demo is / isn’t

- **Is:** a reproducible pipeline: ERA5 → (track/wind proxy) → payloads → Aurora rollout → LLM forecast → metrics.
- **Isn’t:** a perfectly fair “same-input” contest (Aurora consumes grids; LLM consumes extracted track history).

## Outputs (ready)

- Metrics table: `results/era5_demo/idalia2023_metrics.md`
- Plots: `results/era5_demo/plots/idalia2023/track_mae_mean.png`, `results/era5_demo/plots/idalia2023/wind_mae_mean.png`
- Predictions:
  - Aurora: `results/era5_idalia2023/aurora_preds_20.jsonl`
  - LLM (DeepSeek-chat): `results/era5_idalia2023/llm_preds_20.jsonl`
- Truth (20): `results/era5_idalia2023_payloads/truth_20.jsonl`

## Reproduce (commands)

1) Download ERA5 subset (requires `~/.cdsapirc`):
```bash
uv run --with cdsapi python src/era5_dorian_download.py \
  --out-dir results/era5_idalia2023 \
  --year 2023 \
  --area 35,-95,10,-65 \
  --aug-days 27,28,29,30,31 \
  --sep-days 01,02,03,04,05 \
  --force
```

2) Extract a TC proxy track (local min MSLP tracking + max wind in radius):
```bash
uv run --with xarray --with netcdf4 python src/era5_tc_track.py \
  --surf-files results/era5_idalia2023/surface_2023_08.nc,results/era5_idalia2023/surface_2023_09.nc \
  --init-lat 20 --init-lon -86 \
  --center-search-radius-km 800 \
  --radius-km 500 --to-knots --lon-180 \
  --out-jsonl results/era5_idalia2023/track.jsonl
```

3) Build anonymized payloads + truth:
```bash
uv run python src/era5_track_to_payloads.py \
  --track-jsonl results/era5_idalia2023/track.jsonl \
  --out-dir results/era5_idalia2023_payloads \
  --history-hours 48 --lead-hours 24,48,72 --stride-hours 6 \
  --storm-id case_001 --basin ATL --anonymize
head -n 20 results/era5_idalia2023_payloads/truth.jsonl > results/era5_idalia2023_payloads/truth_20.jsonl
```

4) Aurora (20 samples):
```bash
uv run --with microsoft-aurora --with xarray --with netcdf4 python src/aurora_rollout_tc_batch.py \
  --static-path results/era5_idalia2023/static.nc \
  --surf-files results/era5_idalia2023/surface_2023_08.nc,results/era5_idalia2023/surface_2023_09.nc \
  --atmos-files results/era5_idalia2023/atmos_2023_08.nc,results/era5_idalia2023/atmos_2023_09.nc \
  --payloads results/era5_idalia2023_payloads/payloads.jsonl \
  --out-jsonl results/era5_idalia2023/aurora_preds_20.jsonl \
  --center-search-radius-km 800 --radius-km 500 --to-knots --lon-180 \
  --limit 20
```

5) LLM (20 samples, requires `DEEPSEEK_API_KEY`):
```bash
uv run python src/run_forecaster_jsonl.py \
  --payloads results/era5_idalia2023_payloads/payloads.jsonl \
  --out results/era5_idalia2023/llm_preds_20.jsonl \
  --model deepseek-chat --json --max-tokens 2048 --limit 20
```

6) Evaluate + plot:
```bash
uv run python src/evaluate_jsonl.py --predictions results/era5_idalia2023/aurora_preds_20.jsonl --truth results/era5_idalia2023_payloads/truth_20.jsonl
uv run python src/evaluate_jsonl.py --predictions results/era5_idalia2023/llm_preds_20.jsonl --truth results/era5_idalia2023_payloads/truth_20.jsonl

uv run python src/report_metrics.py \
  --truth results/era5_idalia2023_payloads/truth_20.jsonl \
  --model "Aurora-small:results/era5_idalia2023/aurora_preds_20.jsonl" \
  --model "DeepSeek-chat:results/era5_idalia2023/llm_preds_20.jsonl" \
  --out-csv results/era5_demo/idalia2023_metrics.csv \
  --out-json results/era5_demo/idalia2023_metrics.json \
  --out-md results/era5_demo/idalia2023_metrics.md

uv run python src/plot_metrics.py --csv results/era5_demo/idalia2023_metrics.csv --out-dir results/era5_demo/plots/idalia2023
```
