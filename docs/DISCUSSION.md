# Climate Scientist (Hurricane) Paper Notes

## Goal / Claim

We want to test the hypothesis:

- A general-purpose LLM (no domain fine-tuning) + a "climate scientist" workflow
  (structured state, constraints, optional guidance/tools) can beat climate
  foundation-model baselines on a well-defined hurricane prediction task.

In this context, "foundation models" means climate/weather foundation models
(not other LLMs).

## Repos / Artifacts

This repo (`climate_scientist`) currently contains:

- `src/agent_cli.py`: workflow-driven “climate scientist agent” MVP (auditable runs under `runs/`).
- `src/agent/*`: planner/tools/memory/verifier/safety/runner modules for the agent CLI.
- `src/workflows/*`: YAML workflow definitions (start with `reproduce*.yaml`).
- `src/data_prep.py`: build a toy HURDAT2 48h-history -> 24h-forecast dataset.
- `src/payloads.py`: convert a HURDAT2 sample row -> LLM payload JSON.
- `src/forecaster.py`: LLM forecaster scaffold (24/48/72h JSON output).
- `src/evaluate.py`: validate forecast JSON + compute 24h error vs truth JSON.
- `src/sim_adapter.py`: build LLM payloads + truth JSONL from simulated CSVs.
- `src/run_forecaster_jsonl.py`: batch LLM over JSONL payloads.
- `src/evaluate_jsonl.py`: batch evaluator for JSONL predictions vs truth.
- `src/baselines.py`: persistence/kinematic baselines from payload JSONL.

External (simulated data + climate foundation baselines) lives here:

- `DATA_ROOT`: `/Users/mathinvariant/codes/funProjects/research/climate/hurricanes/hurricane_ft`

Current layout (post-merge):

- `data/` (best tracks, reference text files)
- `simulations/` (event sets + preprocessed CSVs)
- `scripts/` (TTM baselines + training/prediction scripts)
- `utils/` (data prep, plotting, HURDAT parsing, etc.)
- `notebooks/` (Aurora/Prithvi experiments)
- `core/` (conformal/GP uncertainty modules)

## Simulated Dataset (from `hurricane_ft`)

Where the simulated data now lives:

- `simulations/preprocessed_data/*/hurricane_long_time_series.csv`
- Combined sets: `simulations/preprocessed_data/20thcal_combined/`,
  `simulations/preprocessed_data/ssp245cal_combined/`

Example columns (26 total in the CSV we inspected):

- Core storm state: `latstore`, `longstore` (0..360 lon), `vstore`, `pstore`
- Environment: `shearstore`, `u850store`, `v850store`, `T600store`, `rhstore`, ...

Two practical notes for using these CSVs:

- `longstore` is 0..360; convert to [-180, 180] before error metrics and LLM prompts.
- The long series contains long stretches of zeros (likely "no storm" windows);
  we should segment events (e.g., contiguous `vstore > 0`) before building samples.

## Climate Foundation Model Baselines (from `hurricane_ft`)

The `hurricane_ft` directory also contains a lot of baseline / FM-related code.
Notably:

Tiny Time Mixer (TTM):

- `utils/climate_moment.py`: uses `tsfm_public.TinyTimeMixerForPrediction` (R1/R2)
  to run zeroshot + finetune on the preprocessed datasets.
- `scripts/hurricane_TTMs.py`, `scripts/hurricane_TTMs_combined.py`,
  `scripts/hurricane_TTMs_gp.py`, `scripts/hurricane_TTMs_conformal.py`:
  variations/experiments around TTM.
- `scripts/train_hurricane_ttm.py`: another training script (uses HF
  `AutoModelForCausalLM` with IBM Granite TTM checkpoints).
- `core/` includes GP + conformal prediction components used by some TTM variants.

Other climate FM experiments (mostly notebooks):

- `notebooks/ft_prithvix_hurricane.ipynb`, `notebooks/ft_prithvix_hurricane_mixing.ipynb`:
  Prithvi-WxC work.
- `notebooks/aurora.ipynb`: Aurora usage (Microsoft).

Outputs:

- `results/*`, `output/*`: plots and logs for different datasets / model runs.
- `simulations/*`: upstream event sets + preprocessed `.mat`/CSV artifacts.

## Proposed Experimental Matrix (paper-friendly)

Define one canonical target:

- Predict lat/lon + intensity at 24h/48h/72h given the past 48h (and optional env vars).

Suggested comparisons:

1) Climate foundation baseline(s) (direct inference; no LLM).
2) LLM naive baseline (same information, but unstructured prompt).
3) LLM + "climate scientist" scaffold (structured state + constraints + schema).
4) Optional: LLM + foundation guidance (FM forecast injected as "guidance" for a
   decision layer / correction step).

Core metrics to report:

- Track error (km) @ 24/48/72 (mean + median).
- Intensity error @ 24/48/72 (mean + median).
- `valid_json_rate` (LLM outputs that are parseable and schema-valid).

## Rapid Intensification (RI) focus (Feb 2026)

Motivation: RI is a high-impact failure mode for intensity forecasting; we treat it as a
separate evaluation target rather than relying on overall wind MAE alone.

Definition (default): **RI if ΔV ≥ 30 kt over 24h**.

We report:
- `truth_ri_rate` on the evaluated dataset/subset.
- RI precision/recall/F1 derived from deterministic wind forecasts.
- Wind MAE conditioned on RI vs non-RI cases.

Script: `src/ri_metrics.py` (line-aligned payloads/truth/predictions JSONL).

### RI baselines (starting point)

- `persistence`: always predicts wind(t+lead)=wind(t).
- `ri_gate` (in `src/baselines.py`): a simple RI-aware heuristic baseline that gates a fixed
  intensity jump using an environment feature threshold (default: `pstore` 75th percentile).

## Literature Snapshot (Feb 2026 refresh)

Key takeaways from recent work:

- Earth FMs (Aurora, AIFS) now beat or match operational NWP on many targets,
  but AIFS smooths extremes and underestimates variance (important for TCs).
- LLMs can act as zero-shot **analysis/decision layers** over structured data,
  with "proof blocks" to ground claims (AI-Meteorologist line of work).
- TC intensity forecasting sees strong gains from specialized ML (e.g., OWZP-Transformer),
  suggesting a credible target where LLM reasoning can add value without finetuning.
- New TC benchmark datasets (e.g., TCN-D) emphasize global, multi-basin evaluation.
- Governance risk is real (e.g., AI-generated weather products hallucinating locations),
  reinforcing the need for strict schema validation and constrained outputs.

Annotated refs (short list, for anchoring related-work text):

- Aurora (Nature 2025; arXiv:2405.13063) — Earth system FM with strong general skill.
- ECMWF AIFS (arXiv:2406.01465; operational Feb 2025) — data-driven global model at NWP scale.
- Prithvi-WxC (arXiv:2409.13598) — foundation model for weather/climate fields (NASA/IMPACT).
- AI-Meteorologist (arXiv:2511.23387) — LLM meteorology reports with structured “proof” grounding.
- TropiCycloneNet (Nature Comms 2025) — global TC benchmark (track + intensity).
- TIFNet (npj Climate & Atmospheric Science 2026) — transformer for TC intensity from satellite imagery.
- WeatherGFM (arXiv:2411.05420) — Global Forecasting Model benchmark/framework.

## Implementation Notes / Decisions Already Made

- We updated `src/forecaster.py` to format lat/lon as N/S/E/W to
  support global basins (not hard-coded N/W).
- DeepSeek docs still list `https://api.deepseek.com` as the base URL, and the
  published V3.2-Speciale endpoint expired on 2025-12-15; we made the Speciale
  base URL overridable via `DEEPSEEK_SPECIALE_BASE_URL`.
- DeepSeek Reasoner: we will **not** run batch evaluation (content can be empty
  if `max_tokens` gets consumed by CoT). Reasoner will be used only for 1–2
  qualitative case studies. Quantitative runs use `deepseek-chat`.

## Latest Baselines (TTM sanity check)

- Script: `src/ttm_baseline.py` (univariate TTM per target; hourly resample).
- Checkpoint: `ibm-granite/granite-timeseries-ttm-r2` (context=512, pred=96).
- Output: `sim_outputs_ttm/predictions.jsonl`, `sim_outputs_ttm/truth.jsonl`.
- 50-sample run (hourly resample, stride 6h):
  - Track MAE (km): ~2287 (24h), ~2343 (48h), ~2382 (72h)
  - Wind MAE (kt): ~15.85 (24h), ~15.83 (48h), ~15.86 (72h)
- Caveat: these metrics are on the TTM-generated windows (not yet aligned
  to the LLM JSONL sampling order).

## Aligned Baselines (LLM vs TTM on same payloads)

Aligned to `sim_outputs/payloads.jsonl` (first 200 samples, hourly cadence).

- **TTM aligned** (`src/ttm_baseline_aligned.py`)
  - Track MAE (km): ~7758 (24h), ~7923 (48h), ~8171 (72h)
  - Wind MAE (kt): ~14.33 (24h), ~14.24 (48h), ~14.04 (72h)
- **DeepSeek-chat LLM** (`src/run_forecaster_jsonl.py`)
  - Track MAE (km): ~7373 (24h), ~8960 (48h), ~8883 (72h)
  - Wind MAE (kt): ~18.92 (24h), ~25.07 (48h), ~24.84 (72h)

Notes:
- TTM is univariate (lat/lon/wind separately); no cross-variable conditioning.
- LLM here is a raw prompt baseline (no FM guidance).

## Agent Baselines (200 samples, same payloads)

- **Claude Code (Opus 4.5)** (`src/agent_baseline.py`)
  - Track MAE (km): ~7159 (24h), ~8512 (48h), ~8523 (72h)
  - Wind MAE (kt): ~18.19 (24h), ~24.45 (48h), ~25.99 (72h)
- **Codex (gpt-5.2-codex)** (`src/agent_baseline.py`)
  - Track MAE (km): ~7382 (24h), ~8902 (48h), ~8822 (72h)
  - Wind MAE (kt): ~18.25 (24h), ~22.45 (48h), ~20.93 (72h)

Artifacts:
- Summary table: `results/metrics.md`
- Raw metrics: `results/metrics.csv`, `results/metrics.json`
- Plots: `results/plots/track_mae_mean.png`, `results/plots/wind_mae_mean.png`

## ERA5 Dorian (2019) – Aurora vs LLM (20 samples, pipeline demo)

This is a **pipeline demo** to compare a climate FM (Aurora) to an LLM on the
same **ERA5-derived truth**. Inputs are *not* identical (Aurora uses grids; LLM
uses extracted track history), so treat as **non‑strict** comparison.

Dataset:
- ERA5 subset: 2019‑08‑26 → 2019‑09‑05, 6‑hourly
- Region: 40N–5N, 120W–40W (Atlantic box)
- Truth extraction: **min MSLP** center + **max 10m wind within 500 km**

Pipeline artifacts:
- ERA5 data: `results/era5_dorian/*`
- Track series: `results/era5_dorian/track.jsonl`
- Payloads/truth: `results/era5_dorian_payloads/payloads.jsonl`, `truth.jsonl`
- Aurora preds: `results/era5_dorian/aurora_preds.jsonl`
- LLM preds (DeepSeek‑chat): `results/era5_dorian/llm_preds_20.jsonl`

Results (20 samples):
- **Aurora (small)**  
  - Track MAE (km): ~2601 (24h), ~2993 (48h), ~3060 (72h)  
  - Wind MAE (kt): ~27.60 (24h), ~34.87 (48h), ~39.90 (72h)
- **DeepSeek‑chat LLM**  
  - Track MAE (km): ~1058 (24h), ~1574 (48h), ~1662 (72h)  
  - Wind MAE (kt): ~8.56 (24h), ~16.16 (48h), ~24.86 (72h)

Notes / caveats:
- Aurora may be **in‑distribution** for 2019 ERA5.
- LLM prompt uses extracted 48h track history (stronger inductive cue).
- This is a **demo** showing pipeline compatibility; not a fair grid‑to‑grid contest.

## ERA5 Idalia (2023) – Aurora vs LLM (20 samples, demo baseline)

This is our main **demo** for “climate FM vs LLM” on a shared ERA5-derived truth.

Key choices for stability (important):
- Track extraction uses **local min MSLP near the previous center** to avoid jumping to unrelated lows.
- LLM payloads are **anonymized** (relative times `t-48h..t0`) to reduce memorization leakage.

Dataset:
- ERA5 subset: 2023‑08‑27 → 2023‑09‑05, 6‑hourly
- Region: 35N–10N, 95W–65W
- Truth extraction:
  - Center = min MSLP within **800 km** of previous center
  - Wind = max 10m wind within **500 km** of center

Artifacts:
- ERA5 data: `results/era5_idalia2023/*`
- Track series: `results/era5_idalia2023/track.jsonl`
- Payloads/truth: `results/era5_idalia2023_payloads/payloads.jsonl`, `truth_20.jsonl`
- Aurora preds: `results/era5_idalia2023/aurora_preds_20.jsonl`
- LLM preds (DeepSeek‑chat): `results/era5_idalia2023/llm_preds_20.jsonl`
- Metrics table: `results/era5_demo/idalia2023_metrics.md`
- Plots: `results/era5_demo/plots/idalia2023/track_mae_mean.png`, `results/era5_demo/plots/idalia2023/wind_mae_mean.png`
- Reproduce guide: `docs/DEMO.md`

Results (20 samples):
- **Aurora (small)**  
  - Track MAE (km): ~348 (24h), ~816 (48h), ~1061 (72h)  
  - Wind MAE (kt): ~20.53 (24h), ~21.66 (48h), ~18.55 (72h)
- **DeepSeek‑chat LLM**  
  - Track MAE (km): ~392 (24h), ~806 (48h), ~1125 (72h)  
  - Wind MAE (kt): ~7.56 (24h), ~17.44 (48h), ~25.61 (72h)

Note:
- Earlier “Idalia” results under `results/era5_idalia/*` were from an accidental **2019** download
  (script default year) and should be treated as obsolete.

## Finalized Plan (MVP)

Phase 1 (small, fast, publishable baseline):

1) **Foundation model baseline:** Tiny Time Mixer (TTM) zeroshot.
2) **LLM baseline:** DeepSeek (no finetune).
3) **Task:** 48h history -> 24/48/72h lat/lon + intensity.
4) **Data:** simulated CSVs from `hurricane_ft/simulations/preprocessed_data/*`
   (segment events where vstore>0; convert lon 0..360 -> [-180, 180]).
5) **Metrics:** track MAE (km) + intensity MAE (kt) at 24/48/72 + `valid_json_rate`.

Phase 2 (optional extensions):

- Add Prithvi/Aurora baselines (from existing notebooks) using the same targets.
- Add an LLM + FM-guidance condition (LLM as decision layer over FM output).
- Add a kinematic extrapolation baseline for sanity check.

## RI literature update
- Added memo: `docs/papers/ri/agent_review_20260205_163118.md`
- Actionable hypotheses: calibrated RI probability, wind-change features, environment gating.

## RI literature update
- Added memo: `docs/papers/ri/agent_review_20260205_163313.md`
- Actionable hypotheses: calibrated RI probability, wind-change features, environment gating.

## RI ablation + calibration (2026-02-05)
- Run dir: `runs/ri_ablation_20260205_164234`
- `ri_logit` (full features): precision 0.227, recall 0.278, F1 0.250; wind MAE all/RI/non-RI = 19.98 / 32.31 / 18.76
- `ri_logit` (drop `dwind_6h`, `dwind_24h`): precision 0.160, recall 0.444, F1 0.235; wind MAE all/RI/non-RI = 22.89 / 28.07 / 22.38
- `ri_logit` (Platt + rate threshold): precision 0.183, recall 0.611, F1 0.282; wind MAE all/RI/non-RI = 21.94 / 19.66 / 22.16

## RI literature update
- Added memo: `docs/papers/ri/agent_review_20260205_173940.md`
- Actionable hypotheses: calibrated RI probability, wind-change features, environment gating.
