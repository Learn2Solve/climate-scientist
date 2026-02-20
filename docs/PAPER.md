# Climate Scientist Agent for Hurricane Forecasting

## Abstract

_TODO: Summarize the claim and results._

## Method

We study a workflow-driven **climate scientist agent** that can run experiments,
log an auditable evidence chain, and update a paper draft based on computed metrics.

## Experiments

- Task: hurricane track (km) + intensity (kt) at 24/48/72h.
- Metrics: track MAE (great-circle), wind MAE, valid JSON rate.

## Results

<!-- AUTO_RESULTS_TABLE_START -->
_Latest auto-run: `20260205_172431_auto_3ca3a5` (see `../runs/20260205_172431_auto_3ca3a5/metrics.json`)._

| Experiment | Model | Samples | Valid JSON | Track@24 (km) | Track@48 (km) | Track@72 (km) | Wind@24 (kt) | Wind@48 (kt) | Wind@72 (kt) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_001_baselines | persistence | 200 | 1.000 | 7402.80 | 8957.67 | 8745.36 | 19.04 | 23.66 | 22.34 |
| exp_001_baselines | kinematic | 200 | 1.000 | 9920.98 | 10300.40 | 10125.84 | 19.04 | 23.66 | 22.34 |
| exp_001_baselines | trend | 200 | 1.000 | 9920.98 | 10300.40 | 10125.84 | 42.12 | 62.77 | 65.60 |
| exp_001_baselines | ri_gate | 200 | 1.000 | 7402.80 | 8957.67 | 8745.36 | 17.66 | 21.04 | 22.22 |
| exp_001_baselines | ri_logit | 200 | 1.000 | 7402.80 | 8957.67 | 8745.36 | 19.98 | 23.99 | 22.46 |
| exp_002_llm_baseline | deepseek-chat | 50 | 1.000 | 7274.00 | 8905.04 | 8935.69 | 17.44 | 19.37 | 21.57 |
| exp_003_guided_persistence | guided_deepseek-chat | 50 | 1.000 | 7281.86 | 8776.23 | 8989.23 | 17.88 | 18.77 | 24.75 |
<!-- AUTO_RESULTS_TABLE_END -->
## Rapid Intensification (RI)

<!-- AUTO_RI_TABLE_START -->
_RI definition: ΔV ≥ 30 kt over 24h; classification derived from predicted wind._

| Experiment | Model | Samples | Truth RI | Valid Wind | Precision | Recall | F1 | Wind MAE (RI) | Wind MAE (non-RI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_001_baselines | persistence | 200 | 0.090 | 1.000 |  | 0.000 |  | 39.17 | 17.05 |
| exp_001_baselines | kinematic | 200 | 0.090 | 1.000 |  | 0.000 |  | 39.17 | 17.05 |
| exp_001_baselines | trend | 200 | 0.090 | 1.000 | 0.019 | 0.056 | 0.029 | 56.72 | 40.68 |
| exp_001_baselines | ri_gate | 200 | 0.090 | 1.000 | 0.220 | 0.611 | 0.324 | 20.15 | 17.41 |
| exp_001_baselines | ri_logit | 200 | 0.090 | 1.000 | 0.227 | 0.278 | 0.250 | 32.31 | 18.76 |
| exp_002_llm_baseline | deepseek-chat | 50 | 0.120 | 1.000 |  | 0.000 |  | 31.07 | 15.58 |
| exp_003_guided_persistence | guided_deepseek-chat | 50 | 0.120 | 1.000 |  | 0.000 |  | 34.97 | 15.55 |
<!-- AUTO_RI_TABLE_END -->


## Discussion / Limitations

- Fairness: different models may consume different input modalities.
- Robustness: LLM output validity requires strict schemas and verifiers.

## Reproducibility

Each run writes an auditable bundle under `runs/<run_id>/` including:
`plan.json`, `tool_calls.jsonl`, `env.lock`, `data_manifest.json`, `metrics.json`, `report.md`.

