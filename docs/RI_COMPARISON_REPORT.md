# RI Detection: LLM Forecasters vs. Logistic Regression Baseline

**Date**: 2025-02-19  
**Agent**: climate_researcher  
**Dataset**: sim_outputs/ (200 simulated hurricane samples, 18 RI events, 9% base rate)  
**RI Definition**: Wind increase ≥30kt in 24 hours (standard NHC definition)

---

## Summary

A simple SHIPS-style logistic regression with 13 environmental features **outperforms all four LLM forecasters** at detecting rapid intensification events, achieving F1=0.25 compared to the best LLM (Claude-Opus-4.5) at F1=0.17.

---

## Results Table

| Model | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy | RI MAE | Non-RI MAE | Overall MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **5** | 17 | 13 | 165 | 22.7% | **27.8%** | **0.250** | 85.0% | 32.3kt | 18.8kt | 20.0kt |
| Claude-Opus-4.5 | 2 | 3 | 16 | 179 | 40.0% | 11.1% | 0.174 | 90.5% | 31.2kt | 16.9kt | 18.2kt |
| DeepSeek-chat | 0 | 0 | 18 | 182 | N/A | 0.0% | N/A | 91.0% | 31.5kt | 17.7kt | 18.9kt |
| TTM-aligned | 0 | 2 | 18 | 180 | 0.0% | 0.0% | N/A | 90.0% | 32.9kt | 12.5kt | 14.3kt |
| Codex-gpt-5.2 | 0 | 0 | 18 | 182 | N/A | 0.0% | N/A | 91.0% | 37.1kt | 16.4kt | 18.2kt |

---

## Key Findings

### 1. LLMs Have Near-Zero RI Skill
All four LLM-based forecasters effectively predict "no RI" for every sample. Their ~91% accuracy merely reflects the 9% RI base rate — equivalent to a trivial "always predict no-RI" classifier. Only Claude-Opus-4.5 detected any RI events (2 of 18), with very low recall (11.1%).

### 2. Logistic Regression Detects RI Events LLMs Miss
The logistic model correctly identified 5 of 18 RI events (27.8% recall), catching 2.5x more RI events than the best LLM. This comes at the cost of 17 false positives, yielding lower precision (22.7%) and accuracy (85%).

### 3. The Accuracy Paradox
The LLMs' higher accuracy (90-91%) is **misleading**. In rare-event detection, accuracy is dominated by the majority class. The logistic model's lower accuracy (85%) reflects genuine attempts to predict RI events, which necessarily produces some false alarms.

### 4. Wind MAE During RI Events Remains High Everywhere
All models show RI wind MAE of ~31-37kt, roughly equal to the RI threshold itself (30kt). This means models are on average missing the *entire* intensification signal during RI events. The logistic model does not improve this — it changes the binary classification but the underlying wind prediction still underpredicts.

### 5. TTM-aligned: Best Overall MAE, Worst RI Ratio
TTM-aligned achieves the best overall wind MAE (14.3kt) and non-RI MAE (12.5kt), but has the worst RI-to-non-RI MAE ratio (2.63x). It's highly optimized for the common case at the expense of rare events.

---

## Interpretation

The results demonstrate a fundamental limitation of using LLMs as generic intensity forecasters: they optimize for expected error across all samples, which for rare events (9% base rate) means learning to predict "no change." This is the well-known **regression to the mean** problem in intensity forecasting.

A dedicated classifier using physically-motivated features (wind speed, pressure, moisture, shear, location, recent intensity trends) can detect RI events that LLMs completely miss, even with a simple logistic model and only 200 training samples.

---

## Logistic Model Features (13 predictors)
1. `wind0` — Current wind speed
2. `pressure0` — Current pressure  
3. `lat0` — Latitude
4. `lon0` — Longitude
5. `shear` — Wind shear
6. `rh` — Relative humidity (850 hPa)
7. `t600` — Temperature at 600 hPa
8. `u850` — Zonal wind at 850 hPa
9. `v850` — Meridional wind at 850 hPa
10. `vp` — Vapor pressure
11. `p_env` — Environmental pressure
12. `dwind_6h` — 6-hour wind change (recent trend)
13. `dwind_24h` — 24-hour wind change (longer trend)

---

## Next Steps

1. **Hybrid ensemble**: Combine LLM base forecast with logistic RI probability to adjust predictions when RI is likely
2. **Intensity correction**: When P(RI) > threshold, boost predicted wind by expected RI delta
3. **Feature importance**: Extract logistic regression coefficients to identify dominant RI predictors
4. **Threshold tuning**: Explore precision-recall tradeoff for operational utility
5. **HURDAT2 validation**: Test on real historical TC data, not just simulations

---

## Reproducibility

```bash
# Run logistic baseline
uv run --no-project python src/ri_logit_baseline.py \
  --payloads sim_outputs/payloads.jsonl \
  --truth sim_outputs/truth.jsonl \
  --out docs/ri_logit_predictions.jsonl \
  --out-meta docs/ri_logit_meta.json \
  --ri-lead-hours 24 --ri-threshold-kt 30 --kfold 5 --calibrate

# Compute RI metrics
uv run --no-project python src/ri_metrics.py \
  --payloads sim_outputs/payloads.jsonl \
  --truth sim_outputs/truth.jsonl \
  --predictions docs/ri_logit_predictions.jsonl \
  --lead-hours 24 --threshold-kt 30 \
  --out-json docs/ri_metrics_logit.json
```
