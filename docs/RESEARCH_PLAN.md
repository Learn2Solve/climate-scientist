# Rapid Intensification Research Plan

**Agent**: climate_researcher
**Created**: 2025-02-19
**Topic**: Rapid Intensification (RI) of Tropical Cyclones

---

## 1. Problem Statement

Rapid Intensification (RI) — defined as ≥30kt wind speed increase in 24 hours — is the most dangerous and least predictable aspect of tropical cyclone behavior. Our experiments show that LLM-based forecasters have **zero RI detection skill**: they achieve ~91% accuracy by simply never predicting RI (base rate = 9%).

**Central Question**: Can we build a hybrid system that combines LLM forecasting with dedicated RI classifiers to dramatically improve RI detection while maintaining overall forecast quality?

---

## 2. Current State of Knowledge

### 2.1 Experimental Results (completed)
- **4 LLM models evaluated** on 200 simulated hurricane samples (sim_outputs/)
- **RI base rate**: 18/200 = 9%
- **All models fail**: 0% recall (DeepSeek, TTM, Codex), 11% recall (Claude, F1=0.17)
- **RI wind MAE disparity**: 31-37kt during RI vs 12-18kt during non-RI (2-3x ratio)
- Models exhibit strong regression-to-mean bias, missing the entire RI signal

### 2.2 Available Infrastructure
- `ri_metrics.py`: RI classification from intensity forecasts
- `ri_logit_baseline.py`: SHIPS-style logistic regression for RI probability
- `baselines.py`: Persistence/kinematic baselines
- HURDAT2 data (1980-2022) + simulated hurricane time series
- Multiple LLM prediction sets (DeepSeek, TTM, Claude, Codex)

### 2.3 Literature Context
- XGBoost classifiers achieve meaningful RI/non-RI separation using environmental features (location, intensity, RH850) — Atmosphere 2025
- Spatiotemporal transformer models represent SOTA for TC intensity — npj Clim. Atmos. Sci. 2025
- Operational SHIPS-RII achieves ~20-40% POD for RI events

---

## 3. Research Hypotheses

### H1: SHIPS-style logistic regression baseline outperforms all LLMs at RI detection
**Rationale**: Even a simple logistic model trained on environmental features (wind, pressure, lat, lon, shear, RH) with class balancing should beat 0% recall.
**Test**: Run `ri_logit_baseline.py` on sim_outputs/ and compare F1/recall to LLM results.
**Priority**: HIGH (next experiment)

### H2: LLM forecast errors are systematically larger for RI events due to mean-regression bias
**Rationale**: LLMs trained on general text minimize expected loss, which for rare events means predicting the population mean.
**Test**: Already confirmed — RI MAE is 2-3x non-RI MAE. Quantify the intensity change bias (predicted delta vs actual delta during RI).
**Priority**: COMPLETED (partially)

### H3: A hybrid "LLM forecast + logistic RI classifier" can improve both RI detection and overall MAE
**Rationale**: Use the logistic model to flag RI-probable samples, then apply an intensity correction to the LLM forecast for those samples.
**Test**: Build a simple ensemble: if P(RI) > threshold, adjust predicted wind upward by expected RI delta.
**Priority**: MEDIUM (after H1)

### H4: Feature importance for RI in simulated data matches known physical predictors
**Rationale**: SHIPS-RII predictors (SST, ocean heat content, wind shear, moisture, current intensity) should dominate.
**Test**: Extract feature weights from the logistic model and compare to literature.
**Priority**: MEDIUM

---

## 4. Experiment Queue

| # | Experiment | Script | Status | Priority |
|---|-----------|--------|--------|----------|
| 1 | RI metrics for all LLM models | ri_metrics.py | ✅ DONE | - |
| 2 | Logistic regression RI baseline | ri_logit_baseline.py | NEXT | HIGH |
| 3 | RI intensity change bias analysis | custom analysis | PLANNED | HIGH |
| 4 | Hybrid LLM+logistic ensemble | custom | PLANNED | MEDIUM |
| 5 | Feature importance analysis | from logistic model | PLANNED | MEDIUM |
| 6 | HURDAT2 RI rate and characteristics | data_prep.py + analysis | PLANNED | LOW |
| 7 | Fetch spatiotemporal transformer paper details | web_fetch | PLANNED | LOW |

---

## 5. Success Metrics

- **Primary**: Achieve RI recall > 30% with precision > 20% (F1 > 0.24) — beating all current LLMs
- **Secondary**: Reduce RI wind MAE from ~33kt to < 25kt
- **Stretch**: Build hybrid system achieving F1 > 0.40 for RI detection

---

## 6. Resource Budget

- $50 API budget, ~$0 spent so far
- Primary cost: LLM inference for new experiments
- Literature searches: ~$0.01-0.02 each
- Prioritize local computation (logistic baseline, metrics) over API calls
