# Rapid Intensification Research Status
## Climate Research Agent — Updated 2025-02-20

---

## 1. Research Question

**How well can we predict tropical cyclone rapid intensification (RI) using different
modeling approaches, and what are the key physical predictors?**

RI Definition: ≥30 kt (15 m/s) increase in maximum sustained wind speed within 24 hours.

---

## 2. Data Available

### HURDAT2 Toy Dataset (hurdat2_llm_toy/all_samples.parquet)
- **12,467 samples** from 1980-2022 Atlantic tropical cyclones
- Features: `last_wind`, `target_wind`, `last_pressure`, `target_pressure`, `last_lat`, `last_lon`, `target_lat`, `target_lon`, `season`
- Split: Train (1980-2014, 9475), Val (2015-2018, 1478), Test (2019-2022, 1514)
- **Limitation**: Only track/intensity data. No environmental fields (SST, shear, humidity, OHC).

### Simulated Data (sim_outputs/)
- 200 samples with 13 features including environmental variables (shear, RH, T600, u850, v850, vp, p_env)
- 18 RI events (9% base rate)
- Used for baseline model development

---

## 3. Experimental Results

### A. Persistence Baseline (HURDAT2 Real Data)
| Split | Samples | Track MAE (km) | Wind MAE (kt) |
|-------|---------|-----------------|----------------|
| Train | 9,475 | 521.3 | 11.5 |
| Val | 1,478 | 507.8 | 11.4 |
| Test | 1,514 | 508.7 | 12.0 |

**Interpretation**: Average wind change is ~12 kt, well below the 30 kt RI threshold. Persistence (assume no change) is a strong baseline for typical storms but has 0% RI recall.

### B. Logistic Regression with Calibration (Simulated Data)
- Features: wind0, pressure0, lat0, lon0, shear, rh, t600, u850, v850, vp, p_env, dwind_6h, dwind_24h
- 5-fold CV with Platt calibration
- **Test Results**: Precision=22.7%, Recall=27.8%, F1=0.25
- TP=5, FP=17, FN=13, TN=165
- Wind MAE: 20.0 kt overall, 32.3 kt for RI cases

### C. ri_gate Threshold Baseline (Simulated Data, from prior work)
- Simple threshold-based RI detection
- F1=0.32, Recall=61.1% (beats logistic regression)

---

## 4. Key Findings from Literature

### 4.1 XGBoost outperforms logistic regression for RI classification
**Source**: SW Pacific TC RI study (Atmosphere, April 2025, doi:10.3390/atmos16040456)
- XGBoost with 10-fold CV achieved highest accuracy, lowest false alarm, highest AUC
- Most important features: **longitude, initial intensity latitude, extent of initial intensity, RH at 850 hPa**
- 324 TCs, 81 RI TCs (25% experienced RI at least once)

**Implication**: Our logistic regression F1=0.25 is likely improvable with XGBoost. Humidity data (missing from HURDAT2) is critical.

### 4.2 Non-classic RI under strong shear is the hardest prediction challenge
**Source**: Hurricane Matthew (2016) study (Atmosphere, March 2024, doi:10.3390/atmos15040395)
- Matthew: Cat 1 → Cat 5 in 24h under STRONG vertical shear
- 4DEnVar data assimilation of inner-core observations improved RI prediction
- Key mechanism: shear-relative convective structure → upshear-left subsidence warming
- Inner-core observations are critical (not available in HURDAT2)

**Implication**: Track/intensity-only features have a ceiling. Environmental and structural observations needed for the hardest RI cases.

---

## 5. Gap Analysis

| Gap | Impact | Feasibility |
|-----|--------|-------------|
| No environmental features in HURDAT2 | High — RH_850, shear, SST are top RI predictors | Medium — could merge with ERA5 reanalysis |
| Only tested logistic regression | Medium — XGBoost likely better | High — easy to implement |
| No RI analysis on real HURDAT2 yet | High — all RI metrics from simulated data | High — script ready (docs/hurdat2_ri_analysis.py) |
| No feature engineering on HURDAT2 | Medium — wind trend, motion, lat/lon features could help | High |
| No deep learning baseline | Low priority first — need more features | Low — small dataset |

---

## 6. Next Steps (Priority Order)

1. **Run HURDAT2 RI analysis on real data** — Copy docs/hurdat2_ri_analysis.py to src/, run it, get first real-data RI metrics
2. **Add feature engineering** — Wind trend (delta), translation speed, distance from coast, season/month
3. **Implement XGBoost** — Based on literature, should beat logistic regression significantly
4. **Merge ERA5 environmental data** — Would add SST, shear, RH_850 (the most important predictors per literature)
5. **Decade trend analysis** — Is RI becoming more common with warming SSTs?

---

## 7. Scripts & Artifacts

| File | Purpose | Status |
|------|---------|--------|
| `src/hurdat2_baseline.py` | Persistence baseline on HURDAT2 | ✅ Run, results saved |
| `src/ri_logit_baseline.py` | Logistic RI classifier on sim data | ✅ Run, F1=0.25 |
| `src/ri_metrics.py` | RI evaluation metrics | ✅ Working |
| `docs/hurdat2_ri_analysis.py` | Real-data RI analysis (needs copy to src/) | 📝 Written, not yet run |
| `docs/hurdat2_persistence_metrics.json` | Persistence baseline results | ✅ Generated |
| `docs/ri_logit_meta.json` | Logistic model metadata | ✅ Generated |
| `docs/ri_metrics_logit_fresh.json` | Logistic RI metrics | ✅ Generated |

---

## 8. Research Hypotheses

### H1: XGBoost will achieve F1 > 0.35 on HURDAT2 RI detection with only track features
**Rationale**: Literature shows XGBoost beats logistic regression; geographic features (lat/lon) are important.
**Test**: Implement XGBoost in hurdat2_ri_analysis.py, compare with logistic baseline.

### H2: RI rate has increased since 2000 in Atlantic basin
**Rationale**: Warming SSTs should increase RI probability. Prior studies suggest upward trend.
**Test**: Compute RI rate by decade from HURDAT2 data.

### H3: Wind trend features (6h and 24h prior changes) significantly improve RI prediction
**Rationale**: Storms that are already intensifying are more likely to undergo RI.
**Test**: Add delta_wind features to model, measure F1 improvement.

---

*Agent: climate_researcher | Topic: rapid intensification | Budget: $0.00/$50.00 spent*
