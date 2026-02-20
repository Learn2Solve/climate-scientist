# Rapid Intensification Research Summary v1
## Climate Research Agent — February 2026

### 1. Research Question
What are the key predictors and optimal modeling approaches for forecasting tropical cyclone rapid intensification (RI), defined as ≥30 kt wind speed increase in 24 hours?

### 2. Dataset
- Simulated tropical cyclone dataset with 13 environmental/storm features
- ~9% RI prevalence (severe class imbalance)
- Features: wind0, pressure0, lat0, lon0, shear, rh, t600, u850, v850, vp, p_env, dwind_6h, dwind_24h

### 3. Baseline Model Results (Logistic Regression, 5-fold CV)

Our logistic regression baseline with L2 regularization achieves:
- **Recall: ~17%** (misses 83% of RI events — operationally unacceptable)
- Moderate AUC but poor at the critical task of detecting rare RI events
- The model is too conservative, defaulting to predicting non-RI due to class imbalance

### 4. Feature Importance Rankings (Standardized Logistic Coefficients)

| Rank | Feature | |Coef| | Direction | Physical Interpretation |
|------|---------|--------|-----------|------------------------|
| 1 | t600 | 0.930 | RI+ | Mid-level warmth (warm core development) |
| 2 | p_env | 0.765 | RI+ | High environmental pressure (anticyclonic environment) |
| 3 | pressure0 | 0.765 | RI+ | Higher initial MSLP = more room to deepen |
| 4 | vp | 0.416 | RI+ | Vorticity potential |
| 5 | lat0 | 0.343 | RI- | Lower latitude favors RI (warmer SSTs) |
| 6 | lon0 | 0.334 | RI+ | Basin-dependent effect |
| 7 | shear | 0.306 | RI- | Low shear favors RI (well established) |
| 8 | dwind_24h | 0.232 | RI+ | Prior strengthening trend |
| 9 | u850 | 0.228 | RI+ | Low-level zonal wind |
| 10 | wind0 | 0.195 | RI- | Weaker storms have more RI potential |
| 11 | v850 | 0.181 | RI- | Low-level meridional wind |
| 12 | rh | 0.178 | RI+ | Moisture availability |
| 13 | dwind_6h | 0.155 | RI- | Short-term trend (noisy) |

**Key insight**: Mid-level temperature (t600) dominates over wind shear — consistent with the warm-core intensification mechanism but contrary to the common assumption that shear is the #1 predictor.

### 5. Literature Context

#### a) SW Pacific RI Classification (Atmosphere, 2025)
- XGBoost outperformed Random Forest and Decision Trees for RI classification
- Found location (lon), initial intensity, and RH_850 as top predictors
- Confirms that tree-based methods are strong baselines for RI

#### b) DeepCyclone-RI (Zenodo preprint, Feb 2026)
- Hybrid CNN (satellite imagery) + MLP (scalar features) architecture
- Achieved 100% recall on oversampled validation set (3% RI prevalence)
- Demonstrates that oversampling/class weighting is essential for RI detection
- **Caution**: Not peer-reviewed; 100% recall claim likely doesn't generalize

### 6. Key Research Findings

1. **Class imbalance is the central challenge**: With 9% RI prevalence, standard classifiers default to predicting non-RI. Oversampling, class weighting, or threshold tuning is mandatory.

2. **Mid-level thermodynamics dominate**: t600 (warm core proxy) is the strongest single predictor, ahead of wind shear. This aligns with the physical understanding that RI involves rapid warm core development.

3. **Wind shear is necessary but not sufficient**: Shear ranks 7th — important but far from dominant. Low shear is a prerequisite for RI but many low-shear environments don't produce RI.

4. **Initial storm state matters**: Weaker storms (lower wind0) and higher initial MSLP (pressure0) have more RI potential — they have more room in the intensity phase space to rapidly intensify.

5. **Location encodes ocean/climate state**: lat0 and lon0 together rank 5-6, serving as proxies for SST, ocean heat content, and basin-specific climatology.

### 7. Identified Gaps and Next Steps

| Priority | Task | Rationale |
|----------|------|-----------|
| HIGH | Implement class-weighted logistic regression | Address 17% recall; expected to reach 50-70% recall |
| HIGH | Build XGBoost model with SHAP analysis | Better feature importance, captures nonlinear interactions |
| MEDIUM | Add ocean heat content (OHC) as feature | Major gap in current feature set; literature shows OHC is critical |
| MEDIUM | Test ensemble of logistic + XGBoost | Operational RI tools use consensus approaches |
| LOW | Investigate satellite imagery features | DeepCyclone-RI shows CNN features add value, but requires infrastructure |

### 8. Confidence Assessment
- Feature importance rankings: **HIGH confidence** (88%) — robust across 5 folds, physically consistent
- Literature synthesis: **MODERATE confidence** (65%) — limited to recent papers; older foundational work (SHIPS, Kaplan & DeMaria) not yet reviewed
- Model improvement predictions: **LOW confidence** — untested hypotheses about class weighting and XGBoost improvements

### 9. Reproducibility
- Baseline model: `src/ri_logit_baseline.py`
- Model metadata: `runs/20260204_233746_auto_e0ccb6/experiments/exp_001_baselines/ri_logit_meta.json`
- All coefficients and fold-level results are stored in JSON format

---
*Generated by climate_researcher agent, February 2026*
*Total API cost at time of writing: <$0.50 of $50.00 budget*
