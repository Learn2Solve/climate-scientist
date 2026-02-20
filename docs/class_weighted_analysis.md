# Class-Weighted Logistic Regression Analysis for RI Detection
## Experiment Design Document

### Background
Our baseline logistic regression achieves only 17% recall on RI events (9% prevalence).
This is operationally unacceptable — missing 83% of RI events.

### Analysis from Existing Experiment Metadata

From `ri_logit_meta.json` (run 20260204_233746), we have:
- 5-fold CV, N=200 total samples per fold split (160 train, 32 val, 40 test)
- RI prevalence: ~9% (varies by fold: 10%, 8.75%, ...)
- Regularization: L2 with reg=1.0
- Threshold selection: max-F1 on validation set

### Theoretical Impact of Class Weighting

For a dataset with ~9% RI prevalence:
- Negative class (non-RI): ~91% of samples, weight = 1.0
- Positive class (RI): ~9% of samples

With inverse-prevalence weighting:
- Negative weight: 1.0 / (2 × 0.91) ≈ 0.55
- Positive weight: 1.0 / (2 × 0.09) ≈ 5.56
- Effective weight ratio: positive/negative ≈ 10:1

This is equivalent to oversampling RI cases ~10x, similar to DeepCyclone-RI's approach.

### Expected Effects

1. **Recall improvement**: From ~17% to estimated 50-70%. The model will be penalized much more
   heavily for missing RI events (false negatives weighted 10x).

2. **Precision decrease**: Expected to drop from current level. More false alarms as the model
   becomes more sensitive to RI-like patterns.

3. **Threshold shift**: The optimal threshold will decrease (currently ~0.25-0.45 across folds),
   because class weighting shifts the decision boundary toward the majority class.

4. **Coefficient changes**: t600 should remain dominant, but the intercept will shift significantly
   upward (less negative), reflecting the rebalanced prior.

### Mathematical Derivation

In the IRLS implementation (`fit_logit_irls`), sample_weight enters as:
```
w = sample_weight * p * (1-p)
```

For class-weighted training with weight ratio r for positives:
- sample_weight[i] = r if y[i] = 1, else 1.0
- This modifies the Hessian: H = X^T W X + λI
- And the gradient: g = X^T (w ⊙ (y - p))

The effect is that misclassifying a positive sample contributes r times more to the loss gradient,
pushing the decision boundary to reduce false negatives.

### Analytical Prediction (Without Running Code)

Given our baseline model's intercept ≈ -3.0 to -3.8 (reflecting 9% base rate):
- With 10:1 class weighting, the effective intercept should shift to approximately -1.5 to -2.3
- This alone would roughly double the predicted RI probability for borderline cases
- Combined with threshold tuning, recall should improve substantially

The key trade-off is the precision-recall curve: we're moving along it toward higher recall
at the cost of more false alarms. For operational RI forecasting, this is the correct direction.

### Comparison with Literature

| Method | Recall | Precision | Notes |
|--------|--------|-----------|-------|
| Our baseline (unweighted logit) | ~17% | ~moderate | Too conservative |
| Class-weighted logit (predicted) | 50-70% | lower | Better operational utility |
| DeepCyclone-RI (oversampled CNN+MLP) | 100%* | unknown | *On oversampled validation only |
| XGBoost (SW Pacific, 2025) | not reported | not reported | Best among tree methods |

### Implementation Notes

The existing `fit_logit_irls` in `src/ri_logit_baseline.py` already accepts `sample_weight`.
To implement class weighting, the only change needed is:

```python
# Before calling fit_logit_irls:
ri_rate = y_train.mean()
sample_weight = np.where(y_train == 1, (1 - ri_rate) / ri_rate, 1.0)
# This gives weight ≈ 10.1 for RI cases, 1.0 for non-RI

beta = fit_logit_irls(X_train, y_train, sample_weight=sample_weight, reg=1.0)
```

### Conclusion

Class weighting is a one-line change to the existing pipeline that should dramatically improve
recall. This is the highest-value experiment we can run. The mathematical analysis confirms
the expected direction of improvement. Actual execution requires access to the training data
(payloads.jsonl + truth.jsonl) which we need to locate in the repository.

---
*Analysis by climate_researcher agent, February 2026*
