#!/usr/bin/env python3
"""
Class-Weighted Logistic Regression Experiment for RI Detection.

Compares baseline (unweighted) vs class-weighted logistic regression
using the existing IRLS implementation.

Author: climate_researcher agent
Date: February 2026
"""
import sys
sys.path.insert(0, "src")

import json
import numpy as np
from pathlib import Path
from ri_logit_baseline import (
    extract_features, fit_logit_irls, sigmoid, standardize,
    f1_for_threshold, pick_threshold_max_f1, FEATURE_NAMES
)

# --- Load data ---
payloads_path = Path("sim_outputs/payloads.jsonl")
truth_path = Path("sim_outputs/truth.jsonl")

payloads = []
with open(payloads_path) as f:
    for line in f:
        payloads.append(json.loads(line))

truths = []
with open(truth_path) as f:
    for line in f:
        truths.append(json.loads(line))

print(f"Loaded {len(payloads)} payloads, {len(truths)} truths")

# --- Extract features and labels ---
X_all = np.array([extract_features(p) for p in payloads])
print(f"Feature matrix shape: {X_all.shape}")

# RI label: wind increase >= 30kt in 24h
RI_THRESHOLD = 30.0
y_all = np.zeros(len(truths), dtype=int)
for i, t in enumerate(truths):
    winds = t.get("wind", {})
    if "24" in winds:
        w24 = float(winds["24"])
        w0 = float(payloads[i].get("storm", {}).get("wind", 0))
        if (w24 - w0) >= RI_THRESHOLD:
            y_all[i] = 1

ri_count = int(y_all.sum())
ri_rate = y_all.mean()
print(f"RI events: {ri_count}/{len(y_all)} = {ri_rate:.3f}")

# --- K-Fold CV ---
K = 5
SEED = 42
rng = np.random.RandomState(SEED)
indices = np.arange(len(y_all))
rng.shuffle(indices)
folds = np.array_split(indices, K)

def run_cv(use_class_weight=False, weight_ratio=None):
    """Run K-fold CV with or without class weighting."""
    all_probs = np.zeros(len(y_all))
    all_preds = np.zeros(len(y_all), dtype=int)
    fold_results = []
    
    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        # Standardize
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        X_train_s = (X_train - mu) / sigma
        X_test_s = (X_test - mu) / sigma
        
        # Add intercept
        X_train_s = np.column_stack([np.ones(len(X_train_s)), X_train_s])
        X_test_s = np.column_stack([np.ones(len(X_test_s)), X_test_s])
        
        # Class weights
        sw = None
        if use_class_weight:
            train_ri_rate = y_train.mean()
            if weight_ratio is not None:
                sw = np.where(y_train == 1, weight_ratio, 1.0)
            else:
                sw = np.where(y_train == 1, (1 - train_ri_rate) / max(train_ri_rate, 0.01), 1.0)
        
        beta = fit_logit_irls(X_train_s, y_train, sample_weight=sw, reg=1.0)
        
        # Predict
        probs_test = sigmoid(X_test_s @ beta)
        all_probs[test_idx] = probs_test
        
        # Use validation split for threshold (use last 20% of train as val)
        val_size = len(train_idx) // 5
        val_idx_local = train_idx[-val_size:]
        X_val_s = (X_all[val_idx_local] - mu) / sigma
        X_val_s = np.column_stack([np.ones(len(X_val_s)), X_val_s])
        probs_val = sigmoid(X_val_s @ beta)
        y_val = y_all[val_idx_local]
        
        thr = pick_threshold_max_f1(probs_val, y_val)
        preds_test = (probs_test >= thr).astype(int)
        all_preds[test_idx] = preds_test
        
        # Fold metrics
        tp = int(np.sum((preds_test == 1) & (y_test == 1)))
        fp = int(np.sum((preds_test == 1) & (y_test == 0)))
        fn = int(np.sum((preds_test == 0) & (y_test == 1)))
        tn = int(np.sum((preds_test == 0) & (y_test == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        fold_results.append({
            "fold": k, "thr": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "intercept": float(beta[0]),
            "top_coefs": {FEATURE_NAMES[j]: float(beta[j+1]) for j in range(min(len(FEATURE_NAMES), len(beta)-1))}
        })
    
    # Aggregate metrics
    tp_total = int(np.sum((all_preds == 1) & (y_all == 1)))
    fp_total = int(np.sum((all_preds == 1) & (y_all == 0)))
    fn_total = int(np.sum((all_preds == 0) & (y_all == 1)))
    tn_total = int(np.sum((all_preds == 0) & (y_all == 0)))
    prec_total = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec_total = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_total = 2 * prec_total * rec_total / (prec_total + rec_total) if (prec_total + rec_total) > 0 else 0.0
    
    return {
        "aggregate": {
            "tp": tp_total, "fp": fp_total, "fn": fn_total, "tn": tn_total,
            "precision": prec_total, "recall": rec_total, "f1": f1_total,
        },
        "folds": fold_results
    }

# --- Run experiments ---
print("\n" + "="*60)
print("EXPERIMENT 1: BASELINE (unweighted)")
print("="*60)
baseline = run_cv(use_class_weight=False)
agg = baseline["aggregate"]
print(f"  Precision: {agg['precision']:.3f}")
print(f"  Recall:    {agg['recall']:.3f}")
print(f"  F1:        {agg['f1']:.3f}")
print(f"  TP={agg['tp']} FP={agg['fp']} FN={agg['fn']} TN={agg['tn']}")
for fr in baseline["folds"]:
    print(f"  Fold {fr['fold']}: thr={fr['thr']:.3f} P={fr['precision']:.3f} R={fr['recall']:.3f} F1={fr['f1']:.3f} intercept={fr['intercept']:.2f}")

print("\n" + "="*60)
print("EXPERIMENT 2: CLASS-WEIGHTED (auto inverse prevalence)")
print("="*60)
weighted = run_cv(use_class_weight=True)
agg = weighted["aggregate"]
print(f"  Precision: {agg['precision']:.3f}")
print(f"  Recall:    {agg['recall']:.3f}")
print(f"  F1:        {agg['f1']:.3f}")
print(f"  TP={agg['tp']} FP={agg['fp']} FN={agg['fn']} TN={agg['tn']}")
for fr in weighted["folds"]:
    print(f"  Fold {fr['fold']}: thr={fr['thr']:.3f} P={fr['precision']:.3f} R={fr['recall']:.3f} F1={fr['f1']:.3f} intercept={fr['intercept']:.2f}")

# Also try several fixed weight ratios
for ratio in [3.0, 5.0, 10.0, 20.0]:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: CLASS-WEIGHTED (ratio={ratio}:1)")
    print("="*60)
    result = run_cv(use_class_weight=True, weight_ratio=ratio)
    agg = result["aggregate"]
    print(f"  Precision: {agg['precision']:.3f}")
    print(f"  Recall:    {agg['recall']:.3f}")
    print(f"  F1:        {agg['f1']:.3f}")
    print(f"  TP={agg['tp']} FP={agg['fp']} FN={agg['fn']} TN={agg['tn']}")

# --- Feature importance comparison ---
print("\n" + "="*60)
print("FEATURE IMPORTANCE COMPARISON (avg |coef| across folds)")
print("="*60)
print(f"{'Feature':<12} {'Baseline':>10} {'Weighted':>10} {'Change':>10}")
print("-" * 44)
for j, fname in enumerate(FEATURE_NAMES):
    base_avg = np.mean([abs(fr["top_coefs"].get(fname, 0)) for fr in baseline["folds"]])
    wt_avg = np.mean([abs(fr["top_coefs"].get(fname, 0)) for fr in weighted["folds"]])
    change = wt_avg - base_avg
    print(f"{fname:<12} {base_avg:>10.3f} {wt_avg:>10.3f} {change:>+10.3f}")

print("\nDone.")
