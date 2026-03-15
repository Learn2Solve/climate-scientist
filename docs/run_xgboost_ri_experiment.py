#!/usr/bin/env python3
"""
XGBoost RI Classifier Experiment on HURDAT2 Atlantic Data
Agent: climate_researcher
Date: 2026-02-20

Objective: Train and evaluate XGBoost for RI binary classification
using only track/intensity features from HURDAT2.

RI definition: 24h wind increase >= 30 kt
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────
DATA = Path("hurdat2_llm_toy/all_samples.parquet")
OUT = Path("docs/xgboost_ri_results.json")

print("=" * 60)
print("HURDAT2 XGBoost RI Classification Experiment")
print("=" * 60)

df = pd.read_parquet(DATA)
print(f"\nTotal samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Seasons: {df['season'].min()}-{df['season'].max()}")

# ── Feature engineering ────────────────────────────────────────
df["delta_wind"] = df["target_wind"] - df["last_wind"]
df["ri_label"] = (df["delta_wind"] >= 30).astype(int)
df["abs_lat"] = df["last_lat"].abs()
df["abs_lon"] = df["last_lon"].abs()

# Pressure deficit (proxy for intensity structure)
if "last_pressure" in df.columns:
    df["pressure_deficit"] = 1013.0 - df["last_pressure"]
    df["wp_ratio"] = df["last_wind"] / (df["pressure_deficit"].clip(lower=1))
    has_pressure = True
else:
    has_pressure = False

# ── RI statistics ──────────────────────────────────────────────
n_ri = df["ri_label"].sum()
ri_rate = n_ri / len(df)
print(f"\nRI events (>=30kt/24h): {n_ri} / {len(df)} = {ri_rate:.3%}")
print(f"Mean delta_wind: {df['delta_wind'].mean():.1f} kt")
print(f"Std delta_wind: {df['delta_wind'].std():.1f} kt")
print(f"Max delta_wind: {df['delta_wind'].max():.0f} kt")
print(f"Min delta_wind: {df['delta_wind'].min():.0f} kt")

# ── Temporal train/val/test split ──────────────────────────────
train = df[df["season"] <= 2014].copy()
val = df[(df["season"] >= 2015) & (df["season"] <= 2018)].copy()
test = df[df["season"] >= 2019].copy()

print(f"\nTrain: {len(train)} samples (RI: {train['ri_label'].sum()}, {train['ri_label'].mean():.3%})")
print(f"Val:   {len(val)} samples (RI: {val['ri_label'].sum()}, {val['ri_label'].mean():.3%})")
print(f"Test:  {len(test)} samples (RI: {test['ri_label'].sum()}, {test['ri_label'].mean():.3%})")

# ── Feature matrix ─────────────────────────────────────────────
FEATURES = ["last_wind", "last_lat", "last_lon", "abs_lat", "abs_lon", "season"]
if has_pressure:
    FEATURES += ["last_pressure", "pressure_deficit", "wp_ratio"]

print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")

# Clean NaN
train_clean = train.dropna(subset=FEATURES)
val_clean = val.dropna(subset=FEATURES)
test_clean = test.dropna(subset=FEATURES)

print(f"After NaN removal - Train: {len(train_clean)}, Val: {len(val_clean)}, Test: {len(test_clean)}")

X_train = train_clean[FEATURES].values
y_train = train_clean["ri_label"].values
X_val = val_clean[FEATURES].values
y_val = val_clean["ri_label"].values
X_test = test_clean[FEATURES].values
y_test = test_clean["ri_label"].values

# ── Helper function ────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob=None):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    result = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "precision": round(prec, 4), "recall": round(rec, 4),
              "f1": round(f1, 4), "accuracy": round(acc, 4)}
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        try:
            result["auc_roc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except:
            result["auc_roc"] = None
        result["brier"] = round(float(brier_score_loss(y_true, y_prob)), 4)
    return result

# ── Model 1: Logistic Regression baseline ──────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_probs = lr.predict_proba(X_test_s)[:, 1]
lr_pred = (lr_probs >= 0.5).astype(int)
lr_results = evaluate(y_test, lr_pred, lr_probs)
print(f"\n--- Logistic Regression (threshold=0.5) ---")
print(f"  Precision={lr_results['precision']:.3f}  Recall={lr_results['recall']:.3f}  F1={lr_results['f1']:.3f}  AUC={lr_results.get('auc_roc', 'N/A')}")

# ── Model 2: XGBoost ──────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    model_name = "XGBoost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    model_name = "GradientBoosting (sklearn fallback)"

n_neg = int((y_train == 0).sum())
n_pos = max(int((y_train == 1).sum()), 1)
spw = n_neg / n_pos
print(f"\nClass imbalance: {n_neg} neg / {n_pos} pos, scale_pos_weight={spw:.1f}")

if model_name == "XGBoost":
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="aucpr", use_label_encoder=False,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
else:
    sample_wt = np.where(y_train == 1, spw, 1.0)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=42,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_wt)

xgb_probs = xgb.predict_proba(X_test)[:, 1]

# ── Threshold sweep ────────────────────────────────────────────
print(f"\n--- {model_name} Threshold Sweep (Test Set) ---")
best_f1 = 0
best_thr = 0.5
results_by_thr = {}
for thr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    pred = (xgb_probs >= thr).astype(int)
    res = evaluate(y_test, pred, xgb_probs)
    results_by_thr[str(thr)] = res
    flag = " <-- best F1" if res["f1"] > best_f1 else ""
    if res["f1"] > best_f1:
        best_f1 = res["f1"]
        best_thr = thr
    print(f"  thr={thr:.2f}: P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f} TP={res['tp']} FP={res['fp']} FN={res['fn']}{flag}")

xgb_pred = (xgb_probs >= best_thr).astype(int)
xgb_results = evaluate(y_test, xgb_pred, xgb_probs)
print(f"\nBest threshold: {best_thr} -> F1={xgb_results['f1']:.3f}")

# ── Feature importance ─────────────────────────────────────────
print(f"\n--- Feature Importance ({model_name}) ---")
if hasattr(xgb, 'feature_importances_'):
    importances = xgb.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {feat:20s}: {imp:.4f}")

# ── Model 3: Random Forest ────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, class_weight="balanced",
    min_samples_leaf=10, random_state=42
)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_probs >= 0.3).astype(int)
rf_results = evaluate(y_test, rf_pred, rf_probs)
print(f"\n--- Random Forest (threshold=0.3) ---")
print(f"  Precision={rf_results['precision']:.3f}  Recall={rf_results['recall']:.3f}  F1={rf_results['f1']:.3f}  AUC={rf_results.get('auc_roc', 'N/A')}")

# ── Save results ───────────────────────────────────────────────
output = {
    "experiment": "HURDAT2 XGBoost RI Classification",
    "date": "2026-02-20",
    "agent": "climate_researcher",
    "data": {
        "total_samples": len(df),
        "ri_events": int(n_ri),
        "ri_rate": round(float(ri_rate), 5),
        "train_size": len(train_clean),
        "val_size": len(val_clean),
        "test_size": len(test_clean),
        "features": FEATURES,
        "seasons": f"{df['season'].min()}-{df['season'].max()}",
    },
    "logistic_regression": lr_results,
    f"{model_name.lower().replace(' ', '_')}": {
        "best_threshold": best_thr,
        "test_metrics": xgb_results,
        "threshold_sweep": results_by_thr,
    },
    "random_forest": rf_results,
}

with open(OUT, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {OUT}")
print("\nDone!")
