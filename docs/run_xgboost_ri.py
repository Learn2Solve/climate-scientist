#!/usr/bin/env python3
"""
XGBoost RI classifier on HURDAT2 real data.
Written by climate_researcher agent, 2026-02-20.

Loads all_samples.parquet, engineers features from track/intensity,
trains logistic regression + XGBoost, evaluates on temporal test split.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, brier_score_loss,
    classification_report
)

# ── Load data ──────────────────────────────────────────────────
DATA = Path("hurdat2_llm_toy/all_samples.parquet")
df = pd.read_parquet(DATA)
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Seasons: {df['season'].min()}-{df['season'].max()}")

# ── Feature engineering ────────────────────────────────────────
df["delta_wind"] = df["target_wind"] - df["last_wind"]
df["ri_label"] = (df["delta_wind"] >= 30).astype(int)
df["abs_lat"] = df["last_lat"].abs()
df["abs_lon"] = df["last_lon"].abs()

# Pressure deficit (proxy for intensity)
if "last_pressure" in df.columns:
    df["pressure_deficit"] = 1013.0 - df["last_pressure"]
    df["wp_ratio"] = df["last_wind"] / (df["pressure_deficit"].clip(lower=1))
else:
    df["pressure_deficit"] = np.nan
    df["wp_ratio"] = np.nan

# ── RI event statistics ────────────────────────────────────────
n_ri = df["ri_label"].sum()
ri_rate = n_ri / len(df)
print(f"\n=== RI Statistics ===")
print(f"RI events: {n_ri} / {len(df)} = {ri_rate:.3%}")
print(f"Mean delta_wind: {df['delta_wind'].mean():.1f} kt")
print(f"Std delta_wind: {df['delta_wind'].std():.1f} kt")
print(f"Max delta_wind: {df['delta_wind'].max():.0f} kt")
print(f"Min delta_wind: {df['delta_wind'].min():.0f} kt")

# ── Temporal train/val/test split ──────────────────────────────
train = df[df["season"] <= 2014].copy()
val = df[(df["season"] >= 2015) & (df["season"] <= 2018)].copy()
test = df[df["season"] >= 2019].copy()

print(f"\n=== Split sizes ===")
print(f"Train: {len(train)} samples, RI: {train['ri_label'].sum()} ({train['ri_label'].mean():.2%})")
print(f"Val:   {len(val)} samples, RI: {val['ri_label'].sum()} ({val['ri_label'].mean():.2%})")
print(f"Test:  {len(test)} samples, RI: {test['ri_label'].sum()} ({test['ri_label'].mean():.2%})")

# ── Feature selection ──────────────────────────────────────────
candidate_features = [
    "last_wind", "last_lat", "last_lon",
    "abs_lat", "abs_lon", "season"
]
# Add pressure features only if available
if df["pressure_deficit"].notna().mean() > 0.5:
    candidate_features += ["last_pressure", "pressure_deficit", "wp_ratio"]

FEATURES = [f for f in candidate_features if f in df.columns]
print(f"\nFeatures used: {FEATURES}")

# Drop NaN rows
train_clean = train.dropna(subset=FEATURES)
val_clean = val.dropna(subset=FEATURES)
test_clean = test.dropna(subset=FEATURES)

print(f"After NaN drop - Train: {len(train_clean)}, Val: {len(val_clean)}, Test: {len(test_clean)}")

X_train = train_clean[FEATURES].values
y_train = train_clean["ri_label"].values
X_val = val_clean[FEATURES].values
y_val = val_clean["ri_label"].values
X_test = test_clean[FEATURES].values
y_test = test_clean["ri_label"].values

# ── Model 1: Logistic Regression ──────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

lr_probs = lr.predict_proba(X_test_s)[:, 1]

# ── Model 2: XGBoost or GradientBoosting ──────────────────────
try:
    from xgboost import XGBClassifier
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    spw = n_neg / n_pos

    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="aucpr",
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    model_name = "XGBoost"
    
    # Feature importance
    importances = xgb.feature_importances_
    print(f"\n=== {model_name} Feature Importance ===")
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {feat:20s}: {imp:.4f}")

except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    sample_weight = np.where(y_train == 1, n_neg / n_pos, 1.0)

    xgb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=42,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weight)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    model_name = "GradientBoosting"
    
    importances = xgb.feature_importances_
    print(f"\n=== {model_name} Feature Importance ===")
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {feat:20s}: {imp:.4f}")

# ── Evaluation function ───────────────────────────────────────
def evaluate(y_true, probs, name, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    tn = ((pred == 0) & (y_true == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    try:
        auc = roc_auc_score(y_true, probs)
    except:
        auc = float("nan")
    
    try:
        brier = brier_score_loss(y_true, probs)
    except:
        brier = float("nan")
    
    print(f"\n=== {name} (threshold={threshold:.2f}) ===")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  Brier:     {brier:.4f}")
    
    return {"model": name, "threshold": threshold, "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4), "auc_roc": round(auc, 4),
            "brier": round(brier, 4)}

# ── Threshold sweep for best F1 ───────────────────────────────
print("\n=== Threshold sweep (XGBoost on validation) ===")
best_f1 = 0
best_thr = 0.5
for thr in np.arange(0.05, 0.95, 0.05):
    pred_v = (xgb.predict_proba(X_val if model_name == "XGBoost" else scaler.transform(X_val))[:, 1] >= thr).astype(int)
    if model_name != "XGBoost":
        pred_v = (xgb.predict_proba(X_val)[:, 1] >= thr).astype(int)
    f1_v = f1_score(y_val, pred_v, zero_division=0)
    rec_v = recall_score(y_val, pred_v, zero_division=0)
    prec_v = precision_score(y_val, pred_v, zero_division=0)
    if f1_v > best_f1:
        best_f1 = f1_v
        best_thr = thr
    if thr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"  thr={thr:.2f}: F1={f1_v:.3f} Prec={prec_v:.3f} Rec={rec_v:.3f}")

print(f"\n  Best val threshold: {best_thr:.2f} (F1={best_f1:.3f})")

# ── Final evaluation on test set ──────────────────────────────
results = []
results.append(evaluate(y_test, lr_probs, "LogisticRegression", 0.5))
results.append(evaluate(y_test, xgb_probs, model_name, 0.5))
results.append(evaluate(y_test, xgb_probs, f"{model_name}_optimized", best_thr))

# ── Persistence baseline ──────────────────────────────────────
persist_pred = np.zeros_like(y_test)  # persistence predicts no RI
persist_probs = np.full_like(y_test, ri_rate, dtype=float)  # climatological probability
results.append(evaluate(y_test, persist_probs, "Climatology", 0.5))

# ── Save results ──────────────────────────────────────────────
out = {
    "experiment": "HURDAT2 XGBoost RI Classification",
    "date": "2026-02-20",
    "dataset": "hurdat2_llm_toy/all_samples.parquet",
    "total_samples": len(df),
    "ri_events": int(n_ri),
    "ri_rate": round(ri_rate, 4),
    "features": FEATURES,
    "train_size": len(train_clean),
    "val_size": len(val_clean),
    "test_size": len(test_clean),
    "best_val_threshold": round(best_thr, 2),
    "results": results,
    "feature_importance": {f: round(float(v), 4) for f, v in zip(FEATURES, importances)},
}

out_path = Path("docs/xgboost_ri_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to {out_path}")
print("\n=== EXPERIMENT COMPLETE ===")
