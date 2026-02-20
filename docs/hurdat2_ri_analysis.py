#!/usr/bin/env python3
"""
HURDAT2 Rapid Intensification Analysis on Real Data

This script must be copied to src/ to run:
  cp docs/hurdat2_ri_analysis.py src/hurdat2_ri_analysis.py
  uv run --no-project python src/hurdat2_ri_analysis.py

Analyzes HURDAT2 parquet for RI events (>=30 kt/24h wind increase)
and trains a logistic regression RI classifier on real data.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=Path("hurdat2_llm_toy/all_samples.parquet"))
    parser.add_argument("--out-json", type=Path, default=Path("docs/hurdat2_ri_real_results.json"))
    parser.add_argument("--ri-threshold", type=float, default=30.0)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} samples, columns: {list(df.columns)}")

    # Compute wind change and RI label
    df["delta_wind"] = df["target_wind"] - df["last_wind"]
    df["is_ri"] = (df["delta_wind"] >= args.ri_threshold).astype(int)

    n = len(df)
    n_ri = int(df["is_ri"].sum())
    print(f"\n=== RI Statistics ===")
    print(f"Total samples: {n}")
    print(f"RI events (>={args.ri_threshold} kt): {n_ri} ({n_ri/n*100:.2f}%)")

    # Distribution
    print(f"\nWind change distribution:")
    print(f"  Mean: {df['delta_wind'].mean():.1f} kt")
    print(f"  Std:  {df['delta_wind'].std():.1f} kt")
    print(f"  Min:  {df['delta_wind'].min():.1f}, Max: {df['delta_wind'].max():.1f}")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {df['delta_wind'].quantile(p/100):.1f}")

    # RI by decade
    df["decade"] = (df["season"] // 10) * 10
    print(f"\nRI by decade:")
    decade_stats = {}
    for d, g in df.groupby("decade"):
        nr = int(g["is_ri"].sum())
        nt = len(g)
        rate = nr / nt * 100
        print(f"  {int(d)}s: {nt} samples, {nr} RI, rate={rate:.1f}%")
        decade_stats[str(int(d))] = {"n": nt, "n_ri": nr, "rate_pct": round(rate, 1)}

    # RI characteristics
    ri_df = df[df["is_ri"] == 1]
    nri_df = df[df["is_ri"] == 0]
    print(f"\nRI cases: wind before={ri_df['last_wind'].mean():.1f} kt (std={ri_df['last_wind'].std():.1f})")
    print(f"Non-RI: wind before={nri_df['last_wind'].mean():.1f} kt")
    if "last_pressure" in df.columns:
        print(f"RI pressure: {ri_df['last_pressure'].mean():.1f} mb")
        print(f"Non-RI pressure: {nri_df['last_pressure'].mean():.1f} mb")
    print(f"RI latitude: {ri_df['last_lat'].mean():.1f}°N")
    print(f"Non-RI latitude: {nri_df['last_lat'].mean():.1f}°N")

    # === Logistic Regression on Real Data ===
    print(f"\n=== Logistic Regression RI Classifier (Real Data) ===")
    feats = ["last_wind", "last_pressure", "last_lat", "last_lon"]
    avail = [f for f in feats if f in df.columns and df[f].notna().all()]
    print(f"Features: {avail}")

    X = df[avail].values.astype(float)
    y = df["is_ri"].values

    # Train/val/test by season
    train_mask = df["season"] < 2015
    val_mask = (df["season"] >= 2015) & (df["season"] < 2019)
    test_mask = df["season"] >= 2019

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    X_te_s = sc.transform(X_te)

    # Train with class weighting
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs")
    lr.fit(X_tr_s, y_tr)

    # Feature importance
    print(f"\nFeature coefficients:")
    for name, coef in zip(avail, lr.coef_[0]):
        print(f"  {name}: {coef:.4f}")

    # Evaluate at multiple thresholds
    print(f"\n--- Threshold sweep on validation set ---")
    probs_val = lr.predict_proba(X_val_s)[:, 1]
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (probs_val >= t).astype(int)
        tp = int(((preds == 1) & (y_val == 1)).sum())
        fp = int(((preds == 1) & (y_val == 0)).sum())
        fn = int(((preds == 0) & (y_val == 1)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
        if tp + fp > 0:
            print(f"  t={t:.2f}: TP={tp} FP={fp} FN={fn} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    print(f"\nBest val threshold: {best_thresh:.2f} (F1={best_f1:.3f})")

    # Final evaluation on test set
    probs_te = lr.predict_proba(X_te_s)[:, 1]
    preds_te = (probs_te >= best_thresh).astype(int)
    tp = int(((preds_te == 1) & (y_te == 1)).sum())
    fp = int(((preds_te == 1) & (y_te == 0)).sum())
    fn = int(((preds_te == 0) & (y_te == 1)).sum())
    tn = int(((preds_te == 0) & (y_te == 0)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

    print(f"\n=== TEST RESULTS (2019-2022) ===")
    print(f"Threshold: {best_thresh:.2f}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    print(f"Test RI events: {int(y_te.sum())} / {len(y_te)}")

    # Persistence baseline for RI
    pers_preds = (df.loc[test_mask, "delta_wind"].shift(1).fillna(0) >= args.ri_threshold).astype(int)
    # Actually persistence = predict no change, so never predict RI
    print(f"\nPersistence baseline (always predict no-RI): F1=0, Recall=0%")

    # Climatology baseline: predict RI at training base rate
    base_rate = y_tr.mean()
    print(f"Climatology base rate: {base_rate*100:.2f}%")

    results = {
        "dataset": "HURDAT2 1980-2022",
        "total_samples": n,
        "ri_events": n_ri,
        "ri_rate_pct": round(n_ri / n * 100, 2),
        "ri_threshold_kt": args.ri_threshold,
        "features": avail,
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "train_ri_rate": round(float(y_tr.mean()) * 100, 2),
        "best_threshold": round(best_thresh, 2),
        "best_val_f1": round(best_f1, 3),
        "test": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "n_ri": int(y_te.sum()),
            "n_total": len(y_te),
        },
        "decade_stats": decade_stats,
        "feature_coefficients": {name: round(float(coef), 4) for name, coef in zip(avail, lr.coef_[0])},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    main()
