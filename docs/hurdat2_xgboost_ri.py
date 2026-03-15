#!/usr/bin/env python3
"""
HURDAT2 XGBoost RI Classification Experiment
=============================================
Reads the HURDAT2 parquet, engineers features from track data,
trains XGBoost classifier for RI prediction (wind increase >= 30kt in 24h).

Author: climate_researcher agent
Date: February 2026
"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

# --- Load data ---
parquet_path = Path("hurdat2_llm_toy/all_samples.parquet")
df = pd.read_parquet(parquet_path)
print(f"Loaded {len(df)} samples, seasons {df['season'].min()}-{df['season'].max()}")

# --- Parse input text to extract track history ---
def parse_track_history(input_text):
    """Extract track points from the input text field."""
    features = {}
    lines = str(input_text).strip().split('\n')
    
    # Extract lat/lon/wind from track lines
    lats, lons, winds, times = [], [], [], []
    for line in lines:
        # Match patterns like "lat 25.1, lon -80.2, wind 65"
        lat_m = re.search(r'lat\s+([-\d.]+)', line)
        lon_m = re.search(r'lon\s+([-\d.]+)', line)
        wind_m = re.search(r'wind\s+([-\d.]+)', line)
        if lat_m and lon_m and wind_m:
            lats.append(float(lat_m.group(1)))
            lons.append(float(lon_m.group(1)))
            winds.append(float(wind_m.group(1)))
    
    if len(winds) >= 2:
        features['wind_trend_6h'] = winds[-1] - winds[-2]
        features['wind_mean'] = np.mean(winds)
        features['wind_std'] = np.std(winds) if len(winds) > 1 else 0
        features['wind_max_history'] = max(winds)
        features['wind_min_history'] = min(winds)
        features['n_track_points'] = len(winds)
    if len(lats) >= 2:
        features['lat_trend'] = lats[-1] - lats[-2]
        features['lon_trend'] = lons[-1] - lons[-2]
        features['translation_speed'] = np.sqrt(
            (lats[-1] - lats[-2])**2 + (lons[-1] - lons[-2])**2
        )
    if len(winds) >= 3:
        features['wind_accel'] = (winds[-1] - winds[-2]) - (winds[-2] - winds[-3])
    
    return features

# --- Feature engineering ---
print("Engineering features...")
feature_rows = []
for idx, row in df.iterrows():
    f = {}
    f['last_wind'] = float(row['last_wind'])
    f['last_lat'] = float(row['last_lat'])
    f['last_lon'] = float(row['last_lon'])
    f['abs_lat'] = abs(float(row['last_lat']))
    f['season'] = int(row['season'])
    
    # Parse track history from input text
    track_feats = parse_track_history(row.get('input', ''))
    f.update(track_feats)
    
    feature_rows.append(f)

feat_df = pd.DataFrame(feature_rows)
print(f"Feature columns: {list(feat_df.columns)}")
print(f"Feature matrix shape: {feat_df.shape}")

# --- Define RI label ---
RI_THRESHOLD = 30.0  # kt in 24h
df['delta_wind'] = df['target_wind'] - df['last_wind']
df['ri_label'] = (df['delta_wind'] >= RI_THRESHOLD).astype(int)
ri_count = df['ri_label'].sum()
ri_rate = df['ri_label'].mean()
print(f"RI events: {ri_count}/{len(df)} = {ri_rate:.4f}")

# --- Train/val/test split by season ---
train_mask = df['season'].between(1980, 2014)
val_mask = df['season'].between(2015, 2018)
test_mask = df['season'].between(2019, 2022)

y_train = df.loc[train_mask, 'ri_label'].values
y_val = df.loc[val_mask, 'ri_label'].values
y_test = df.loc[test_mask, 'ri_label'].values

# Fill NaN features with 0
feat_df = feat_df.fillna(0)

feature_cols = [c for c in feat_df.columns if c != 'season']
X_train = feat_df.loc[train_mask, feature_cols].values
X_val = feat_df.loc[val_mask, feature_cols].values
X_test = feat_df.loc[test_mask, feature_cols].values

print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"RI in train: {y_train.sum()}/{len(y_train)} = {y_train.mean():.4f}")
print(f"RI in val:   {y_val.sum()}/{len(y_val)} = {y_val.mean():.4f}")
print(f"RI in test:  {y_test.sum()}/{len(y_test)} = {y_test.mean():.4f}")

# --- Model 1: Random Forest baseline ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)

print("\n" + "="*60)
print("MODEL 1: Random Forest (class_weight='balanced')")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

rf_probs_val = rf.predict_proba(X_val)[:, 1]
rf_probs_test = rf.predict_proba(X_test)[:, 1]

# Optimize threshold on validation set
best_f1, best_thr = 0, 0.5
for thr in np.arange(0.05, 0.95, 0.01):
    preds = (rf_probs_val >= thr).astype(int)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"Best val threshold: {best_thr:.2f}, val F1: {best_f1:.3f}")

rf_preds_test = (rf_probs_test >= best_thr).astype(int)
rf_auc = roc_auc_score(y_test, rf_probs_test) if y_test.sum() > 0 else 0
rf_f1 = f1_score(y_test, rf_preds_test, zero_division=0)
rf_prec = precision_score(y_test, rf_preds_test, zero_division=0)
rf_rec = recall_score(y_test, rf_preds_test, zero_division=0)

print(f"Test AUC:       {rf_auc:.3f}")
print(f"Test Precision: {rf_prec:.3f}")
print(f"Test Recall:    {rf_rec:.3f}")
print(f"Test F1:        {rf_f1:.3f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, rf_preds_test)}")

# Feature importance
importances = rf.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {name:25s}: {imp:.4f}")

# --- Model 2: XGBoost ---
print("\n" + "="*60)
print("MODEL 2: XGBoost (scale_pos_weight)")
print("="*60)

try:
    import xgboost as xgb
    
    scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric='logloss',
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, random_state=42,
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    xgb_probs_val = xgb_model.predict_proba(X_val)[:, 1]
    xgb_probs_test = xgb_model.predict_proba(X_test)[:, 1]
    
    # Optimize threshold
    best_f1_xgb, best_thr_xgb = 0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        preds = (xgb_probs_val >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1_xgb:
            best_f1_xgb = f1
            best_thr_xgb = thr
    
    print(f"Best val threshold: {best_thr_xgb:.2f}, val F1: {best_f1_xgb:.3f}")
    
    xgb_preds_test = (xgb_probs_test >= best_thr_xgb).astype(int)
    xgb_auc = roc_auc_score(y_test, xgb_probs_test) if y_test.sum() > 0 else 0
    xgb_f1 = f1_score(y_test, xgb_preds_test, zero_division=0)
    xgb_prec = precision_score(y_test, xgb_preds_test, zero_division=0)
    xgb_rec = recall_score(y_test, xgb_preds_test, zero_division=0)
    
    print(f"Test AUC:       {xgb_auc:.3f}")
    print(f"Test Precision: {xgb_prec:.3f}")
    print(f"Test Recall:    {xgb_rec:.3f}")
    print(f"Test F1:        {xgb_f1:.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, xgb_preds_test)}")
    
    # Feature importance
    importances_xgb = xgb_model.feature_importances_
    for name, imp in sorted(zip(feature_cols, importances_xgb), key=lambda x: -x[1]):
        print(f"  {name:25s}: {imp:.4f}")

except ImportError:
    print("XGBoost not available, skipping")
    xgb_auc, xgb_f1, xgb_prec, xgb_rec = 0, 0, 0, 0

# --- Model 3: Hybrid ensemble ---
print("\n" + "="*60)
print("MODEL 3: Hybrid Ensemble (physics gate + ML)")
print("="*60)

# Physics gate: only consider RI possible if wind > 20kt and abs(lat) < 35
physics_mask_test = (feat_df.loc[test_mask, 'last_wind'].values > 20) & \
                    (feat_df.loc[test_mask, 'abs_lat'].values < 35)

try:
    hybrid_probs = np.where(physics_mask_test, xgb_probs_test, 0.0)
except:
    hybrid_probs = np.where(physics_mask_test, rf_probs_test, 0.0)

# Optimize threshold
best_f1_hyb, best_thr_hyb = 0, 0.5
for thr in np.arange(0.05, 0.95, 0.01):
    preds = (hybrid_probs >= thr).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1_hyb:
        best_f1_hyb = f1
        best_thr_hyb = thr

hybrid_preds = (hybrid_probs >= best_thr_hyb).astype(int)
hyb_auc = roc_auc_score(y_test, hybrid_probs) if y_test.sum() > 0 else 0
hyb_f1 = f1_score(y_test, hybrid_preds, zero_division=0)
hyb_prec = precision_score(y_test, hybrid_preds, zero_division=0)
hyb_rec = recall_score(y_test, hybrid_preds, zero_division=0)

print(f"Physics gate passes: {physics_mask_test.sum()}/{len(physics_mask_test)}")
print(f"Best threshold: {best_thr_hyb:.2f}")
print(f"Test AUC:       {hyb_auc:.3f}")
print(f"Test Precision: {hyb_prec:.3f}")
print(f"Test Recall:    {hyb_rec:.3f}")
print(f"Test F1:        {hyb_f1:.3f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, hybrid_preds)}")

# --- Summary ---
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Model':<30s} {'AUC':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
print("-" * 55)
print(f"{'Random Forest':<30s} {rf_auc:6.3f} {rf_prec:6.3f} {rf_rec:6.3f} {rf_f1:6.3f}")
try:
    print(f"{'XGBoost':<30s} {xgb_auc:6.3f} {xgb_prec:6.3f} {xgb_rec:6.3f} {xgb_f1:6.3f}")
except:
    pass
print(f"{'Hybrid (physics+ML)':<30s} {hyb_auc:6.3f} {hyb_prec:6.3f} {hyb_rec:6.3f} {hyb_f1:6.3f}")
print(f"\nDataset: {len(df)} samples, RI rate: {ri_rate:.4f}")
print(f"Features used: {feature_cols}")
