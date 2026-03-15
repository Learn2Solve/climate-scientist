#!/usr/bin/env python3
"""
RI Classification v3: GradientBoosting + enhanced features
Uses sklearn's GradientBoostingClassifier (no xgboost dependency).
Adds temporal features (month, season-based) and interaction terms.
"""
import re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Load data
df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
print(f"Loaded {len(df)} samples, seasons {df['season'].min()}-{df['season'].max()}")

# Parse track history
def parse_track(text):
    f = {}
    lines = str(text).strip().split('\n')
    lats, lons, winds = [], [], []
    for line in lines:
        lat_m = re.search(r'lat\s+([-\d.]+)', line)
        lon_m = re.search(r'lon\s+([-\d.]+)', line)
        wind_m = re.search(r'wind\s+([-\d.]+)', line)
        if lat_m and lon_m and wind_m:
            lats.append(float(lat_m.group(1)))
            lons.append(float(lon_m.group(1)))
            winds.append(float(wind_m.group(1)))
    if len(winds) >= 2:
        f['wind_trend_6h'] = winds[-1] - winds[-2]
        f['wind_mean'] = np.mean(winds)
        f['wind_std'] = np.std(winds)
        f['wind_max_history'] = max(winds)
        f['wind_min_history'] = min(winds)
        f['wind_range'] = max(winds) - min(winds)
        f['n_track_points'] = len(winds)
    if len(lats) >= 2:
        f['lat_trend'] = lats[-1] - lats[-2]
        f['lon_trend'] = lons[-1] - lons[-2]
        f['translation_speed'] = np.sqrt((lats[-1]-lats[-2])**2 + (lons[-1]-lons[-2])**2)
    if len(winds) >= 3:
        f['wind_accel'] = (winds[-1]-winds[-2]) - (winds[-2]-winds[-3])
    # New: track length and direction
    if len(lats) >= 2:
        f['total_lat_change'] = lats[-1] - lats[0]
        f['total_lon_change'] = lons[-1] - lons[0]
    return f

# Feature engineering
print("Engineering features...")
rows = []
for idx, row in df.iterrows():
    f = {
        'last_wind': float(row['last_wind']),
        'last_lat': float(row['last_lat']),
        'last_lon': float(row['last_lon']),
        'abs_lat': abs(float(row['last_lat'])),
    }
    # Temporal features - extract month from input text if possible
    month_m = re.search(r'(\d{4})(\d{2})\d{2}', str(row.get('input', '')))
    if month_m:
        f['month'] = int(month_m.group(2))
        f['is_peak_season'] = 1 if int(month_m.group(2)) in [8, 9, 10] else 0
    else:
        f['month'] = 9  # default peak
        f['is_peak_season'] = 1
    # Interaction features
    f['wind_x_lat'] = f['last_wind'] * f['abs_lat']
    f['wind_squared'] = f['last_wind'] ** 2
    # Track features
    track_f = parse_track(row.get('input', ''))
    f.update(track_f)
    rows.append(f)

feat_df = pd.DataFrame(rows).fillna(0)
print(f"Features ({len(feat_df.columns)}): {sorted(feat_df.columns.tolist())}")

# Labels
df['delta_wind'] = df['target_wind'] - df['last_wind']
df['ri_label'] = (df['delta_wind'] >= 30.0).astype(int)
print(f"RI events: {df['ri_label'].sum()}/{len(df)} = {df['ri_label'].mean():.4f}")

# Split
train_mask = df['season'].between(1980, 2014)
val_mask = df['season'].between(2015, 2018)
test_mask = df['season'].between(2019, 2022)
feature_cols = sorted(feat_df.columns.tolist())

X_train = feat_df.loc[train_mask, feature_cols].values
X_val = feat_df.loc[val_mask, feature_cols].values
X_test = feat_df.loc[test_mask, feature_cols].values
y_train = df.loc[train_mask, 'ri_label'].values
y_val = df.loc[val_mask, 'ri_label'].values
y_test = df.loc[test_mask, 'ri_label'].values

print(f"Train: {len(X_train)} (RI={y_train.sum()}), Val: {len(X_val)} (RI={y_val.sum()}), Test: {len(X_test)} (RI={y_test.sum()})")

# Helper
def evaluate(name, probs_val, probs_test, y_val, y_test):
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_val, (probs_val >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    preds = (probs_test >= best_thr).astype(int)
    auc = roc_auc_score(y_test, probs_test)
    f1 = f1_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    print(f"\n{name} (threshold={best_thr:.2f}, val_F1={best_f1:.3f})")
    print(f"  AUC={auc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
    print(f"  CM: {cm.tolist()}")
    return {'model': name, 'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1, 'thr': best_thr}

results = []

# Model 1: Logistic Regression baseline
print("\n" + "="*60)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_va_s = scaler.transform(X_val)
X_te_s = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
lr.fit(X_tr_s, y_train)
r = evaluate("Logistic Regression", lr.predict_proba(X_va_s)[:,1], lr.predict_proba(X_te_s)[:,1], y_val, y_test)
results.append(r)

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                            class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
r = evaluate("Random Forest", rf.predict_proba(X_val)[:,1], rf.predict_proba(X_test)[:,1], y_val, y_test)
results.append(r)

# Print RF feature importance
print("  Feature importance (top 10):")
for name, imp in sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1])[:10]:
    print(f"    {name:25s}: {imp:.4f}")

# Model 3: GradientBoosting
scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
# Use sample weights for class balance
sample_weights = np.where(y_train == 1, scale_pos, 1.0)

gbm = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gbm.fit(X_train, y_train, sample_weight=sample_weights)
r = evaluate("GradientBoosting", gbm.predict_proba(X_val)[:,1], gbm.predict_proba(X_test)[:,1], y_val, y_test)
results.append(r)

print("  Feature importance (top 10):")
for name, imp in sorted(zip(feature_cols, gbm.feature_importances_), key=lambda x: -x[1])[:10]:
    print(f"    {name:25s}: {imp:.4f}")

# Model 4: GBM tuned (deeper, more trees)
gbm2 = GradientBoostingClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, min_samples_leaf=5, max_features='sqrt', random_state=42
)
gbm2.fit(X_train, y_train, sample_weight=sample_weights)
r = evaluate("GBM-tuned", gbm2.predict_proba(X_val)[:,1], gbm2.predict_proba(X_test)[:,1], y_val, y_test)
results.append(r)

# Model 5: Hybrid (physics gate + GBM)
physics_mask_val = (feat_df.loc[val_mask, 'last_wind'].values > 20) & (feat_df.loc[val_mask, 'abs_lat'].values < 35)
physics_mask_test = (feat_df.loc[test_mask, 'last_wind'].values > 20) & (feat_df.loc[test_mask, 'abs_lat'].values < 35)

# Use best GBM
best_gbm = gbm2  # or gbm, whichever is better
hyb_probs_val = np.where(physics_mask_val, best_gbm.predict_proba(X_val)[:,1], 0.0)
hyb_probs_test = np.where(physics_mask_test, best_gbm.predict_proba(X_test)[:,1], 0.0)
r = evaluate("Hybrid (physics+GBM)", hyb_probs_val, hyb_probs_test, y_val, y_test)
results.append(r)

# Model 6: Stacking ensemble (RF + GBM meta-learner)
print("\nBuilding stacking ensemble...")
rf_probs_tr = rf.predict_proba(X_train)[:,1]
gbm_probs_tr = best_gbm.predict_proba(X_train)[:,1]
meta_X_tr = np.column_stack([rf_probs_tr, gbm_probs_tr])

rf_probs_va = rf.predict_proba(X_val)[:,1]
gbm_probs_va = best_gbm.predict_proba(X_val)[:,1]
meta_X_va = np.column_stack([rf_probs_va, gbm_probs_va])

rf_probs_te = rf.predict_proba(X_test)[:,1]
gbm_probs_te = best_gbm.predict_proba(X_test)[:,1]
meta_X_te = np.column_stack([rf_probs_te, gbm_probs_te])

meta_lr = LogisticRegression(class_weight='balanced', max_iter=500)
meta_lr.fit(meta_X_tr, y_train)
r = evaluate("Stack (RF+GBM->LR)", meta_lr.predict_proba(meta_X_va)[:,1], meta_lr.predict_proba(meta_X_te)[:,1], y_val, y_test)
results.append(r)

# Final summary
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"{'Model':<30s} {'AUC':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
print("-"*55)
for r in results:
    print(f"{r['model']:<30s} {r['auc']:6.3f} {r['prec']:6.3f} {r['rec']:6.3f} {r['f1']:6.3f}")
print(f"\nDataset: {len(df)} samples, RI rate: {df['ri_label'].mean():.4f}")
print(f"Features ({len(feature_cols)}): {feature_cols}")
