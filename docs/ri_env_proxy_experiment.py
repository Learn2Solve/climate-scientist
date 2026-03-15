#!/usr/bin/env python3
"""
RI Classification v4: Environmental proxy features from track data.
Key idea: SST correlates with latitude/longitude/month. We can build
climatological SST and shear proxies from location + season.
"""
import re, numpy as np, pandas as pd, math
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss, confusion_matrix)

# --- Climatological SST proxy ---
def sst_proxy(lat, lon, month):
    """Rough Atlantic SST climatology (deg C) based on location and month.
    Based on Reynolds OI SST patterns: warmest near equator in summer."""
    base = 30.0 - 0.4 * abs(lat)  # latitude gradient
    # Seasonal cycle: peak in Sep (month 9)
    seasonal = 1.5 * math.cos(2 * math.pi * (month - 9) / 12)
    # Gulf Stream warm tongue: warmer west of -60 lon
    if lon > -100 and lon < -60:
        gulf = 1.0
    elif lon > -60 and lon < -20:
        gulf = -0.5  # cooler mid-Atlantic
    else:
        gulf = 0.0
    return base + seasonal + gulf

def shear_proxy(lat, lon, month):
    """Rough vertical wind shear proxy (m/s). Higher shear at higher latitudes,
    in spring, and near jet stream."""
    base = 5.0 + 0.3 * abs(lat)
    # Seasonal: lower shear in Aug-Oct
    seasonal = 3.0 * math.cos(2 * math.pi * (month - 9) / 12)
    # More shear at higher latitudes
    if abs(lat) > 30:
        base += 5.0
    return max(base - seasonal, 2.0)

def mpi_proxy(sst):
    """Maximum potential intensity proxy from SST (Emanuel 1988 approx).
    Very rough: MPI ~ 0 if SST < 26C, rises to ~80m/s at 30C."""
    if sst < 26.0:
        return 0.0
    return min(80.0, 20.0 * (sst - 26.0))

# Load data
df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
print(f"Loaded {len(df)} samples")

# Parse track
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
        f['total_lat_change'] = lats[-1] - lats[0]
        f['total_lon_change'] = lons[-1] - lons[0]
    if len(winds) >= 3:
        f['wind_accel'] = (winds[-1]-winds[-2]) - (winds[-2]-winds[-3])
    return f

# Feature engineering
print("Engineering features with environmental proxies...")
rows = []
for idx, row in df.iterrows():
    lat = float(row['last_lat'])
    lon = float(row['last_lon'])
    wind = float(row['last_wind'])
    
    # Extract month
    month_m = re.search(r'(\d{4})(\d{2})\d{2}', str(row.get('input', '')))
    month = int(month_m.group(2)) if month_m else 9
    
    f = {
        'last_wind': wind,
        'last_lat': lat,
        'last_lon': lon,
        'abs_lat': abs(lat),
        'month': month,
        'is_peak_season': 1 if month in [8, 9, 10] else 0,
        'wind_x_lat': wind * abs(lat),
        'wind_squared': wind ** 2,
    }
    
    # Environmental proxies
    sst = sst_proxy(lat, lon, month)
    shear = shear_proxy(lat, lon, month)
    mpi = mpi_proxy(sst)
    
    f['sst_proxy'] = sst
    f['shear_proxy'] = shear
    f['mpi_proxy'] = mpi
    f['intensity_deficit'] = mpi - wind  # room to grow
    f['sst_minus_26'] = max(sst - 26.0, 0.0)  # excess SST
    f['favorable_env'] = 1 if (sst > 26.5 and shear < 10 and abs(lat) < 30) else 0
    f['shear_x_wind'] = shear * wind  # interaction
    
    # Track features
    track_f = parse_track(row.get('input', ''))
    f.update(track_f)
    rows.append(f)

feat_df = pd.DataFrame(rows).fillna(0)
feature_cols = sorted(feat_df.columns.tolist())
print(f"Features ({len(feature_cols)}): {feature_cols}")

# Labels
df['delta_wind'] = df['target_wind'] - df['last_wind']
df['ri_label'] = (df['delta_wind'] >= 30.0).astype(int)
print(f"RI: {df['ri_label'].sum()}/{len(df)} = {df['ri_label'].mean():.4f}")

# Split
train_mask = df['season'].between(1980, 2014)
val_mask = df['season'].between(2015, 2018)
test_mask = df['season'].between(2019, 2022)

X_train = feat_df.loc[train_mask, feature_cols].values
X_val = feat_df.loc[val_mask, feature_cols].values
X_test = feat_df.loc[test_mask, feature_cols].values
y_train = df.loc[train_mask, 'ri_label'].values
y_val = df.loc[val_mask, 'ri_label'].values
y_test = df.loc[test_mask, 'ri_label'].values

print(f"Train: {len(X_train)} (RI={y_train.sum()}), Val: {len(X_val)} (RI={y_val.sum()}), Test: {len(X_test)} (RI={y_test.sum()})")

# Evaluation
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
    brier = brier_score_loss(y_test, probs_test)
    cm = confusion_matrix(y_test, preds)
    print(f"\n{name} (thr={best_thr:.2f})")
    print(f"  AUC={auc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  Brier={brier:.4f}")
    print(f"  CM: {cm.tolist()}")
    return {'model': name, 'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1, 'brier': brier}

results = []
scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
sw = np.where(y_train == 1, scale_pos, 1.0)

# Model 1: GBM baseline (same as v3 best)
gbm = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, min_samples_leaf=10, random_state=42)
gbm.fit(X_train, y_train, sample_weight=sw)
r = evaluate("GBM (with env proxies)", gbm.predict_proba(X_val)[:,1], 
             gbm.predict_proba(X_test)[:,1], y_val, y_test)
results.append(r)

print("  Feature importance (top 15):")
for name, imp in sorted(zip(feature_cols, gbm.feature_importances_), key=lambda x: -x[1])[:15]:
    print(f"    {name:25s}: {imp:.4f}")

# Model 2: RF with env proxies
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                            class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
r = evaluate("RF (with env proxies)", rf.predict_proba(X_val)[:,1],
             rf.predict_proba(X_test)[:,1], y_val, y_test)
results.append(r)

# Model 3: Calibrated GBM (Platt scaling)
cal_gbm = CalibratedClassifierCV(gbm, cv=3, method='sigmoid')
cal_gbm.fit(X_train, y_train, sample_weight=sw)
r = evaluate("GBM Calibrated", cal_gbm.predict_proba(X_val)[:,1],
             cal_gbm.predict_proba(X_test)[:,1], y_val, y_test)
results.append(r)

# Model 4: Favorable-environment-gated GBM
fav_val = feat_df.loc[val_mask, 'favorable_env'].values.astype(bool)
fav_test = feat_df.loc[test_mask, 'favorable_env'].values.astype(bool)
gbm_probs_val = gbm.predict_proba(X_val)[:,1]
gbm_probs_test = gbm.predict_proba(X_test)[:,1]
gated_val = np.where(fav_val, gbm_probs_val * 1.3, gbm_probs_val * 0.5)
gated_test = np.where(fav_test, gbm_probs_test * 1.3, gbm_probs_test * 0.5)
gated_val = np.clip(gated_val, 0, 1)
gated_test = np.clip(gated_test, 0, 1)
r = evaluate("Env-gated GBM", gated_val, gated_test, y_val, y_test)
results.append(r)

# Summary
print("\n" + "="*65)
print("COMPARISON: v4 (env proxies) vs v3 (track-only)")
print("="*65)
print(f"{'Model':<25s} {'AUC':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Brier':>7s}")
print("-"*60)
for r in results:
    print(f"{r['model']:<25s} {r['auc']:6.3f} {r['prec']:6.3f} {r['rec']:6.3f} {r['f1']:6.3f} {r['brier']:7.4f}")
print(f"\nv3 best (GBM track-only): AUC=0.874  F1=0.351")
print(f"Features ({len(feature_cols)}): {feature_cols}")
