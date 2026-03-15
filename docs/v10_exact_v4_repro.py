#!/usr/bin/env python3
"""
v10: EXACT reproduction of v4 (ri_env_proxy_experiment.py).
No changes, no additions. Just verify we can reproduce F1=0.372.
Then ablation: add features one at a time to see what helps.
"""
import re, math, json, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss, confusion_matrix)

# ── Physics proxies (EXACT v4) ────────────────────────────────
def sst_proxy(lat, lon, month):
    base = 30.0 - 0.4 * abs(lat)
    seasonal = 1.5 * math.cos(2 * math.pi * (month - 9) / 12)
    if lon > -100 and lon < -60:
        gulf = 1.0
    elif lon > -60 and lon < -20:
        gulf = -0.5
    else:
        gulf = 0.0
    return base + seasonal + gulf

def shear_proxy(lat, lon, month):
    base = 5.0 + 0.3 * abs(lat)
    seasonal = 3.0 * math.cos(2 * math.pi * (month - 9) / 12)
    if abs(lat) > 30:
        base += 5.0
    return max(base - seasonal, 2.0)

def mpi_proxy(sst):
    if sst < 26.0:
        return 0.0
    return min(80.0, 20.0 * (sst - 26.0))

# ── Parse track (EXACT v4) ────────────────────────────────────
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

# ── Evaluate helper (EXACT v4) ────────────────────────────────
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
    return {'model': name, 'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1, 'brier': brier, 'thr': best_thr}

# ── Load & feature engineer (EXACT v4) ────────────────────────
df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
print(f"Loaded {len(df)} samples")

rows = []
for idx, row in df.iterrows():
    lat = float(row['last_lat'])
    lon = float(row['last_lon'])
    wind = float(row['last_wind'])
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
    sst = sst_proxy(lat, lon, month)
    shear = shear_proxy(lat, lon, month)
    mpi = mpi_proxy(sst)
    f['sst_proxy'] = sst
    f['shear_proxy'] = shear
    f['mpi_proxy'] = mpi
    f['intensity_deficit'] = mpi - wind
    f['sst_minus_26'] = max(sst - 26.0, 0.0)
    f['favorable_env'] = 1 if (sst > 26.5 and shear < 10 and abs(lat) < 30) else 0
    f['shear_x_wind'] = shear * wind
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

# Split (EXACT v4)
train_mask = df['season'].between(1980, 2014)
val_mask = df['season'].between(2015, 2018)
test_mask = df['season'].between(2019, 2022)

X_train = feat_df.loc[train_mask, feature_cols].values
X_val = feat_df.loc[val_mask, feature_cols].values
X_test = feat_df.loc[test_mask, feature_cols].values
y_train = df.loc[train_mask, 'ri_label'].values
y_val = df.loc[val_mask, 'ri_label'].values
y_test = df.loc[test_mask, 'ri_label'].values

print(f"Train: {len(X_train)} (RI={y_train.sum()})")
print(f"Val:   {len(X_val)} (RI={y_val.sum()})")
print(f"Test:  {len(X_test)} (RI={y_test.sum()})")

# Sample weight (EXACT v4)
scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
sw = np.where(y_train == 1, scale_pos, 1.0)

# ── EXACT v4 GBM ──────────────────────────────────────────────
print("\n" + "="*60)
print("EXACT v4 GBM reproduction")
print("="*60)
gbm = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gbm.fit(X_train, y_train, sample_weight=sw)
r_v4 = evaluate("GBM_v4_exact", gbm.predict_proba(X_val)[:,1],
                gbm.predict_proba(X_test)[:,1], y_val, y_test)

print("\n  Feature importance (top 15):")
for name, imp in sorted(zip(feature_cols, gbm.feature_importances_), key=lambda x: -x[1])[:15]:
    print(f"    {name:25s}: {imp:.4f}")

# ── EXACT v4 RF ────────────────────────────────────────────────
print("\n" + "="*60)
print("EXACT v4 RF reproduction")
print("="*60)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
r_rf = evaluate("RF_v4_exact", rf.predict_proba(X_val)[:,1],
                rf.predict_proba(X_test)[:,1], y_val, y_test)

# ── Now ablation: add features one at a time ───────────────────
print("\n" + "="*60)
print("ABLATION: Adding features one at a time to v4 base")
print("="*60)

# Generate extra features
extra_features = {}

# 1. consec_intensifying
consec_vals = []
for idx, row in df.iterrows():
    lines = str(row.get('input', '')).strip().split('\n')
    winds = []
    for line in lines:
        wind_m = re.search(r'wind\s+([-\d.]+)', line)
        if wind_m:
            winds.append(float(wind_m.group(1)))
    consec = 0
    if len(winds) >= 2:
        for i in range(len(winds)-1, 0, -1):
            if winds[i] > winds[i-1]:
                consec += 1
            else:
                break
    consec_vals.append(consec)
extra_features['consec_intensifying'] = consec_vals

# 2. wind_trend_12h
wt12_vals = []
for idx, row in df.iterrows():
    lines = str(row.get('input', '')).strip().split('\n')
    winds = []
    for line in lines:
        wind_m = re.search(r'wind\s+([-\d.]+)', line)
        if wind_m:
            winds.append(float(wind_m.group(1)))
    if len(winds) >= 4:
        wt12_vals.append(winds[-1] - winds[-3])
    else:
        wt12_vals.append(0.0)
extra_features['wind_trend_12h'] = wt12_vals

# 3. sst_x_lat (from v9)
sst_x_lat_vals = []
for idx, row in df.iterrows():
    lat = float(row['last_lat'])
    lon = float(row['last_lon'])
    month_m = re.search(r'(\d{4})(\d{2})\d{2}', str(row.get('input', '')))
    month = int(month_m.group(2)) if month_m else 9
    sst = sst_proxy(lat, lon, month)
    sst_x_lat_vals.append(sst * (30 - abs(lat)) / 30 if abs(lat) < 30 else 0)
extra_features['sst_x_lat'] = sst_x_lat_vals

# 4. intensity_ratio
int_ratio_vals = []
for idx, row in df.iterrows():
    lat = float(row['last_lat'])
    lon = float(row['last_lon'])
    wind = float(row['last_wind'])
    month_m = re.search(r'(\d{4})(\d{2})\d{2}', str(row.get('input', '')))
    month = int(month_m.group(2)) if month_m else 9
    sst = sst_proxy(lat, lon, month)
    mpi = mpi_proxy(sst)
    int_ratio_vals.append(wind / max(mpi, 1.0))
extra_features['intensity_ratio'] = int_ratio_vals

# 5. deficit_x_favorable 
def_fav_vals = []
for idx, row in df.iterrows():
    lat = float(row['last_lat'])
    lon = float(row['last_lon'])
    wind = float(row['last_wind'])
    month_m = re.search(r'(\d{4})(\d{2})\d{2}', str(row.get('input', '')))
    month = int(month_m.group(2)) if month_m else 9
    sst = sst_proxy(lat, lon, month)
    shear = shear_proxy(lat, lon, month)
    mpi = mpi_proxy(sst)
    deficit = mpi - wind
    fav = 1 if (sst > 26.5 and shear < 10 and abs(lat) < 30) else 0
    def_fav_vals.append(deficit * fav)
extra_features['deficit_x_favorable'] = def_fav_vals

ablation_results = []
for extra_name, extra_vals in extra_features.items():
    feat_df_ext = feat_df.copy()
    feat_df_ext[extra_name] = extra_vals
    ext_cols = feature_cols + [extra_name]
    
    X_tr = feat_df_ext.loc[train_mask, ext_cols].values
    X_v = feat_df_ext.loc[val_mask, ext_cols].values
    X_te = feat_df_ext.loc[test_mask, ext_cols].values
    
    gbm_ext = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    gbm_ext.fit(X_tr, y_train, sample_weight=sw)
    r = evaluate(f"GBM+{extra_name}", gbm_ext.predict_proba(X_v)[:,1],
                 gbm_ext.predict_proba(X_te)[:,1], y_val, y_test)
    ablation_results.append(r)
    
    # Find importance of the new feature
    imp_idx = ext_cols.index(extra_name)
    imp_val = gbm_ext.feature_importances_[imp_idx]
    print(f"  {extra_name} importance: {imp_val:.4f}")

# ── All extras combined ───────────────────────────────────────
print("\n" + "="*60)
print("ALL extras combined")
print("="*60)
feat_df_all = feat_df.copy()
for name, vals in extra_features.items():
    feat_df_all[name] = vals
all_cols = feature_cols + list(extra_features.keys())

X_tr_all = feat_df_all.loc[train_mask, all_cols].values
X_v_all = feat_df_all.loc[val_mask, all_cols].values
X_te_all = feat_df_all.loc[test_mask, all_cols].values

gbm_all = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gbm_all.fit(X_tr_all, y_train, sample_weight=sw)
evaluate("GBM+all_extras", gbm_all.predict_proba(X_v_all)[:,1],
         gbm_all.predict_proba(X_te_all)[:,1], y_val, y_test)

# ── Best extras + slow learner ────────────────────────────────
print("\n" + "="*60)
print("All extras + slow learner (500 trees, lr=0.03, depth=4)")
print("="*60)
gbm_slow_all = GradientBoostingClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.8, min_samples_leaf=15, random_state=42
)
gbm_slow_all.fit(X_tr_all, y_train, sample_weight=sw)
evaluate("GBM_slow+all", gbm_slow_all.predict_proba(X_v_all)[:,1],
         gbm_slow_all.predict_proba(X_te_all)[:,1], y_val, y_test)

# ── Ensemble: v4 GBM + slow + RF ─────────────────────────────
print("\n" + "="*60)
print("Ensemble: v4_GBM + slow_all + RF")
print("="*60)
ens_val = (gbm.predict_proba(X_val)[:,1] + 
           gbm_slow_all.predict_proba(X_v_all)[:,1] + 
           rf.predict_proba(X_val)[:,1]) / 3
ens_test = (gbm.predict_proba(X_test)[:,1] + 
            gbm_slow_all.predict_proba(X_te_all)[:,1] + 
            rf.predict_proba(X_test)[:,1]) / 3
evaluate("Ensemble_v4+slow+RF", ens_val, ens_test, y_val, y_test)

print("\n" + "="*60)
print("TARGET: v4 F1=0.372, AUC=0.884")
print("="*60)
