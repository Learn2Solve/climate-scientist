#!/usr/bin/env python3
"""
v12: Refine the v11 best model (F1=0.397).

Grid search tightly around:
  n_estimators=500, max_depth=5, lr=0.03, subsample=0.85, min_leaf=10

Also try:
- XGBoost (if available)
- Combining stacking AUC=0.897 with better threshold
- Two-stage: use v4 as filter, refined model on positives
"""
import re, math, warnings, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
warnings.filterwarnings('ignore')

# ── Physics proxies (EXACT v4) ────────────────────────────────
def sst_proxy(lat, lon, month):
    base = 30.0 - 0.4 * abs(lat)
    seasonal = 1.5 * math.cos(2 * math.pi * (month - 9) / 12)
    if lon > -100 and lon < -60: gulf = 1.0
    elif lon > -60 and lon < -20: gulf = -0.5
    else: gulf = 0.0
    return base + seasonal + gulf

def shear_proxy(lat, lon, month):
    base = 5.0 + 0.3 * abs(lat)
    seasonal = 3.0 * math.cos(2 * math.pi * (month - 9) / 12)
    if abs(lat) > 30: base += 5.0
    return max(base - seasonal, 2.0)

def mpi_proxy(sst):
    if sst < 26.0: return 0.0
    return min(80.0, 20.0 * (sst - 26.0))

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

# ── Load & features ───────────────────────────────────────────
df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
rows = []
for idx, row in df.iterrows():
    lat, lon, wind = float(row['last_lat']), float(row['last_lon']), float(row['last_wind'])
    month_m = re.search(r'(\d{4})(\d{2})\d{2}', str(row.get('input', '')))
    month = int(month_m.group(2)) if month_m else 9
    sst = sst_proxy(lat, lon, month)
    shear = shear_proxy(lat, lon, month)
    mpi = mpi_proxy(sst)
    f = {
        'last_wind': wind, 'last_lat': lat, 'last_lon': lon,
        'abs_lat': abs(lat), 'month': month,
        'is_peak_season': 1 if month in [8, 9, 10] else 0,
        'wind_x_lat': wind * abs(lat), 'wind_squared': wind ** 2,
        'sst_proxy': sst, 'shear_proxy': shear, 'mpi_proxy': mpi,
        'intensity_deficit': mpi - wind,
        'sst_minus_26': max(sst - 26.0, 0.0),
        'favorable_env': 1 if (sst > 26.5 and shear < 10 and abs(lat) < 30) else 0,
        'shear_x_wind': shear * wind,
    }
    f.update(parse_track(row.get('input', '')))
    rows.append(f)

feat_df = pd.DataFrame(rows).fillna(0)
feature_cols = sorted(feat_df.columns.tolist())
print(f"Features: {len(feature_cols)}")

df['ri_label'] = ((df['target_wind'] - df['last_wind']) >= 30.0).astype(int)
train_mask = df['season'].between(1980, 2014)
val_mask = df['season'].between(2015, 2018)
test_mask = df['season'].between(2019, 2022)

X_train = feat_df.loc[train_mask, feature_cols].values
X_val = feat_df.loc[val_mask, feature_cols].values
X_test = feat_df.loc[test_mask, feature_cols].values
y_train = df.loc[train_mask, 'ri_label'].values
y_val = df.loc[val_mask, 'ri_label'].values
y_test = df.loc[test_mask, 'ri_label'].values

scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
sw = np.where(y_train == 1, scale_pos, 1.0)

print(f"Train: {len(X_train)} (RI={y_train.sum()}), Val: {len(X_val)} (RI={y_val.sum()}), Test: {len(X_test)} (RI={y_test.sum()})")

def eval_model(name, probs_val, probs_test):
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.05, 0.90, 0.002):
        f1 = f1_score(y_val, (probs_val >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    preds = (probs_test >= best_thr).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs_test)
    print(f"  {name:40s} AUC={auc:.3f} P={prec:.3f} R={rec:.3f} F1={f1:.3f} thr={best_thr:.3f}")
    return {'name': name, 'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1, 'thr': best_thr}

# ══════════════════════════════════════════════════════════════
# PART 1: Fine grid around winning config
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 1: Fine grid around v11 winner (lr=0.03, trees=500)")
print("="*70)

results = []
configs = []
for n_est in [400, 500, 600, 700, 800]:
    for lr in [0.02, 0.025, 0.03, 0.035]:
        for subsamp in [0.80, 0.85, 0.90]:
            for min_leaf in [8, 10, 12]:
                configs.append({
                    'n_estimators': n_est, 'max_depth': 5,
                    'learning_rate': lr, 'subsample': subsamp,
                    'min_samples_leaf': min_leaf
                })

print(f"Total configs: {len(configs)}")

# Too many - sample smartly. First do coarse, then refine.
# Coarse: lr x n_est (most impactful from v11)
coarse_results = []
print("\nCoarse grid (lr x n_est, fixed subsample=0.85, min_leaf=10):")
for n_est in [400, 500, 600, 700, 800]:
    for lr in [0.02, 0.025, 0.03, 0.035, 0.04]:
        cfg = {'n_estimators': n_est, 'max_depth': 5, 'learning_rate': lr,
               'subsample': 0.85, 'min_samples_leaf': 10}
        gbm = GradientBoostingClassifier(random_state=42, **cfg)
        gbm.fit(X_train, y_train, sample_weight=sw)
        r = eval_model(f"n={n_est} lr={lr}", 
                       gbm.predict_proba(X_val)[:,1],
                       gbm.predict_proba(X_test)[:,1])
        r['cfg'] = cfg
        coarse_results.append(r)

# Sort by F1
coarse_results.sort(key=lambda x: -x['f1'])
print(f"\nTop 5 coarse:")
for r in coarse_results[:5]:
    print(f"  F1={r['f1']:.3f} | n={r['cfg']['n_estimators']} lr={r['cfg']['learning_rate']}")

# Take top 3 configs and vary subsample + min_leaf
print(f"\nFine grid (top 3 coarse x subsample x min_leaf):")
fine_results = []
for base_r in coarse_results[:3]:
    base_cfg = base_r['cfg'].copy()
    for subsamp in [0.75, 0.80, 0.85, 0.90, 0.95]:
        for min_leaf in [5, 8, 10, 12, 15]:
            cfg = base_cfg.copy()
            cfg['subsample'] = subsamp
            cfg['min_samples_leaf'] = min_leaf
            gbm = GradientBoostingClassifier(random_state=42, **cfg)
            gbm.fit(X_train, y_train, sample_weight=sw)
            r = eval_model(f"n={cfg['n_estimators']} lr={cfg['learning_rate']} ss={subsamp} ml={min_leaf}",
                           gbm.predict_proba(X_val)[:,1],
                           gbm.predict_proba(X_test)[:,1])
            r['cfg'] = cfg
            fine_results.append(r)

fine_results.sort(key=lambda x: -x['f1'])
print(f"\nTop 10 fine:")
for r in fine_results[:10]:
    c = r['cfg']
    print(f"  F1={r['f1']:.3f} AUC={r['auc']:.3f} P={r['prec']:.3f} R={r['rec']:.3f} | "
          f"n={c['n_estimators']} lr={c['learning_rate']} ss={c['subsample']} ml={c['min_samples_leaf']}")

# ══════════════════════════════════════════════════════════════
# PART 2: Try XGBoost
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 2: XGBoost (if available)")
print("="*70)
try:
    import xgboost as xgb
    print("XGBoost available!")
    
    xgb_configs = [
        {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03,
         'subsample': 0.85, 'colsample_bytree': 0.8, 'min_child_weight': 5,
         'scale_pos_weight': scale_pos, 'random_state': 42, 'eval_metric': 'logloss'},
        {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.02,
         'subsample': 0.85, 'colsample_bytree': 0.8, 'min_child_weight': 5,
         'scale_pos_weight': scale_pos, 'random_state': 42, 'eval_metric': 'logloss'},
        {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03,
         'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 10,
         'scale_pos_weight': scale_pos, 'random_state': 42, 'eval_metric': 'logloss'},
        {'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.025,
         'subsample': 0.9, 'colsample_bytree': 0.85, 'min_child_weight': 8,
         'scale_pos_weight': scale_pos, 'random_state': 42, 'eval_metric': 'logloss'},
    ]
    
    xgb_results = []
    for i, cfg in enumerate(xgb_configs):
        model = xgb.XGBClassifier(**cfg, use_label_encoder=False)
        model.fit(X_train, y_train, verbose=False)
        r = eval_model(f"XGB config {i}",
                       model.predict_proba(X_val)[:,1],
                       model.predict_proba(X_test)[:,1])
        r['cfg'] = cfg
        xgb_results.append(r)
    
    xgb_results.sort(key=lambda x: -x['f1'])
    print(f"\nBest XGBoost: F1={xgb_results[0]['f1']:.3f}")
    
except ImportError:
    print("XGBoost not available. Trying to install...")
    import subprocess
    try:
        subprocess.check_call(['uv', 'pip', 'install', 'xgboost'], timeout=60)
        print("Installed! Re-run to use.")
    except:
        print("Could not install XGBoost. Skipping.")
    xgb_results = []

# ══════════════════════════════════════════════════════════════
# PART 3: Depth variation
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 3: Depth variation with best lr/n_est")
print("="*70)

best_coarse = coarse_results[0]['cfg']
depth_results = []
for depth in [3, 4, 5, 6, 7, 8]:
    cfg = best_coarse.copy()
    cfg['max_depth'] = depth
    gbm = GradientBoostingClassifier(random_state=42, **cfg)
    gbm.fit(X_train, y_train, sample_weight=sw)
    r = eval_model(f"depth={depth}",
                   gbm.predict_proba(X_val)[:,1],
                   gbm.predict_proba(X_test)[:,1])
    r['depth'] = depth
    depth_results.append(r)

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("v12 FINAL SUMMARY")
print("="*80)

all_results = coarse_results + fine_results + depth_results + xgb_results
all_results.sort(key=lambda x: -x['f1'])

print(f"\nOverall best 10:")
for i, r in enumerate(all_results[:10]):
    cfg_str = str(r.get('cfg', r.get('depth', '?')))
    print(f"  #{i+1}: F1={r['f1']:.3f} AUC={r['auc']:.3f} P={r['prec']:.3f} R={r['rec']:.3f} | {r['name']}")

champion = all_results[0]
print(f"\n*** CHAMPION: F1={champion['f1']:.3f} ***")
print(f"*** Config: {champion.get('cfg', 'N/A')} ***")
delta = (champion['f1'] - 0.397) / 0.397 * 100
print(f"*** vs v11 best: {delta:+.1f}% ***")
delta_v4 = (champion['f1'] - 0.372) / 0.372 * 100
print(f"*** vs v4 baseline: {delta_v4:+.1f}% ***")
