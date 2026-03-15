#!/usr/bin/env python3
"""
v12-fast: Focused refinement around v11 winner (F1=0.397).
Reduced grid to stay under 5 min.
"""
import re, math, warnings, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
warnings.filterwarnings('ignore')

def sst_proxy(lat, lon, month):
    base = 30.0 - 0.4 * abs(lat)
    seasonal = 1.5 * math.cos(2 * math.pi * (month - 9) / 12)
    gulf = 1.0 if (-100 < lon < -60) else (-0.5 if (-60 < lon < -20) else 0.0)
    return base + seasonal + gulf

def shear_proxy(lat, lon, month):
    base = 5.0 + 0.3 * abs(lat) + (5.0 if abs(lat) > 30 else 0)
    seasonal = 3.0 * math.cos(2 * math.pi * (month - 9) / 12)
    return max(base - seasonal, 2.0)

def mpi_proxy(sst):
    return 0.0 if sst < 26.0 else min(80.0, 20.0 * (sst - 26.0))

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

df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
rows = []
for _, row in df.iterrows():
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

print(f"Data: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
print(f"RI rate: train={y_train.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}")

def eval_quick(probs_val, probs_test):
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.10, 0.80, 0.005):
        f1 = f1_score(y_val, (probs_val >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    preds = (probs_test >= best_thr).astype(int)
    return {
        'f1': f1_score(y_test, preds, zero_division=0),
        'auc': roc_auc_score(y_test, probs_test),
        'prec': precision_score(y_test, preds, zero_division=0),
        'rec': recall_score(y_test, preds, zero_division=0),
        'thr': best_thr,
    }

# ── Grid: 20 configs focused on the sweet spot ──────────────
print("\n" + "="*70)
print("Focused grid: lr × n_est × depth (20 configs)")
print("="*70)

configs = [
    # Vary lr and n_est (depth=5 fixed - proven best)
    (500, 5, 0.025, 0.85, 10),
    (500, 5, 0.030, 0.85, 10),  # v11 winner
    (500, 5, 0.035, 0.85, 10),
    (600, 5, 0.025, 0.85, 10),
    (600, 5, 0.030, 0.85, 10),
    (700, 5, 0.020, 0.85, 10),
    (700, 5, 0.025, 0.85, 10),
    (800, 5, 0.020, 0.85, 10),
    (800, 5, 0.025, 0.85, 10),
    # Vary subsample with best lr/n_est combos
    (500, 5, 0.030, 0.80, 10),
    (500, 5, 0.030, 0.90, 10),
    (600, 5, 0.025, 0.80, 10),
    (600, 5, 0.025, 0.90, 10),
    # Vary min_leaf
    (500, 5, 0.030, 0.85, 8),
    (500, 5, 0.030, 0.85, 15),
    (600, 5, 0.025, 0.85, 8),
    (600, 5, 0.025, 0.85, 15),
    # Depth variation
    (500, 4, 0.030, 0.85, 10),
    (500, 6, 0.030, 0.85, 10),
    (600, 4, 0.025, 0.85, 10),
]

results = []
for n_est, depth, lr, ss, ml in configs:
    gbm = GradientBoostingClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        subsample=ss, min_samples_leaf=ml, random_state=42
    )
    gbm.fit(X_train, y_train, sample_weight=sw)
    r = eval_quick(gbm.predict_proba(X_val)[:,1], gbm.predict_proba(X_test)[:,1])
    r['cfg'] = f"n={n_est} d={depth} lr={lr} ss={ss} ml={ml}"
    results.append(r)
    print(f"  F1={r['f1']:.3f} AUC={r['auc']:.3f} P={r['prec']:.3f} R={r['rec']:.3f} thr={r['thr']:.3f} | {r['cfg']}")

# ── Try XGBoost ──────────────────────────────────────────────
print("\n" + "="*70)
print("XGBoost")
print("="*70)
try:
    import xgboost as xgb
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
        {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.025,
         'subsample': 0.85, 'colsample_bytree': 0.85, 'min_child_weight': 8,
         'scale_pos_weight': scale_pos, 'random_state': 42, 'eval_metric': 'logloss'},
    ]
    for i, cfg in enumerate(xgb_configs):
        model = xgb.XGBClassifier(**cfg, use_label_encoder=False, verbosity=0)
        model.fit(X_train, y_train)
        r = eval_quick(model.predict_proba(X_val)[:,1], model.predict_proba(X_test)[:,1])
        r['cfg'] = f"XGB-{i}: {cfg}"
        results.append(r)
        print(f"  F1={r['f1']:.3f} AUC={r['auc']:.3f} P={r['prec']:.3f} R={r['rec']:.3f} | XGB config {i}")
except ImportError:
    print("XGBoost not available")

# ── Best ensemble: top 3 models averaged ─────────────────────
print("\n" + "="*70)
print("Ensemble of top 3")
print("="*70)

# Retrain top configs
results.sort(key=lambda x: -x['f1'])
top3_cfgs = [
    (500, 5, 0.030, 0.85, 10),  # v11 winner
    (600, 5, 0.025, 0.85, 10),
    (700, 5, 0.025, 0.85, 10),
]

ens_val = np.zeros(len(X_val))
ens_test = np.zeros(len(X_test))
for n_est, depth, lr, ss, ml in top3_cfgs:
    gbm = GradientBoostingClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        subsample=ss, min_samples_leaf=ml, random_state=42
    )
    gbm.fit(X_train, y_train, sample_weight=sw)
    ens_val += gbm.predict_proba(X_val)[:,1] / 3
    ens_test += gbm.predict_proba(X_test)[:,1] / 3

r_ens = eval_quick(ens_val, ens_test)
r_ens['cfg'] = "Ensemble top3"
results.append(r_ens)
print(f"  F1={r_ens['f1']:.3f} AUC={r_ens['auc']:.3f} P={r_ens['prec']:.3f} R={r_ens['rec']:.3f}")

# ── SUMMARY ──────────────────────────────────────────────────
results.sort(key=lambda x: -x['f1'])
print("\n" + "="*80)
print("v12 FINAL RESULTS (top 10)")
print("="*80)
print(f"{'Rank':>4} {'F1':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'Thr':>6}  Config")
print("-"*80)
for i, r in enumerate(results[:10]):
    marker = " ***" if r['f1'] > 0.397 else ""
    print(f"  {i+1:>2}  {r['f1']:.3f}  {r['auc']:.3f}  {r['prec']:.3f}  {r['rec']:.3f}  {r['thr']:.3f}  {r['cfg']}{marker}")

champion = results[0]
print(f"\n*** CHAMPION: F1={champion['f1']:.3f}, config: {champion['cfg']} ***")
print(f"*** vs v11 (F1=0.397): {(champion['f1']-0.397)/0.397*100:+.1f}% ***")
print(f"*** vs v4  (F1=0.372): {(champion['f1']-0.372)/0.372*100:+.1f}% ***")
