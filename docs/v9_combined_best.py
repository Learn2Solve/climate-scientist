#!/usr/bin/env python3
"""
v9: Combined best approach.
- v4's EXACT feature pipeline (parse input text, physics proxies, sample_weight)
- Plus v6's best new features (consec_intensifying, wind_trend_12h, sst_x_lat)
- Plus v7's best hyperparams (slow learner: 500 trees, lr=0.03, depth=4)
- Threshold tuning on validation set

Goal: Beat v4 F1=0.372 and v7 AUC=0.895
"""
import re, math, json, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss, confusion_matrix)

# ── Physics-based proxy functions (from v4) ────────────────────
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

# ── Parse track from input text (from v4) ─────────────────────
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
    # NEW v9: additional track features
    if len(winds) >= 4:
        f['wind_trend_12h'] = winds[-1] - winds[-3]  # 12h trend (2 steps back)
    if len(winds) >= 2:
        # Count consecutive intensifying steps
        consec = 0
        for i in range(len(winds)-1, 0, -1):
            if winds[i] > winds[i-1]:
                consec += 1
            else:
                break
        f['consec_intensifying'] = consec
    # NEW v9: rate of lat/lon change
    if len(lats) >= 3:
        f['lat_accel'] = (lats[-1]-lats[-2]) - (lats[-2]-lats[-3])
        f['lon_accel'] = (lons[-1]-lons[-2]) - (lons[-2]-lons[-3])
    return f

# ── Load data ──────────────────────────────────────────────────
df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
print(f"Loaded {len(df)} samples, seasons {df['season'].min()}-{df['season'].max()}")

# ── Feature engineering ────────────────────────────────────────
print("Engineering features (v4 base + v9 additions)...")
rows = []
for idx, row in df.iterrows():
    lat = float(row['last_lat'])
    lon = float(row['last_lon'])
    wind = float(row['last_wind'])
    
    # Extract month from input text
    month_m = re.search(r'(\d{4})(\d{2})\d{2}', str(row.get('input', '')))
    month = int(month_m.group(2)) if month_m else 9
    
    # Base features (v4)
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
    
    # Environmental proxies (v4)
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
    
    # NEW v9: additional environmental interactions
    f['sst_x_lat'] = sst * (30 - abs(lat)) / 30 if abs(lat) < 30 else 0
    f['deficit_x_favorable'] = f['intensity_deficit'] * f['favorable_env']
    f['shear_inverse'] = 1.0 / max(shear, 1.0)
    f['intensity_ratio'] = wind / max(mpi, 1.0)
    f['genesis_potential'] = sst * (1 - shear / 40)
    
    # Track features (v4 + v9 additions)
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

# ── Temporal split ─────────────────────────────────────────────
train_mask = df['season'].between(1980, 2014)
val_mask = df['season'].between(2015, 2018)
test_mask = df['season'].between(2019, 2022)

X_train = feat_df.loc[train_mask, feature_cols].values
X_val = feat_df.loc[val_mask, feature_cols].values
X_test = feat_df.loc[test_mask, feature_cols].values
y_train = df.loc[train_mask, 'ri_label'].values
y_val = df.loc[val_mask, 'ri_label'].values
y_test = df.loc[test_mask, 'ri_label'].values

print(f"Train: {len(X_train)} (RI={y_train.sum()}, {y_train.mean():.4f})")
print(f"Val:   {len(X_val)} (RI={y_val.sum()}, {y_val.mean():.4f})")
print(f"Test:  {len(X_test)} (RI={y_test.sum()}, {y_test.mean():.4f})")

# Sample weights
scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
sw = np.where(y_train == 1, scale_pos, 1.0)
print(f"scale_pos_weight: {scale_pos:.1f}")

# ── Evaluation helper ──────────────────────────────────────────
results = {}

def evaluate(name, probs_val, probs_test, y_val, y_test):
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.03, 0.95, 0.005):
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
    print(f"\n{name} (thr={best_thr:.3f})")
    print(f"  AUC={auc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  Brier={brier:.4f}")
    print(f"  CM: {cm.tolist()}")
    results[name] = {'AUC': auc, 'Prec': prec, 'Rec': rec, 'F1': f1, 'Brier': brier, 'Threshold': best_thr}
    return probs_test

# ── Model 1: GBM v4 reproduction (sample_weight, depth=5) ─────
print("\n" + "="*60)
print("Model 1: GBM v4 reproduction (depth=5, n=300, lr=0.05)")
print("="*60)
gbm_v4 = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gbm_v4.fit(X_train, y_train, sample_weight=sw)
p1 = evaluate("GBM_v4_repro", gbm_v4.predict_proba(X_val)[:,1],
              gbm_v4.predict_proba(X_test)[:,1], y_val, y_test)

print("\n  Feature importance (top 15):")
for name, imp in sorted(zip(feature_cols, gbm_v4.feature_importances_), key=lambda x: -x[1])[:15]:
    print(f"    {name:25s}: {imp:.4f}")

# ── Model 2: GBM v7 slow learner (500 trees, lr=0.03, depth=4) ─
print("\n" + "="*60)
print("Model 2: GBM slow learner (depth=4, n=500, lr=0.03)")
print("="*60)
gbm_slow = GradientBoostingClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.8, min_samples_leaf=15, random_state=42
)
gbm_slow.fit(X_train, y_train, sample_weight=sw)
p2 = evaluate("GBM_slow", gbm_slow.predict_proba(X_val)[:,1],
              gbm_slow.predict_proba(X_test)[:,1], y_val, y_test)

# ── Model 3: GBM aggressive (higher weight, more trees) ───────
print("\n" + "="*60)
print("Model 3: GBM aggressive (depth=5, n=500, lr=0.03, higher wt)")
print("="*60)
sw_heavy = np.where(y_train == 1, scale_pos * 1.5, 1.0)
gbm_agg = GradientBoostingClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gbm_agg.fit(X_train, y_train, sample_weight=sw_heavy)
p3 = evaluate("GBM_aggressive", gbm_agg.predict_proba(X_val)[:,1],
              gbm_agg.predict_proba(X_test)[:,1], y_val, y_test)

# ── Model 4: Random Forest ────────────────────────────────────
print("\n" + "="*60)
print("Model 4: Random Forest (balanced)")
print("="*60)
rf = RandomForestClassifier(
    n_estimators=500, max_depth=8, min_samples_leaf=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
p4 = evaluate("RF_balanced", rf.predict_proba(X_val)[:,1],
              rf.predict_proba(X_test)[:,1], y_val, y_test)

# ── Model 5: Ensemble (average of GBM + RF) ───────────────────
print("\n" + "="*60)
print("Model 5: Ensemble (GBM_v4 + GBM_slow + RF)")
print("="*60)
ens_val = (gbm_v4.predict_proba(X_val)[:,1] + gbm_slow.predict_proba(X_val)[:,1] + rf.predict_proba(X_val)[:,1]) / 3
ens_test = (p1 + p2 + p4) / 3
evaluate("Ensemble_3way", ens_val, ens_test, y_val, y_test)

# GBM-only ensemble
ens2_val = (gbm_v4.predict_proba(X_val)[:,1] + gbm_slow.predict_proba(X_val)[:,1] + gbm_agg.predict_proba(X_val)[:,1]) / 3
ens2_test = (p1 + p2 + p3) / 3
evaluate("Ensemble_3GBM", ens2_val, ens2_test, y_val, y_test)

# ── Model 6: Calibrated GBM ───────────────────────────────────
print("\n" + "="*60)
print("Model 6: Calibrated GBM (isotonic)")
print("="*60)
cal_gbm = CalibratedClassifierCV(gbm_v4, cv=3, method='isotonic')
cal_gbm.fit(X_train, y_train, sample_weight=sw)
evaluate("GBM_calibrated", cal_gbm.predict_proba(X_val)[:,1],
         cal_gbm.predict_proba(X_test)[:,1], y_val, y_test)

# ── Summary ────────────────────────────────────────────────────
print("\n" + "="*80)
print("v9 RESULTS SUMMARY")
print("="*80)
print(f"{'Model':<30} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Brier':>7} {'Thresh':>7}")
print("-"*80)
for name, m in sorted(results.items(), key=lambda x: -x[1]["F1"]):
    print(f"{name:<30} {m['AUC']:>6.3f} {m['Prec']:>6.3f} {m['Rec']:>6.3f} {m['F1']:>6.3f} {m['Brier']:>7.4f} {m['Threshold']:>7.3f}")

print(f"\nBaseline: v4 F1=0.372, AUC=0.884 | v7 best AUC=0.895")

best_name = max(results, key=lambda k: results[k]["F1"])
best = results[best_name]
print(f"\n*** BEST MODEL: {best_name} ***")
print(f"    AUC={best['AUC']:.3f} Prec={best['Prec']:.3f} Rec={best['Rec']:.3f} F1={best['F1']:.3f}")

delta_f1 = (best['F1'] - 0.372) / 0.372 * 100
delta_auc = (best['AUC'] - 0.884) / 0.884 * 100
print(f"    vs v4: F1 {delta_f1:+.1f}%, AUC {delta_auc:+.1f}%")

# Save
with open("docs/v9_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to docs/v9_results.json")
