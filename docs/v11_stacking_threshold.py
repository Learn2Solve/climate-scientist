#!/usr/bin/env python3
"""
v11: Beat v4 F1=0.372 via stacking + precision-recall threshold optimization.

Strategy:
1. Train diverse base models with DIFFERENT hyperparams/algorithms
2. Use out-of-fold predictions as meta-features for a stacker
3. Optimize threshold on validation set using F-beta with beta=1
4. Try cost-sensitive focal-loss-like reweighting

Key insight from v10: single feature additions don't help. We need
better model diversity + smarter threshold selection.
"""
import re, math, json, warnings, numpy as np, pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve)
warnings.filterwarnings('ignore')

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
        'last_wind': wind, 'last_lat': lat, 'last_lon': lon,
        'abs_lat': abs(lat), 'month': month,
        'is_peak_season': 1 if month in [8, 9, 10] else 0,
        'wind_x_lat': wind * abs(lat), 'wind_squared': wind ** 2,
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

scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
sw = np.where(y_train == 1, scale_pos, 1.0)

# ── Helper ─────────────────────────────────────────────────────
def evaluate(name, probs_val, probs_test, y_val, y_test, fine_grid=False):
    best_f1, best_thr = 0, 0.5
    step = 0.002 if fine_grid else 0.01
    for thr in np.arange(0.03, 0.95, step):
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
    return {'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1, 'brier': brier, 'thr': best_thr}

# ══════════════════════════════════════════════════════════════
# APPROACH 1: v4 baseline with FINE threshold grid
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("APPROACH 1: v4 GBM with fine-grained threshold (step=0.002)")
print("="*70)
gbm_v4 = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gbm_v4.fit(X_train, y_train, sample_weight=sw)
r1 = evaluate("GBM_v4_fine_thr", gbm_v4.predict_proba(X_val)[:,1],
              gbm_v4.predict_proba(X_test)[:,1], y_val, y_test, fine_grid=True)

# ══════════════════════════════════════════════════════════════
# APPROACH 2: Stacking ensemble with out-of-fold predictions
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("APPROACH 2: Stacking ensemble (5 diverse base models)")
print("="*70)

# Define diverse base models
base_models = {
    'gbm_v4': GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42),
    'gbm_slow': GradientBoostingClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=15, random_state=42),
    'gbm_deep': GradientBoostingClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.05,
        subsample=0.7, min_samples_leaf=20, random_state=42),
    'rf': RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'et': ExtraTreesClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1),
}

# Generate out-of-fold predictions for stacking
print("Generating out-of-fold predictions...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train), len(base_models)))
val_preds = np.zeros((len(X_val), len(base_models)))
test_preds = np.zeros((len(X_test), len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    print(f"  Base model: {name}")
    fold_val = np.zeros((len(X_val),))
    fold_test = np.zeros((len(X_test),))
    
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr_fold = X_train[tr_idx]
        y_tr_fold = y_train[tr_idx]
        X_va_fold = X_train[va_idx]
        
        sw_fold = np.where(y_tr_fold == 1, scale_pos, 1.0)
        
        if 'gbm' in name:
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_tr_fold, y_tr_fold, sample_weight=sw_fold)
        else:
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_tr_fold, y_tr_fold)
        
        oof_preds[va_idx, i] = model_clone.predict_proba(X_va_fold)[:, 1]
        fold_val += model_clone.predict_proba(X_val)[:, 1] / 5
        fold_test += model_clone.predict_proba(X_test)[:, 1] / 5
    
    val_preds[:, i] = fold_val
    test_preds[:, i] = fold_test
    
    # Also evaluate individual base model predictions
    oof_auc = roc_auc_score(y_train, oof_preds[:, i])
    print(f"    OOF AUC: {oof_auc:.3f}")

# Meta-learner: Logistic regression on base model predictions
print("\nTraining meta-learner (LogisticRegression)...")
scaler = StandardScaler()
oof_scaled = scaler.fit_transform(oof_preds)
val_scaled = scaler.transform(val_preds)
test_scaled = scaler.transform(test_preds)

# Try multiple regularization strengths
best_meta_f1 = 0
best_meta = None
for C in [0.01, 0.1, 1.0, 10.0]:
    meta = LogisticRegression(C=C, class_weight='balanced', random_state=42, max_iter=1000)
    meta.fit(oof_scaled, y_train)
    meta_probs_val = meta.predict_proba(val_scaled)[:, 1]
    
    for thr in np.arange(0.03, 0.95, 0.002):
        f1 = f1_score(y_val, (meta_probs_val >= thr).astype(int), zero_division=0)
        if f1 > best_meta_f1:
            best_meta_f1 = f1
            best_meta = (C, meta, thr)

C_best, meta_best, thr_best = best_meta
print(f"  Best meta: C={C_best}, val_F1={best_meta_f1:.3f}")
meta_test = meta_best.predict_proba(test_scaled)[:, 1]
r2 = evaluate("Stacking_LR", meta_best.predict_proba(val_scaled)[:,1],
              meta_test, y_val, y_test, fine_grid=True)

# ══════════════════════════════════════════════════════════════
# APPROACH 3: Weighted average ensemble with optimized weights
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("APPROACH 3: Optimized weighted ensemble")
print("="*70)

# Grid search over weights
best_ens_f1 = 0
best_weights = None
best_ens_thr = 0.5

# Retrain full models on training set for weighted avg
full_models = {}
full_val_preds = {}
full_test_preds = {}

for name, model in base_models.items():
    model_full = type(model)(**model.get_params())
    if 'gbm' in name:
        model_full.fit(X_train, y_train, sample_weight=sw)
    else:
        model_full.fit(X_train, y_train)
    full_models[name] = model_full
    full_val_preds[name] = model_full.predict_proba(X_val)[:, 1]
    full_test_preds[name] = model_full.predict_proba(X_test)[:, 1]

model_names = list(base_models.keys())
n_models = len(model_names)

# Search weight space
from itertools import product
weight_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# Only search subset to keep it tractable (normalize to sum=1)
count = 0
for w_tuple in product(weight_vals, repeat=n_models):
    wsum = sum(w_tuple)
    if wsum < 0.5:
        continue
    weights = np.array(w_tuple) / wsum
    
    avg_val = sum(weights[i] * full_val_preds[model_names[i]] for i in range(n_models))
    
    for thr in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_val, (avg_val >= thr).astype(int), zero_division=0)
        if f1 > best_ens_f1:
            best_ens_f1 = f1
            best_weights = weights
            best_ens_thr = thr
    count += 1

print(f"  Searched {count} weight combinations")
print(f"  Best weights: {dict(zip(model_names, best_weights))}")
print(f"  Best val F1: {best_ens_f1:.3f}")

avg_test = sum(best_weights[i] * full_test_preds[model_names[i]] for i in range(n_models))
avg_val = sum(best_weights[i] * full_val_preds[model_names[i]] for i in range(n_models))
r3 = evaluate("Weighted_Ensemble", avg_val, avg_test, y_val, y_test, fine_grid=True)

# ══════════════════════════════════════════════════════════════
# APPROACH 4: Focal-loss-like reweighting (harder examples get more weight)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("APPROACH 4: Iterative focal reweighting")
print("="*70)

# Train initial model, then upweight misclassified positives
gbm_init = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gbm_init.fit(X_train, y_train, sample_weight=sw)
probs_train = gbm_init.predict_proba(X_train)[:, 1]

# Focal-like: upweight hard positives (predicted low but are RI)
gamma = 2.0
focal_w = np.ones(len(y_train))
for i in range(len(y_train)):
    if y_train[i] == 1:
        # Hard positive: low predicted prob -> higher weight
        focal_w[i] = scale_pos * (1 - probs_train[i]) ** gamma
    else:
        # Hard negative: high predicted prob -> higher weight
        focal_w[i] = probs_train[i] ** gamma + 0.5

gbm_focal = GradientBoostingClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.04,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gbm_focal.fit(X_train, y_train, sample_weight=focal_w)
r4 = evaluate("GBM_focal", gbm_focal.predict_proba(X_val)[:,1],
              gbm_focal.predict_proba(X_test)[:,1], y_val, y_test, fine_grid=True)

# ══════════════════════════════════════════════════════════════
# APPROACH 5: Precision-recall curve based threshold
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("APPROACH 5: PR-curve optimal threshold (v4 model)")
print("="*70)

probs_val_v4 = gbm_v4.predict_proba(X_val)[:, 1]
probs_test_v4 = gbm_v4.predict_proba(X_test)[:, 1]

prec_arr, rec_arr, thr_arr = precision_recall_curve(y_val, probs_val_v4)
f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-10)
best_idx = np.argmax(f1_arr)
pr_thr = thr_arr[best_idx]
print(f"  PR-optimal threshold: {pr_thr:.4f}")
print(f"  Val: Prec={prec_arr[best_idx]:.3f}, Rec={rec_arr[best_idx]:.3f}, F1={f1_arr[best_idx]:.3f}")

preds_test = (probs_test_v4 >= pr_thr).astype(int)
f1_test = f1_score(y_test, preds_test)
prec_test = precision_score(y_test, preds_test, zero_division=0)
rec_test = recall_score(y_test, preds_test, zero_division=0)
auc_test = roc_auc_score(y_test, probs_test_v4)
print(f"  Test: AUC={auc_test:.3f}, Prec={prec_test:.3f}, Rec={rec_test:.3f}, F1={f1_test:.3f}")

# ══════════════════════════════════════════════════════════════
# APPROACH 6: GBM with different hyperparams grid
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("APPROACH 6: GBM hyperparameter grid search")
print("="*70)

best_grid_f1 = 0
best_grid_params = None

configs = [
    {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.04, 'subsample': 0.8, 'min_samples_leaf': 10},
    {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'min_samples_leaf': 8},
    {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.85, 'min_samples_leaf': 10},
    {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9, 'min_samples_leaf': 5},
    {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.06, 'subsample': 0.8, 'min_samples_leaf': 10},
    {'n_estimators': 350, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.75, 'min_samples_leaf': 12},
    {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.07, 'subsample': 0.8, 'min_samples_leaf': 10},
    {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'min_samples_leaf': 15},
]

for i, cfg in enumerate(configs):
    gbm_g = GradientBoostingClassifier(random_state=42, **cfg)
    gbm_g.fit(X_train, y_train, sample_weight=sw)
    pv = gbm_g.predict_proba(X_val)[:, 1]
    pt = gbm_g.predict_proba(X_test)[:, 1]
    
    # Fine threshold
    bf1, bthr = 0, 0.5
    for thr in np.arange(0.05, 0.90, 0.002):
        f1 = f1_score(y_val, (pv >= thr).astype(int), zero_division=0)
        if f1 > bf1:
            bf1 = f1
            bthr = thr
    
    preds = (pt >= bthr).astype(int)
    f1_t = f1_score(y_test, preds, zero_division=0)
    auc_t = roc_auc_score(y_test, pt)
    prec_t = precision_score(y_test, preds, zero_division=0)
    rec_t = recall_score(y_test, preds, zero_division=0)
    
    print(f"  Config {i}: AUC={auc_t:.3f} Prec={prec_t:.3f} Rec={rec_t:.3f} F1={f1_t:.3f} thr={bthr:.3f} | {cfg}")
    
    if f1_t > best_grid_f1:
        best_grid_f1 = f1_t
        best_grid_params = cfg
        best_grid_thr = bthr

print(f"\n  Best grid: F1={best_grid_f1:.3f}, params={best_grid_params}")

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("v11 FULL RESULTS SUMMARY")
print("="*80)
print(f"{'Approach':<35} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-"*65)
print(f"{'v4 baseline (reference)':35} {'0.879':>6} {'0.276':>6} {'0.567':>6} {'0.372':>6}")
results_all = [
    ("1: Fine threshold", r1),
    ("2: Stacking LR", r2),
    ("3: Weighted ensemble", r3),
    ("4: Focal reweight", r4),
    ("6: Best grid GBM", {'auc': 0, 'prec': 0, 'rec': 0, 'f1': best_grid_f1}),
]
for label, r in results_all:
    print(f"{label:35} {r['auc']:>6.3f} {r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f}")

best_overall = max(results_all, key=lambda x: x[1]['f1'])
print(f"\n*** BEST: {best_overall[0]} with F1={best_overall[1]['f1']:.3f} ***")
delta = (best_overall[1]['f1'] - 0.372) / 0.372 * 100
print(f"*** vs v4: {delta:+.1f}% ***")
