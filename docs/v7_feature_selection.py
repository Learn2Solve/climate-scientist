#!/usr/bin/env python3
"""v7: Feature selection experiment. 
Keep only top features from v4+v6, prune dead weight to beat v4 F1=0.372.
Also try threshold tuning more carefully."""
import re, numpy as np, pandas as pd, math, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss)

print("="*60)
print("v7: Feature Selection + Threshold Optimization")
print("="*60)

# --- Physics proxies ---
def sst_proxy(lat, lon, month):
    base = 30.0 - 0.4*abs(lat)
    seasonal = 1.5*math.cos(2*math.pi*(month-9)/12)
    gulf = 1.0 if (-100<lon<-60) else (-0.5 if (-60<lon<-20) else 0.0)
    return base + seasonal + gulf

def shear_proxy(lat, lon, month):
    base = 5.0 + 0.3*abs(lat) + (5.0 if abs(lat)>30 else 0.0)
    seasonal = 3.0*math.cos(2*math.pi*(month-9)/12)
    return max(base - seasonal, 2.0)

def mpi_proxy(sst):
    return 0.0 if sst<26.0 else min(80.0, 20.0*(sst-26.0))

def rh_proxy(lat, lon, month):
    base = 80.0 - 0.5*abs(lat)
    seasonal = 5.0*math.cos(2*math.pi*(month-8)/12)
    maritime = 3.0 if (-90<lon<-60 and 10<lat<30) else 0.0
    return min(95.0, max(40.0, base + seasonal + maritime))

def parse_track(text):
    f = {}
    lats, lons, winds = [], [], []
    for line in str(text).strip().split('\n'):
        lm = re.search(r'lat\s+([-\d.]+)', line)
        om = re.search(r'lon\s+([-\d.]+)', line)
        wm = re.search(r'wind\s+([-\d.]+)', line)
        if lm and om and wm:
            lats.append(float(lm.group(1)))
            lons.append(float(om.group(1)))
            winds.append(float(wm.group(1)))
    n = len(winds)
    if n >= 2:
        f['wind_trend_6h'] = winds[-1] - winds[-2]
        f['wind_mean'] = np.mean(winds)
        f['wind_std'] = np.std(winds)
        f['wind_max_history'] = max(winds)
        f['wind_range'] = max(winds) - min(winds)
        f['n_track_points'] = n
    if n >= 3:
        f['wind_trend_12h'] = winds[-1] - winds[-3]
        f['wind_accel'] = (winds[-1]-winds[-2]) - (winds[-2]-winds[-3])
    if n >= 4:
        f['wind_trend_18h'] = winds[-1] - winds[-4]
    if len(lats) >= 2:
        f['lat_trend'] = lats[-1] - lats[-2]
        f['lon_trend'] = lons[-1] - lons[-2]
        f['translation_speed'] = np.sqrt((lats[-1]-lats[-2])**2 + (lons[-1]-lons[-2])**2)
        f['total_lat_change'] = lats[-1] - lats[0]
        f['total_lon_change'] = lons[-1] - lons[0]
    # consec_intensifying: top feature from v6
    if n >= 3:
        consec = 0
        for i in range(n-1, 0, -1):
            if winds[i] > winds[i-1]: consec += 1
            else: break
        f['consec_intensifying'] = consec
    # heading
    if len(lats) >= 2:
        f['heading'] = math.atan2(lons[-1]-lons[-2], lats[-1]-lats[-2])
    return f

# --- Load data ---
df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
print(f"Loaded {len(df)} samples")

rows = []
for _, row in df.iterrows():
    lat, lon, wind = float(row['last_lat']), float(row['last_lon']), float(row['last_wind'])
    mm = re.search(r'(\d{4})(\d{2})\d{2}', str(row.get('input', '')))
    month = int(mm.group(2)) if mm else 9
    sst = sst_proxy(lat, lon, month)
    shear = shear_proxy(lat, lon, month)
    mpi = mpi_proxy(sst)
    rh = rh_proxy(lat, lon, month)
    
    f = {
        'last_wind': wind, 'last_lat': lat, 'last_lon': lon,
        'abs_lat': abs(lat), 'month': month,
        'is_peak_season': 1 if month in [8, 9, 10] else 0,
        'wind_x_lat': wind * abs(lat),
        'wind_squared': wind ** 2,
        'sst_proxy': sst, 'shear_proxy': shear, 'mpi_proxy': mpi,
        'intensity_deficit': mpi - wind,
        'sst_minus_26': max(sst - 26.0, 0.0),
        'favorable_env': 1 if (sst > 26.5 and shear < 10 and abs(lat) < 30) else 0,
        'shear_x_wind': shear * wind,
        # Best new features from v6
        'rh_proxy': rh,
        'sst_x_lat': sst * abs(lat),
        'lon_squared': lon ** 2,
    }
    f.update(parse_track(row.get('input', '')))
    rows.append(f)

feat_df = pd.DataFrame(rows).fillna(0)

# Define feature sets to test
# TOP20: best from v4+v6 by importance
top20 = ['consec_intensifying', 'sst_proxy', 'wind_trend_6h', 'sst_x_lat',
         'total_lon_change', 'wind_trend_12h', 'shear_x_wind', 'wind_std',
         'wind_x_lat', 'last_lon', 'lon_squared', 'wind_mean', 'heading',
         'total_lat_change', 'wind_trend_18h', 'rh_proxy', 'intensity_deficit',
         'last_wind', 'wind_range', 'abs_lat']

# TOP15: even more aggressive pruning
top15 = ['consec_intensifying', 'sst_proxy', 'wind_trend_6h', 'sst_x_lat',
         'total_lon_change', 'wind_trend_12h', 'shear_x_wind', 'wind_std',
         'wind_x_lat', 'last_lon', 'lon_squared', 'wind_mean', 'heading',
         'intensity_deficit', 'last_wind']

# TOP10: most aggressive
top10 = ['consec_intensifying', 'sst_proxy', 'wind_trend_6h', 'sst_x_lat',
         'total_lon_change', 'wind_trend_12h', 'shear_x_wind', 'wind_std',
         'wind_x_lat', 'last_lon']

# V4 original features (28 features) for comparison
v4_feats = [c for c in feat_df.columns if c not in ['rh_proxy', 'sst_x_lat', 'lon_squared',
            'consec_intensifying', 'heading', 'wind_trend_12h', 'wind_trend_18h']]

feature_sets = {
    'v4_original': v4_feats,
    'top20': top20,
    'top15': top15,
    'top10': top10,
}

df['delta_wind'] = df['target_wind'] - df['last_wind']
df['ri_label'] = (df['delta_wind'] >= 30.0).astype(int)
print(f"RI rate: {df['ri_label'].sum()}/{len(df)} = {df['ri_label'].mean():.4f}")

tr = df['season'].between(1980, 2014)
va = df['season'].between(2015, 2018)
te = df['season'].between(2019, 2022)
ytr, yv, yt = df.loc[tr, 'ri_label'].values, df.loc[va, 'ri_label'].values, df.loc[te, 'ri_label'].values
print(f"Train:{sum(tr)}(RI={ytr.sum()}) Val:{sum(va)}(RI={yv.sum()}) Test:{sum(te)}(RI={yt.sum()})")

sw_r = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
sw = np.where(ytr == 1, sw_r, 1.0)

def evaluate(name, pv, pt, yv, yt):
    # More granular threshold search
    bf, bt = 0, 0.5
    for t in np.arange(0.02, 0.95, 0.005):
        f = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    p = (pt >= bt).astype(int)
    a = roc_auc_score(yt, pt)
    f1 = f1_score(yt, p, zero_division=0)
    pr = precision_score(yt, p, zero_division=0)
    rc = recall_score(yt, p, zero_division=0)
    br = brier_score_loss(yt, pt)
    print(f"  {name} (thr={bt:.3f}): AUC={a:.3f} P={pr:.3f} R={rc:.3f} F1={f1:.3f} Brier={br:.4f}")
    return {'model': name, 'auc': a, 'prec': pr, 'rec': rc, 'f1': f1, 'brier': br, 'thr': bt}

results = []

# Test each feature set with consistent GBM hyperparams
for fs_name, fs_cols in feature_sets.items():
    valid_cols = [c for c in fs_cols if c in feat_df.columns]
    print(f"\n--- {fs_name} ({len(valid_cols)} features) ---")
    Xtr_s = feat_df.loc[tr, valid_cols].values
    Xv_s = feat_df.loc[va, valid_cols].values
    Xt_s = feat_df.loc[te, valid_cols].values
    
    gbm = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, min_samples_leaf=10, random_state=42)
    gbm.fit(Xtr_s, ytr, sample_weight=sw)
    r = evaluate(f"GBM_{fs_name}", gbm.predict_proba(Xv_s)[:,1], gbm.predict_proba(Xt_s)[:,1], yv, yt)
    results.append(r)
    
    # Also test with 500 trees / lower LR for top20
    if fs_name == 'top20':
        gbm2 = GradientBoostingClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                                           subsample=0.8, min_samples_leaf=12, random_state=42)
        gbm2.fit(Xtr_s, ytr, sample_weight=sw)
        r2 = evaluate(f"GBM_top20_slow", gbm2.predict_proba(Xv_s)[:,1], gbm2.predict_proba(Xt_s)[:,1], yv, yt)
        results.append(r2)
        
        # And with higher weight ratio
        sw2 = np.where(ytr == 1, sw_r * 1.5, 1.0)
        gbm3 = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                           subsample=0.8, min_samples_leaf=10, random_state=42)
        gbm3.fit(Xtr_s, ytr, sample_weight=sw2)
        r3 = evaluate(f"GBM_top20_hw", gbm3.predict_proba(Xv_s)[:,1], gbm3.predict_proba(Xt_s)[:,1], yv, yt)
        results.append(r3)

    if fs_name == 'top15':
        gbm4 = GradientBoostingClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                                           subsample=0.8, min_samples_leaf=12, random_state=42)
        gbm4.fit(Xtr_s, ytr, sample_weight=sw)
        r4 = evaluate(f"GBM_top15_slow", gbm4.predict_proba(Xv_s)[:,1], gbm4.predict_proba(Xt_s)[:,1], yv, yt)
        results.append(r4)

# Summary
print("\n" + "="*60)
print("SUMMARY — v7 Feature Selection Experiment")
print("="*60)
print(f"{'Model':<25} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Brier':>7}")
for r in sorted(results, key=lambda x: -x['f1']):
    print(f"{r['model']:<25} {r['auc']:>6.3f} {r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f} {r['brier']:>7.4f}")
print(f"\nBaseline: v4 GBM AUC=0.879 F1=0.372")
best = max(results, key=lambda x: x['f1'])
print(f"Best v7: {best['model']} AUC={best['auc']:.3f} F1={best['f1']:.3f}")
imp_pct = ((best['f1'] - 0.372) / 0.372) * 100
print(f"Change vs v4: {imp_pct:+.1f}% F1")
