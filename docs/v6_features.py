#!/usr/bin/env python3
"""RI Classification v6: Enhanced feature engineering to beat v4 F1=0.372.
New features: RH proxy, OHC proxy, polynomial interactions, multi-step wind trends."""
import re, numpy as np, pandas as pd, math, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss)

print("="*60)
print("RI Classification v6: Enhanced Feature Engineering")
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
    """Relative humidity at 850 hPa proxy. Higher in tropics, peak season, near ITCZ."""
    base = 80.0 - 0.5*abs(lat)
    seasonal = 5.0*math.cos(2*math.pi*(month-8)/12)
    maritime = 3.0 if (-90<lon<-60 and 10<lat<30) else 0.0  # Caribbean moisture
    return min(95.0, max(40.0, base + seasonal + maritime))

def ohc_proxy(lat, lon, month, sst):
    """Ocean heat content proxy (kJ/cm^2). Based on SST, warm pools, eddies."""
    base = max(0, (sst - 26.0) * 30.0)
    warm_pool = 20.0 if (-100<lon<-80 and 20<lat<30) else 0.0  # Gulf/Caribbean warm pool
    seasonal = 10.0*math.cos(2*math.pi*(month-9)/12)
    return max(0, base + warm_pool + seasonal)

def parse_track(text):
    f = {}
    lats, lons, winds, times = [], [], [], []
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
        f['wind_min_history'] = min(winds)
        f['wind_range'] = max(winds) - min(winds)
        f['n_track_points'] = n
    if n >= 3:
        f['wind_trend_12h'] = winds[-1] - winds[-3]
        f['wind_accel'] = (winds[-1]-winds[-2]) - (winds[-2]-winds[-3])
    if n >= 4:
        f['wind_trend_18h'] = winds[-1] - winds[-4]
    if n >= 5:
        f['wind_trend_24h'] = winds[-1] - winds[-5]
        # Sustained intensification: how many consecutive positive changes
        consec = 0
        for i in range(n-1, 0, -1):
            if winds[i] > winds[i-1]: consec += 1
            else: break
        f['consec_intensifying'] = consec
    if len(lats) >= 2:
        f['lat_trend'] = lats[-1] - lats[-2]
        f['lon_trend'] = lons[-1] - lons[-2]
        f['translation_speed'] = np.sqrt((lats[-1]-lats[-2])**2 + (lons[-1]-lons[-2])**2)
        f['total_lat_change'] = lats[-1] - lats[0]
        f['total_lon_change'] = lons[-1] - lons[0]
        # Heading (approximate)
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
    ohc = ohc_proxy(lat, lon, month, sst)
    
    f = {
        # Base features (from v4)
        'last_wind': wind, 'last_lat': lat, 'last_lon': lon,
        'abs_lat': abs(lat), 'month': month,
        'is_peak_season': 1 if month in [8, 9, 10] else 0,
        'wind_x_lat': wind * abs(lat), 'wind_squared': wind ** 2,
        'sst_proxy': sst, 'shear_proxy': shear, 'mpi_proxy': mpi,
        'intensity_deficit': mpi - wind,
        'sst_minus_26': max(sst - 26.0, 0.0),
        'favorable_env': 1 if (sst > 26.5 and shear < 10 and abs(lat) < 30) else 0,
        'shear_x_wind': shear * wind,
        # NEW v6 features
        'rh_proxy': rh,
        'ohc_proxy': ohc,
        'rh_x_sst': rh * sst,          # moisture-warmth interaction
        'ohc_x_deficit': ohc * max(mpi - wind, 0),  # energy potential interaction
        'shear_x_deficit': shear * max(mpi - wind, 0),  # shear suppressing potential
        'low_shear_warm': 1 if (shear < 8 and sst > 27.5) else 0,
        'wind_pct_mpi': wind / max(mpi, 1.0),  # fraction of MPI reached
        'intensification_potential': max(mpi - wind, 0) * max(0, 1 - shear/15),
        'sst_x_lat': sst * abs(lat),
        'lon_squared': lon ** 2,
    }
    f.update(parse_track(row.get('input', '')))
    rows.append(f)

feat_df = pd.DataFrame(rows).fillna(0)
feature_cols = sorted(feat_df.columns.tolist())
print(f"Features ({len(feature_cols)}): {feature_cols[:10]}... (+{len(feature_cols)-10} more)")

df['delta_wind'] = df['target_wind'] - df['last_wind']
df['ri_label'] = (df['delta_wind'] >= 30.0).astype(int)
print(f"RI rate: {df['ri_label'].sum()}/{len(df)} = {df['ri_label'].mean():.4f}")

tr = df['season'].between(1980, 2014)
va = df['season'].between(2015, 2018)
te = df['season'].between(2019, 2022)
Xtr, Xv, Xt = feat_df.loc[tr, feature_cols].values, feat_df.loc[va, feature_cols].values, feat_df.loc[te, feature_cols].values
ytr, yv, yt = df.loc[tr, 'ri_label'].values, df.loc[va, 'ri_label'].values, df.loc[te, 'ri_label'].values
print(f"Train:{len(Xtr)}(RI={ytr.sum()}) Val:{len(Xv)}(RI={yv.sum()}) Test:{len(Xt)}(RI={yt.sum()})")

# --- Evaluation ---
def evaluate(name, pv, pt, yv, yt):
    bf, bt = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    p = (pt >= bt).astype(int)
    a = roc_auc_score(yt, pt); f1 = f1_score(yt, p, zero_division=0)
    pr = precision_score(yt, p, zero_division=0); rc = recall_score(yt, p, zero_division=0)
    br = brier_score_loss(yt, pt)
    print(f"  {name} (thr={bt:.2f}): AUC={a:.3f} Prec={pr:.3f} Rec={rc:.3f} F1={f1:.3f} Brier={br:.4f}")
    return {'model': name, 'auc': a, 'prec': pr, 'rec': rc, 'f1': f1, 'brier': br, 'thr': bt}

results = []
sw_r = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
sw = np.where(ytr == 1, sw_r, 1.0)

# M1: GBM with v6 features (default hyperparams from v4)
print("\n--- M1: GBM v6 features (v4 hyperparams) ---")
gbm1 = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, min_samples_leaf=10, random_state=42)
gbm1.fit(Xtr, ytr, sample_weight=sw)
results.append(evaluate("GBM_v6_default", gbm1.predict_proba(Xv)[:,1], gbm1.predict_proba(Xt)[:,1], yv, yt))

# M2: GBM with more trees and lower LR
print("\n--- M2: GBM v6 features (more trees) ---")
gbm2 = GradientBoostingClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                                   subsample=0.8, min_samples_leaf=15, random_state=42)
gbm2.fit(Xtr, ytr, sample_weight=sw)
results.append(evaluate("GBM_v6_500trees", gbm2.predict_proba(Xv)[:,1], gbm2.predict_proba(Xt)[:,1], yv, yt))

# M3: GBM deeper trees
print("\n--- M3: GBM v6 features (deeper) ---")
gbm3 = GradientBoostingClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
                                   subsample=0.7, min_samples_leaf=20, random_state=42)
gbm3.fit(Xtr, ytr, sample_weight=sw)
results.append(evaluate("GBM_v6_deep", gbm3.predict_proba(Xv)[:,1], gbm3.predict_proba(Xt)[:,1], yv, yt))

# Feature importances for best model
print("\n--- Feature Importances (M1) ---")
imp = sorted(zip(feature_cols, gbm1.feature_importances_), key=lambda x: -x[1])
for name, val in imp[:15]:
    print(f"  {name}: {val:.4f}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Model':<25} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Brier':>7}")
for r in sorted(results, key=lambda x: -x['f1']):
    print(f"{r['model']:<25} {r['auc']:>6.3f} {r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f} {r['brier']:>7.4f}")
print(f"\nv4 best: GBM AUC=0.879 F1=0.372")
best = max(results, key=lambda x: x['f1'])
print(f"v6 best: {best['model']} AUC={best['auc']:.3f} F1={best['f1']:.3f}")
imp_pct = ((best['f1'] - 0.372) / 0.372) * 100
print(f"Improvement over v4: {imp_pct:+.1f}% F1")

# New features contribution check
print("\n--- New v6 feature importances ---")
new_feats = ['rh_proxy','ohc_proxy','rh_x_sst','ohc_x_deficit','shear_x_deficit',
             'low_shear_warm','wind_pct_mpi','intensification_potential','sst_x_lat',
             'lon_squared','wind_trend_12h','wind_trend_18h','wind_trend_24h',
             'consec_intensifying','heading']
for name, val in imp:
    if name in new_feats:
        print(f"  {name}: {val:.4f}")
