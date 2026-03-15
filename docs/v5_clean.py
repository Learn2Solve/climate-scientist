#!/usr/bin/env python3
"""RI Classification v5: MLP + SMOTE + Stacking"""
import re, numpy as np, pandas as pd, math, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression

print("=" * 60)
print("RI Classification v5: MLP + SMOTE + Stacking Ensemble")
print("=" * 60)


def sst_proxy(lat, lon, month):
    base = 30.0 - 0.4 * abs(lat)
    seasonal = 1.5 * math.cos(2 * math.pi * (month - 9) / 12)
    gulf = 1.0 if (-100 < lon < -60) else (-0.5 if (-60 < lon < -20) else 0.0)
    return base + seasonal + gulf

def shear_proxy(lat, lon, month):
    base = 5.0 + 0.3 * abs(lat) + (5.0 if abs(lat) > 30 else 0.0)
    seasonal = 3.0 * math.cos(2 * math.pi * (month - 9) / 12)
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


# --- Load data and build features ---
df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
print(f"Loaded {len(df)} samples")

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
print(f"Features ({len(feature_cols)}): {feature_cols}")

df['delta_wind'] = df['target_wind'] - df['last_wind']
df['ri_label'] = (df['delta_wind'] >= 30.0).astype(int)
print(f"RI rate: {df['ri_label'].sum()}/{len(df)} = {df['ri_label'].mean():.4f}")

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


# --- Load data and build features ---
df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
print(f"Loaded {len(df)} samples")

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
print(f"Features ({len(feature_cols)}): {feature_cols}")

df['delta_wind'] = df['target_wind'] - df['last_wind']
df['ri_label'] = (df['delta_wind'] >= 30.0).astype(int)
print(f"RI rate: {df['ri_label'].sum()}/{len(df)} = {df['ri_label'].mean():.4f}")


# --- Load data and build features ---
df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
print(f"Loaded {len(df)} samples")

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
print(f"Features ({len(feature_cols)}): {feature_cols}")

df['delta_wind'] = df['target_wind'] - df['last_wind']
df['ri_label'] = (df['delta_wind'] >= 30.0).astype(int)
print(f"RI rate: {df['ri_label'].sum()}/{len(df)} = {df['ri_label'].mean():.4f}")

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


# --- Simple SMOTE (no imblearn needed) ---
def simple_smote(X, y, target_ratio=0.3, k=5, random_state=42):
    rng = np.random.RandomState(random_state)
    min_idx = np.where(y == 1)[0]
    maj_idx = np.where(y == 0)[0]
    X_min = X[min_idx]
    n_synth = int(len(maj_idx) * target_ratio / (1 - target_ratio)) - len(min_idx)
    if n_synth <= 0:
        return X, y
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, len(min_idx))).fit(X_min)
    synth = []
    for _ in range(n_synth):
        i = rng.randint(len(min_idx))
        dists, idxs = nn.kneighbors(X_min[i:i+1])
        j = idxs[0, rng.randint(1, len(idxs[0]))] if len(idxs[0]) > 1 else idxs[0, 0]
        lam = rng.random()
        synth.append(X_min[i] + lam * (X_min[j] - X_min[i]))
    X_new = np.vstack([X, np.array(synth)])
    y_new = np.concatenate([y, np.ones(n_synth)])
    print(f"  SMOTE: {len(min_idx)} -> {len(min_idx)+n_synth} minority, total {len(X_new)}")
    return X_new, y_new

def evaluate(name, probs_val, probs_test, y_v, y_t):
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_v, (probs_val >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    preds = (probs_test >= best_thr).astype(int)
    auc = roc_auc_score(y_t, probs_test)
    f1 = f1_score(y_t, preds, zero_division=0)
    prec = precision_score(y_t, preds, zero_division=0)
    rec = recall_score(y_t, preds, zero_division=0)
    brier = brier_score_loss(y_t, probs_test)
    print(f"\n{name} (thr={best_thr:.2f})")
    print(f"  AUC={auc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  Brier={brier:.4f}")
    return {'model': name, 'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1, 'brier': brier, 'thr': best_thr}

results = []
