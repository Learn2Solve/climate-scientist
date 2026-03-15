#!/usr/bin/env python3
"""
RI Classification v5: MLP classifier + SMOTE oversampling
Building on v4 environmental proxy features, adding:
1. MLPClassifier (neural net) as alternative to GBM
2. SMOTE oversampling to handle class imbalance
3. Stacking ensemble combining GBM + MLP
"""
import re, numpy as np, pandas as pd, math, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss, confusion_matrix)
from sklearn.linear_model import LogisticRegression

print("="*60)
print("RI Classification v5: MLP + SMOTE + Stacking Ensemble")
print("="*60)

# --- Climatological proxy functions (same as v4) ---
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

# --- Load data and build features ---
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
print(f"RI rate: {df['ri_label'].sum()}/{len(df)} = {df['ri_label'].mean():.4f}")

# --- Split ---
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

# --- Load data and engineer features ---
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
print(f"RI rate: {df['ri_label'].sum()}/{len(df)} = {df['ri_label'].mean():.4f}")

# --- Split ---
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

# --- SMOTE implementation (simple, no imblearn dependency) ---
def simple_smote(X, y, target_ratio=0.3, k=5, random_state=42):
    """Simple SMOTE oversampling without imblearn dependency.
    Generates synthetic minority samples by interpolating between neighbors."""
    rng = np.random.RandomState(random_state)
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    X_min = X[minority_idx]
    n_minority = len(minority_idx)
    n_majority = len(majority_idx)
    
    # Target: bring minority up to target_ratio of total
    n_synthetic = int(n_majority * target_ratio / (1 - target_ratio)) - n_minority
    if n_synthetic <= 0:
        return X, y
    
    print(f"  SMOTE: generating {n_synthetic} synthetic minority samples (from {n_minority})")
    
    # Find k nearest neighbors within minority class
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, n_minority), metric='euclidean')
    nn.fit(X_min)
    
    synthetic = []
    for _ in range(n_synthetic):
        # Pick random minority sample
        idx = rng.randint(0, n_minority)
        # Find its neighbors
        distances, neighbors = nn.kneighbors(X_min[idx:idx+1])
        # Pick random neighbor
        nn_idx = rng.randint(0, len(neighbors[0]))
        neighbor = X_min[neighbors[0][nn_idx]]
        # Interpolate
        lam = rng.uniform(0, 1)
        new_sample = X_min[idx] + lam * (neighbor - X_min[idx])
        synthetic.append(new_sample)
    
    X_new = np.vstack([X, np.array(synthetic)])
    y_new = np.concatenate([y, np.ones(n_synthetic)])
    
    # Shuffle
    perm = rng.permutation(len(X_new))
    return X_new[perm], y_new[perm]

# --- SMOTE implementation (no imblearn dependency) ---
def simple_smote(X, y, target_ratio=0.3, k=5, random_state=42):
    """Simple SMOTE: oversample minority class by interpolating between neighbors."""
    rng = np.random.RandomState(random_state)
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    n_minority = len(minority_idx)
    n_majority = len(majority_idx)
    
    # How many synthetic samples needed
    n_target = int(n_majority * target_ratio / (1 - target_ratio))
    n_synthetic = max(0, n_target - n_minority)
    
    if n_synthetic == 0 or n_minority < 2:
        return X, y
    
    X_min = X[minority_idx]
    
    # For each minority sample, find k nearest neighbors within minority class
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, n_minority), metric='euclidean')
    nn.fit(X_min)
    
    synthetic_samples = []
    for i in range(n_synthetic):
        idx = rng.randint(0, n_minority)
        neighbors = nn.kneighbors([X_min[idx]], return_distance=False)[0]
        nn_idx = neighbors[rng.randint(0, len(neighbors))]
        # Interpolate
        lam = rng.random()
        synthetic = X_min[idx] + lam * (X_min[nn_idx] - X_min[idx])
        synthetic_samples.append(synthetic)
    
    X_syn = np.array(synthetic_samples)
    y_syn = np.ones(n_synthetic, dtype=int)
    
    X_new = np.vstack([X, X_syn])
    y_new = np.concatenate([y, y_syn])
    
    # Shuffle
    perm = rng.permutation(len(X_new))
    print(f"  SMOTE: {n_minority} -> {n_minority + n_synthetic} minority samples (ratio: {(n_minority+n_synthetic)/(len(X_new)):.3f})")
    return X_new[perm], y_new[perm]
