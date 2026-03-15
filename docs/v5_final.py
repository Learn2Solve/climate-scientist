#!/usr/bin/env python3
"""RI Classification v5: MLP + SMOTE + Stacking. Complete script."""
import re, numpy as np, pandas as pd, math, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss)
from sklearn.linear_model import LogisticRegression
print("="*60)
print("RI Classification v5: MLP + SMOTE + Stacking")
print("="*60)

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
    if len(winds)>=2:
        f['wind_trend_6h']=winds[-1]-winds[-2]; f['wind_mean']=np.mean(winds)
        f['wind_std']=np.std(winds); f['wind_max_history']=max(winds)
        f['wind_min_history']=min(winds); f['wind_range']=max(winds)-min(winds)
        f['n_track_points']=len(winds)
    if len(lats)>=2:
        f['lat_trend']=lats[-1]-lats[-2]; f['lon_trend']=lons[-1]-lons[-2]
        f['translation_speed']=np.sqrt((lats[-1]-lats[-2])**2+(lons[-1]-lons[-2])**2)
        f['total_lat_change']=lats[-1]-lats[0]; f['total_lon_change']=lons[-1]-lons[0]
    if len(winds)>=3:
        f['wind_accel']=(winds[-1]-winds[-2])-(winds[-2]-winds[-3])
    return f

df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
print(f"Loaded {len(df)} samples")
rows = []
for _, row in df.iterrows():
    lat,lon,wind = float(row['last_lat']),float(row['last_lon']),float(row['last_wind'])
    mm = re.search(r'(\d{4})(\d{2})\d{2}', str(row.get('input','')))
    month = int(mm.group(2)) if mm else 9
    sst=sst_proxy(lat,lon,month); shear=shear_proxy(lat,lon,month); mpi=mpi_proxy(sst)
    f = {'last_wind':wind,'last_lat':lat,'last_lon':lon,'abs_lat':abs(lat),
         'month':month,'is_peak_season':1 if month in [8,9,10] else 0,
         'wind_x_lat':wind*abs(lat),'wind_squared':wind**2,
         'sst_proxy':sst,'shear_proxy':shear,'mpi_proxy':mpi,
         'intensity_deficit':mpi-wind,'sst_minus_26':max(sst-26.0,0.0),
         'favorable_env':1 if (sst>26.5 and shear<10 and abs(lat)<30) else 0,
         'shear_x_wind':shear*wind}
    f.update(parse_track(row.get('input',''))); rows.append(f)
feat_df = pd.DataFrame(rows).fillna(0)
feature_cols = sorted(feat_df.columns.tolist())
print(f"Features ({len(feature_cols)})")
df['delta_wind']=df['target_wind']-df['last_wind']
df['ri_label']=(df['delta_wind']>=30.0).astype(int)
print(f"RI rate: {df['ri_label'].sum()}/{len(df)} = {df['ri_label'].mean():.4f}")
tr=df['season'].between(1980,2014); va=df['season'].between(2015,2018); te=df['season'].between(2019,2022)
Xtr,Xv,Xt = feat_df.loc[tr,feature_cols].values, feat_df.loc[va,feature_cols].values, feat_df.loc[te,feature_cols].values
ytr,yv,yt = df.loc[tr,'ri_label'].values, df.loc[va,'ri_label'].values, df.loc[te,'ri_label'].values
print(f"Train:{len(Xtr)}(RI={ytr.sum()}) Val:{len(Xv)}(RI={yv.sum()}) Test:{len(Xt)}(RI={yt.sum()})")


# SMOTE
def smote(X, y, ratio=0.3, k=5):
    rng=np.random.RandomState(42); mi=np.where(y==1)[0]; ma=np.where(y==0)[0]
    ns=int(len(ma)*ratio/(1-ratio))-len(mi)
    if ns<=0: return X,y
    Xm=X[mi]; nn=NearestNeighbors(n_neighbors=min(k,len(mi))).fit(Xm); syn=[]
    for _ in range(ns):
        i=rng.randint(len(mi)); _,idx=nn.kneighbors(Xm[i:i+1])
        j=idx[0,rng.randint(1,len(idx[0]))] if len(idx[0])>1 else idx[0,0]
        syn.append(Xm[i]+rng.random()*(Xm[j]-Xm[i]))
    return np.vstack([X,np.array(syn)]), np.concatenate([y,np.ones(ns)])

def evaluate(name, pv, pt, yv, yt):
    bf,bt=0,0.5
    for t in np.arange(0.05,0.95,0.01):
        f=f1_score(yv,(pv>=t).astype(int),zero_division=0)
        if f>bf: bf,bt=f,t
    p=(pt>=bt).astype(int)
    a=roc_auc_score(yt,pt); f1=f1_score(yt,p,zero_division=0)
    pr=precision_score(yt,p,zero_division=0); rc=recall_score(yt,p,zero_division=0)
    br=brier_score_loss(yt,pt)
    print(f"\n{name} (thr={bt:.2f})")
    print(f"  AUC={a:.3f} Prec={pr:.3f} Rec={rc:.3f} F1={f1:.3f} Brier={br:.4f}")
    return {'model':name,'auc':a,'prec':pr,'rec':rc,'f1':f1,'brier':br}

results=[]
sw_r=(len(ytr)-ytr.sum())/max(ytr.sum(),1)
sw=np.where(ytr==1,sw_r,1.0)

# Scale for MLP
scaler=StandardScaler(); Xtr_s=scaler.fit_transform(Xtr); Xv_s=scaler.transform(Xv); Xt_s=scaler.transform(Xt)

# M1: GBM baseline (v4 best)
print("\n--- Training GBM (v4 baseline) ---")
gbm=GradientBoostingClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,min_samples_leaf=10,random_state=42)
gbm.fit(Xtr,ytr,sample_weight=sw)
results.append(evaluate("GBM baseline",gbm.predict_proba(Xv)[:,1],gbm.predict_proba(Xt)[:,1],yv,yt))

# M2: MLP
print("\n--- Training MLP ---")
mlp=MLPClassifier(hidden_layer_sizes=(128,64,32),activation='relu',max_iter=500,early_stopping=True,
                  validation_fraction=0.15,learning_rate='adaptive',random_state=42)
mlp.fit(Xtr_s,ytr)
results.append(evaluate("MLP",mlp.predict_proba(Xv_s)[:,1],mlp.predict_proba(Xt_s)[:,1],yv,yt))

# M3: GBM with SMOTE
print("\n--- Training GBM+SMOTE ---")
Xtr_sm,ytr_sm=smote(Xtr,ytr,ratio=0.2)
print(f"  SMOTE: {len(Xtr)} -> {len(Xtr_sm)} samples")
gbm2=GradientBoostingClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,min_samples_leaf=10,random_state=42)
gbm2.fit(Xtr_sm,ytr_sm)
results.append(evaluate("GBM+SMOTE",gbm2.predict_proba(Xv)[:,1],gbm2.predict_proba(Xt)[:,1],yv,yt))

# M4: MLP with SMOTE
print("\n--- Training MLP+SMOTE ---")
Xtr_sm_s=scaler.fit_transform(Xtr_sm); Xv_s2=scaler.transform(Xv); Xt_s2=scaler.transform(Xt)
mlp2=MLPClassifier(hidden_layer_sizes=(128,64,32),activation='relu',max_iter=500,early_stopping=True,
                   validation_fraction=0.15,learning_rate='adaptive',random_state=42)
mlp2.fit(Xtr_sm_s,ytr_sm)
results.append(evaluate("MLP+SMOTE",mlp2.predict_proba(Xv_s2)[:,1],mlp2.predict_proba(Xt_s2)[:,1],yv,yt))

# M5: Stacking (GBM + MLP -> LR)
print("\n--- Training Stacking Ensemble ---")
stack=StackingClassifier(
    estimators=[('gbm',GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,subsample=0.8,min_samples_leaf=10,random_state=42)),
                ('mlp',MLPClassifier(hidden_layer_sizes=(64,32),max_iter=300,early_stopping=True,random_state=42))],
    final_estimator=LogisticRegression(class_weight='balanced',random_state=42),cv=3)
stack.fit(Xtr_s,ytr)
results.append(evaluate("Stacking(GBM+MLP)",stack.predict_proba(Xv_s)[:,1],stack.predict_proba(Xt_s)[:,1],yv,yt))

# M6: Simple average ensemble
print("\n--- Simple Average Ensemble ---")
pv_avg=(gbm.predict_proba(Xv)[:,1]+mlp.predict_proba(Xv_s)[:,1])/2
pt_avg=(gbm.predict_proba(Xt)[:,1]+mlp.predict_proba(Xt_s)[:,1])/2
results.append(evaluate("Avg(GBM+MLP)",pv_avg,pt_avg,yv,yt))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Model':<25} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Brier':>7}")
for r in sorted(results,key=lambda x:-x['f1']):
    print(f"{r['model']:<25} {r['auc']:>6.3f} {r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f} {r['brier']:>7.4f}")
print("\nv4 best: GBM AUC=0.879 F1=0.372")
best=max(results,key=lambda x:x['f1'])
print(f"v5 best: {best['model']} AUC={best['auc']:.3f} F1={best['f1']:.3f}")
imp=((best['f1']-0.372)/0.372)*100
print(f"Improvement over v4: {imp:+.1f}% F1")
