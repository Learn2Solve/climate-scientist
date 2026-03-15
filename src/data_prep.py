import numpy as np
import pandas as pd
from pathlib import Path

try:
    import tropycal.tracks as tracks
except ModuleNotFoundError as exc:
    # Provide a helpful message rather than a raw traceback when the dependency is missing.
    raise SystemExit(
        "Missing dependency 'tropycal'. Install with `pip install -r requirements.txt` "
        "or `pip install tropycal`."
    ) from exc

# -----------------------------
# 1. 读 HURDAT2（Atlantic）
# -----------------------------
print("Loading HURDAT2 via tropycal ...")
basin = tracks.TrackDataset(
    basin="north_atlantic",
    source="hurdat",
    include_btk=False,
)
print(basin)  # 会打印 year range / number of storms 等信息

# -----------------------------
# 2. 构造 48h -> 24h 样本
# -----------------------------
HISTORY_H = 48
LEAD_H = 24
DT_H = 6

MIN_POINTS = HISTORY_H // DT_H + LEAD_H // DT_H + 1

SEASON_MIN = 1980
SEASON_MAX = 2022

def build_samples():
    samples = []

    # Older/newer tropycal versions expose `keys` as a list attribute instead of a method.
    try:
        storm_ids = basin.keys()
    except TypeError:
        storm_ids = basin.keys

    for storm_id in storm_ids:
        storm = basin.get_storm(storm_id)

        # Storm 的年份，tropycal 里是属性 year 或 attrs["year"]
        year = int(getattr(storm, "year", storm.attrs.get("year")))
        if year < SEASON_MIN or year > SEASON_MAX:
            continue

        df = storm.to_dataframe()  # time, lat, lon, vmax, mslp, type, ...

        if len(df) < MIN_POINTS:
            continue

        # sliding window: 过去 48h，预测 24h
        for last_hist_idx in range(HISTORY_H // DT_H - 1,
                                   len(df) - LEAD_H // DT_H):
            first_hist_idx = last_hist_idx - HISTORY_H // DT_H + 1
            target_idx = last_hist_idx + LEAD_H // DT_H

            hist = df.iloc[first_hist_idx:last_hist_idx + 1]
            target = df.iloc[target_idx]
            last_hist = hist.iloc[-1]

            # 构造 past observations 文本
            past_lines = []
            for _, row in hist.iterrows():
                t_str = row["time"].strftime("%Y-%m-%d %H:%M UTC")
                lon_abs = abs(row["lon"])
                hemi_lon = "W" if row["lon"] < 0 else "E"
                hemi_lat = "N" if row["lat"] >= 0 else "S"
                past_lines.append(
                    f"{t_str}: lat {abs(row['lat']):.1f}{hemi_lat}, "
                    f"lon {lon_abs:.1f}{hemi_lon}, "
                    f"max wind {int(row['vmax'])} kt, "
                    f"status {row['type']}"
                )

            past_block = "\n".join(past_lines)
            ref_time = last_hist["time"]

            storm_name = getattr(storm, "name", storm.attrs.get("name", "UNKNOWN"))
            storm_year = year

            prompt = f"""You are an expert tropical cyclone forecaster.

Here is the past track and intensity of a storm from the North Atlantic HURDAT2 database.

Storm id: {storm_id}
Storm name: {storm_name}
Year: {storm_year}

Past observations (every 6 hours):
{past_block}

Task: Based on the above information and your knowledge of hurricane climatology,
predict the storm's center location (lat, lon) and maximum sustained wind 24 hours after {ref_time.strftime("%Y-%m-%d %H:%M UTC")}.

Respond in the following JSON format:
{{"lat_24h": <latitude in degrees>, "lon_24h": <longitude in degrees>, "wind_24h": <max wind in kt>}}
"""

            samples.append(
                dict(
                    input=prompt,
                    target_lat=float(target["lat"]),
                    target_lon=float(target["lon"]),
                    target_wind=float(target["vmax"]),
                    last_lat=float(last_hist["lat"]),
                    last_lon=float(last_hist["lon"]),
                    last_wind=float(last_hist["vmax"]),
                    season=year,
                    storm_id=storm_id,
                    ref_time=ref_time.isoformat(),
                )
            )

    return pd.DataFrame(samples)


print("Building 48h->24h samples ...")
df = build_samples()
print(f"Total samples: {len(df)}")
print(df[["season", "storm_id"]].head())

print("\nSamples per season (1980–2022):")
print(df["season"].value_counts().sort_index())

# -----------------------------
# 3. 划分 train/val/test
# -----------------------------
train_years = list(range(1980, 2015))
val_years = list(range(2015, 2019))
test_years = list(range(2019, 2023))

df_train = df[df["season"].isin(train_years)].copy()
df_val = df[df["season"].isin(val_years)].copy()
df_test = df[df["season"].isin(test_years)].copy()

print(f"\nTrain samples: {len(df_train)}")
print(f"Val samples:   {len(df_val)}")
print(f"Test samples:  {len(df_test)}")

# -----------------------------
# 4. 24h persistence baseline
# -----------------------------
def great_circle_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def eval_persistence(df_split, name="test"):
    d = df_split
    dist_km = great_circle_distance_km(
        d["last_lat"], d["last_lon"],
        d["target_lat"], d["target_lon"]
    )
    wind_err = (d["last_wind"] - d["target_wind"]).abs()
    print(f"\nPersistence baseline on {name}:")
    print(f"  Track MAE (km):  {dist_km.mean():.1f}")
    print(f"  Wind  MAE (kt):  {wind_err.mean():.1f}")

eval_persistence(df_test, "test")
eval_persistence(df_val, "val")

# -----------------------------
# 5. 导出 SFT 数据
# -----------------------------
outdir = Path("hurdat2_llm_toy")
outdir.mkdir(exist_ok=True)

def df_to_sft_jsonl(df_split, path):
    def build_output(row):
        return (
            '{'
            f"\"lat_24h\": {row['target_lat']:.2f}, "
            f"\"lon_24h\": {row['target_lon']:.2f}, "
            f"\"wind_24h\": {row['target_wind']:.0f}"
            '}'
        )

    sft_df = pd.DataFrame({
        "input": df_split["input"],
        "output": df_split.apply(build_output, axis=1),
    })
    sft_df.to_json(path, orient="records", lines=True, force_ascii=False)

df.to_parquet(outdir / "all_samples.parquet")
df_to_sft_jsonl(df_train, outdir / "train_sft.jsonl")
df_to_sft_jsonl(df_val, outdir / "val_sft.jsonl")
df_to_sft_jsonl(df_test, outdir / "test_sft.jsonl")

print(f"\nSaved to directory: {outdir.resolve()}")
print("  - all_samples.parquet")
print("  - train_sft.jsonl / val_sft.jsonl / test_sft.jsonl")
print("\nNow you can plug train_sft.jsonl into DeepSeek-V3.2-Speciale for SFT.")
