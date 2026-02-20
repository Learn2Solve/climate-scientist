#!/usr/bin/env python3
"""
Inspect HURDAT2 data structure to understand available features for RI prediction
"""

import pandas as pd
import json
from pathlib import Path

# Load the parquet data
data_path = Path("../hurdat2_llm_toy/all_samples.parquet")
df = pd.read_parquet(data_path)

print("HURDAT2 Data Structure Analysis")
print("="*50)
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Data types: {df.dtypes}")
print()

# Show first few rows
print("Sample data structure:")
for i in range(min(3, len(df))):
    print(f"\nSample {i+1}:")
    for col in df.columns:
        val = df.iloc[i][col]
        if isinstance(val, str) and len(val) > 200:
            print(f"  {col}: {val[:200]}...")
        else:
            print(f"  {col}: {val}")

# Look for patterns in the data that indicate RI events
print("\n" + "="*50)
print("Rapid Intensification Pattern Analysis")

# Check if we have wind intensity information
sample_texts = df.iloc[:100]['prompt'] if 'prompt' in df.columns else df.iloc[:100].iloc[:,0]

ri_keywords = ['rapid intensification', 'RI', 'intensify', 'strengthen', 'wind speed', 'maximum wind']
ri_samples = []

for i, text in enumerate(sample_texts):
    if any(keyword.lower() in str(text).lower() for keyword in ri_keywords):
        ri_samples.append(i)

print(f"Found {len(ri_samples)} samples with RI-related keywords")
if ri_samples:
    print("Example RI-related sample:")
    print(sample_texts.iloc[ri_samples[0]][:500] + "...")