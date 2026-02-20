#!/usr/bin/env python3
"""
Simple examination of HURDAT2 data structure without complex imports
"""

import pandas as pd
import json

def main():
    try:
        # Load the HURDAT2 parquet file
        df = pd.read_parquet("hurdat2_llm_toy/all_samples.parquet")
        
        print("=== HURDAT2 Data Structure Analysis ===")
        print(f"Total samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Examine first few rows
        print("\nFirst 3 samples:")
        for i in range(min(3, len(df))):
            print(f"\nSample {i}:")
            sample = df.iloc[i]
            for col in ['season', 'last_wind', 'target_wind', 'last_pressure', 'target_pressure']:
                if col in sample:
                    print(f"  {col}: {sample[col]}")
        
        # Look for wind speed changes that indicate RI
        if 'last_wind' in df.columns and 'target_wind' in df.columns:
            df['wind_change'] = df['target_wind'] - df['last_wind']
            
            # Count RI events (≥30 kt increase)
            ri_events = (df['wind_change'] >= 30).sum()
            total_samples = len(df)
            
            print(f"\n=== RI Analysis ===")
            print(f"Samples with ≥30 kt wind increase: {ri_events}")
            print(f"Total samples: {total_samples}")
            print(f"RI frequency: {ri_events/total_samples*100:.2f}%")
            
            # Wind change distribution
            print(f"\nWind change statistics:")
            print(f"Mean: {df['wind_change'].mean():.2f} kt")
            print(f"Std: {df['wind_change'].std():.2f} kt")
            print(f"Max: {df['wind_change'].max():.2f} kt")
            print(f"Min: {df['wind_change'].min():.2f} kt")
            
            # Show some RI cases
            ri_cases = df[df['wind_change'] >= 30].head(5)
            print(f"\nFirst 5 RI cases (≥30 kt increase):")
            for i, (idx, row) in enumerate(ri_cases.iterrows()):
                print(f"  Case {i+1}: {row['last_wind']:.1f} -> {row['target_wind']:.1f} kt (+{row['wind_change']:.1f} kt)")
        
        # Save basic statistics
        stats = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'ri_events': ri_events if 'wind_change' in locals() else 0,
            'ri_frequency': ri_events/total_samples if 'wind_change' in locals() else 0
        }
        
        with open('docs/hurdat2_analysis.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"\nAnalysis saved to docs/hurdat2_analysis.json")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()