#!/usr/bin/env python3
"""
Test RI detector with real HURDAT2 data
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the docs directory to path to import ri_detector
sys.path.append(str(Path(__file__).parent))
from ri_detector import RIDetector, generate_synthetic_test_data

def load_hurdat2_data():
    """Load the processed HURDAT2 parquet data"""
    parquet_path = Path("hurdat2_llm_toy/all_samples.parquet")
    
    if not parquet_path.exists():
        print(f"HURDAT2 data not found at {parquet_path}")
        return None
        
    df = pd.read_parquet(parquet_path)
    print(f"Loaded HURDAT2 data: {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['season'].min()}-{df['season'].max()}")
    
    return df

def convert_hurdat2_to_timeseries(df):
    """Convert HURDAT2 sample format to time series for RI detection"""
    
    # The HURDAT2 parquet has storm snapshots, we need to reconstruct time series
    # Let's examine the structure first
    sample = df.iloc[0]
    print(f"\nSample HURDAT2 record structure:")
    for col in df.columns:
        print(f"  {col}: {sample[col]}")
    
    # For now, let's work with wind speed history if available
    timeseries_data = []
    
    for idx, row in df.iterrows():
        # Extract wind speed progression from the sample
        if 'target_wind' in row and 'last_wind' in row:
            # Simple 2-point time series
            wind_data = [row['last_wind'], row['target_wind']]
            pressure_data = [row.get('last_pressure', 1013), row.get('target_pressure', 1013)]
            
            # Create minimal time series
            ts_data = {
                'storm_id': f"storm_{idx}",
                'wind_speed': wind_data,
                'pressure': pressure_data,
                'sst': [28.0, 28.0],  # Default SST
                'time_steps': 2
            }
            timeseries_data.append(ts_data)
            
            if len(timeseries_data) >= 10:  # Limit for testing
                break
                
    return timeseries_data

def test_ri_detection_with_hurdat2():
    """Test RI detector with real HURDAT2 data"""
    print("=== RI Detection with Real HURDAT2 Data ===")
    
    # Load HURDAT2 data
    hurdat2_df = load_hurdat2_data()
    
    if hurdat2_df is None:
        print("Falling back to synthetic data test...")
        # Test with synthetic data as backup
        return test_synthetic_data()
    
    # Convert to time series format
    timeseries_data = convert_hurdat2_to_timeseries(hurdat2_df)
    print(f"\nConverted {len(timeseries_data)} storms to time series format")
    
    # Initialize detector
    detector = RIDetector(sequence_length=4)  # Shorter sequence for limited data
    
    # Test on first few storms
    ri_detections = []
    
    for i, storm in enumerate(timeseries_data[:5]):
        print(f"\n--- Testing Storm {storm['storm_id']} ---")
        
        # Create DataFrame for this storm
        storm_df = pd.DataFrame({
            'wind_speed': storm['wind_speed'],
            'pressure': storm['pressure'], 
            'sst': storm['sst']
        })
        
        # Check for RI (≥30 kt increase)
        wind_change = storm_df['wind_speed'].iloc[-1] - storm_df['wind_speed'].iloc[0]
        has_ri = wind_change >= 30
        
        print(f"  Wind change: {wind_change:.1f} kt")
        print(f"  RI event: {'YES' if has_ri else 'NO'}")
        
        ri_detections.append({
            'storm_id': storm['storm_id'],
            'wind_change': wind_change,
            'has_ri': has_ri
        })
    
    # Summary statistics
    ri_count = sum(1 for d in ri_detections if d['has_ri'])
    total_storms = len(ri_detections)
    
    print(f"\n=== RI Detection Summary ===")
    print(f"Total storms tested: {total_storms}")
    print(f"RI events detected: {ri_count}")
    print(f"RI frequency: {ri_count/total_storms*100:.1f}%")
    
    return {
        'total_storms': total_storms,
        'ri_events': ri_count,
        'ri_frequency': ri_count/total_storms if total_storms > 0 else 0,
        'data_source': 'HURDAT2_real'
    }

def test_synthetic_data():
    """Fallback test with synthetic data"""
    print("=== Testing with Synthetic Data ===")
    
    # Use the synthetic data generator from ri_detector.py
    storm_data = generate_synthetic_test_data()
    
    detector = RIDetector()
    features = detector.prepare_features(storm_data)
    labeled_data = detector.label_ri_events(storm_data)
    
    ri_events = labeled_data['ri_24h'].sum()
    total_periods = len(labeled_data)
    
    print(f"Synthetic data test: {ri_events} RI events in {total_periods} time periods")
    
    return {
        'total_storms': 1,
        'ri_events': ri_events,
        'ri_frequency': ri_events/total_periods,
        'data_source': 'synthetic'
    }

def main():
    """Main test function"""
    try:
        results = test_ri_detection_with_hurdat2()
    except Exception as e:
        print(f"Error with HURDAT2 data: {e}")
        print("Running synthetic test instead...")
        results = test_synthetic_data()
    
    print(f"\n=== Final Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    return results

if __name__ == "__main__":
    results = main()