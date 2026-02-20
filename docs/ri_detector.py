#!/usr/bin/env python3
"""
Rapid Intensification Detection Pipeline
Based on VORTEX framework architecture findings
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class RIDetector:
    """LSTM+Transformer hybrid for RI prediction following VORTEX design"""
    
    def __init__(self, sequence_length=24, hidden_dim=128, num_heads=8):
        self.sequence_length = sequence_length  # 24-hour lookback
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, storm_data):
        """Extract features for RI prediction"""
        features = []
        
        # Core VORTEX features: SST, wind shear, humidity, pressure, vorticity
        if 'sst' in storm_data.columns:
            features.append('sst')
        if 'pressure' in storm_data.columns:
            features.append('pressure')
        if 'wind_speed' in storm_data.columns:
            features.append('wind_speed')
        
        # Derived features
        storm_data['pressure_change'] = storm_data['pressure'].diff()
        storm_data['wind_change'] = storm_data['wind_speed'].diff()
        
        features.extend(['pressure_change', 'wind_change'])
        
        return storm_data[features].fillna(0)
    
    def label_ri_events(self, storm_data, threshold=30):
        """Label RI events (≥30 kt increase in 24h)"""
        storm_data = storm_data.copy()
        storm_data['ri_24h'] = 0
        
        for i in range(len(storm_data) - 4):  # 4 = 24h in 6h intervals
            wind_change = storm_data.iloc[i+4]['wind_speed'] - storm_data.iloc[i]['wind_speed']
            if wind_change >= threshold:
                storm_data.iloc[i, storm_data.columns.get_loc('ri_24h')] = 1
                
        return storm_data
    
    def create_sequences(self, features, labels):
        """Create temporal sequences for LSTM input"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features.iloc[i-self.sequence_length:i].values)
            y.append(labels.iloc[i])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_dim):
        """Build LSTM+Transformer hybrid following VORTEX architecture"""
        
        class VortexModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_heads, sequence_length):
                super(VortexModel, self).__init__()
                
                # LSTM encoder
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
                
                # Multi-head attention transformer
                self.transformer = nn.MultiheadAttention(
                    embed_dim=hidden_dim * 2,
                    num_heads=num_heads,
                    batch_first=True
                )
                
                # Classification head with confidence
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 2)  # [no_ri_prob, ri_prob]
                )
                
            def forward(self, x):
                # LSTM encoding
                lstm_out, _ = self.lstm(x)
                
                # Transformer attention
                attn_out, _ = self.transformer(lstm_out, lstm_out, lstm_out)
                
                # Global average pooling
                pooled = torch.mean(attn_out, dim=1)
                
                # Classification with confidence
                logits = self.classifier(pooled)
                probs = torch.softmax(logits, dim=1)
                
                return probs
        
        self.model = VortexModel(input_dim, self.hidden_dim, self.num_heads, self.sequence_length)
        return self.model

def generate_synthetic_test_data():
    """Generate synthetic storm data for testing"""
    np.random.seed(42)
    
    # Simulate 100 time steps for a storm
    n_steps = 100
    
    data = {
        'datetime': pd.date_range('2023-09-01', periods=n_steps, freq='6H'),
        'pressure': 1010 - np.random.normal(0, 10, n_steps).cumsum(),  # Pressure drops
        'wind_speed': np.maximum(35, 35 + np.random.normal(0, 5, n_steps).cumsum()),  # Wind increases
        'sst': 28 + np.random.normal(0, 1, n_steps),  # Sea surface temperature
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic RI events
    ri_times = [30, 60, 85]  # Simulate RI at these indices
    for t in ri_times:
        if t < len(df) - 4:
            df.loc[t:t+4, 'wind_speed'] += np.linspace(0, 35, 5)  # 35 kt increase over 24h
    
    return df

def main():
    """Test the RI detector with synthetic data"""
    print("=== RI Detection Pipeline Test ===")
    
    # Generate test data
    storm_data = generate_synthetic_test_data()
    print(f"Generated {len(storm_data)} time steps of synthetic storm data")
    
    # Initialize detector
    detector = RIDetector()
    
    # Prepare features
    features = detector.prepare_features(storm_data)
    print(f"Extracted {features.shape[1]} features: {list(features.columns)}")
    
    # Label RI events
    labeled_data = detector.label_ri_events(storm_data)
    ri_events = labeled_data['ri_24h'].sum()
    print(f"Detected {ri_events} RI events (≥30 kt in 24h)")
    
    # Create sequences
    if len(labeled_data) > detector.sequence_length:
        X, y = detector.create_sequences(features, labeled_data['ri_24h'])
        print(f"Created {len(X)} sequences of length {detector.sequence_length}")
        
        # Build model architecture
        model = detector.build_model(features.shape[1])
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Built VORTEX-inspired model with {total_params:,} parameters")
        
        # Test forward pass
        test_input = torch.FloatTensor(X[:1])  # Single sequence
        with torch.no_grad():
            output = model(test_input)
            ri_prob = output[0, 1].item()
            print(f"Model output - RI probability: {ri_prob:.3f}")
    
    print("\n=== Test completed successfully! ===")
    
    return {
        'n_samples': len(storm_data),
        'n_features': features.shape[1], 
        'ri_events': ri_events,
        'model_params': total_params if 'total_params' in locals() else 0
    }

if __name__ == "__main__":
    results = main()