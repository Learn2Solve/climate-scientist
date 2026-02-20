# Rapid Intensification Research Plan

## Objective
Improve rapid intensification prediction accuracy using available HURDAT2 data and modern deep learning approaches, building on recent 2025-2026 breakthroughs.

## Current State (2026-02-19)
- Recent literature shows 92% RI prediction possible with LSTM+Transformer hybrid (VORTEX framework)
- High-resolution modeling (<2km) critical for eyewall processes
- Key features: SST, wind shear, humidity, pressure, vorticity
- Available data: HURDAT2 samples in parquet format

## Immediate Actions
1. **Baseline Assessment**: Run existing RI logistic baseline on HURDAT2 data
2. **Literature Integration**: Download and analyze recent papers (VORTEX, NPJ Climate transformer)  
3. **Feature Engineering**: Implement intensity-change features (6h/24h wind deltas)
4. **Deep Learning Prototype**: Build LSTM+Transformer hybrid following VORTEX architecture

## Success Metrics
- RI classification F1 score improvement
- Maintained or improved wind intensity MAE
- Confidence quantification for operational use

## Timeline
- Week 1: Baseline + literature analysis
- Week 2: Feature engineering + prototype
- Week 3: Model validation + documentation