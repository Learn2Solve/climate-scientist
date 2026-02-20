# Related Work (quick list)

This is a lightweight, paper-writing oriented list of recent work related to:
(1) weather/climate foundation models, and (2) tropical cyclone (TC) forecasting benchmarks, and (3) LLMs for meteorology.

## Rapid intensification (RI) forecasting

- **SHIPS Rapid Intensification Index (SHIPS-RII / RII)** (Kaplan et al., *Weather and Forecasting* 2010; DOI: 10.1175/2009WAF2222280.1)
  - Classic operational-style baseline: linear-discriminant RI probabilities using SHIPS predictors.
- **SHIPS-RII verification + environmental predictability** (Kaplan et al., *Weather and Forecasting* 2015; DOI: 10.1175/WAF-D-15-0032.1)
  - Verification methodology + predictability analysis; serves as a canonical reference for SHIPS-RII skill.
- **HWRF → consensus machine learning RI forecasts (CML)** (Ko et al., *Weather and Forecasting* 2023; DOI: 10.1175/WAF-D-22-0217.1)
  - Uses predictors from high-resolution HWRF output; compares ML consensus vs SHIPS and operational guidance.
- **Sea surface salinity as an RI predictor** (Eusebi et al., *Environmental Research Letters* 2025; DOI: 10.1088/1748-9326/adac7f)
  - Shows added predictive skill when including SSS, suggesting missing surface-ocean information is important for RI.
- **Contrastive learning for RI forecasting** (Wang et al., *PNAS* 2025; DOI: 10.1073/pnas.2415501122)
  - Contrastive/self-supervised objectives can improve RI skill and reduce false alarms (satellite + atmosphere/ocean predictors).

## Weather / Earth system foundation models

- **Aurora** (Bodnar et al., *Nature* 2025; arXiv:2405.13063)
  - Earth system FM; strong general forecasting skill across variables/domains.
- **AIFS (ECMWF)** (Klampt et al., arXiv:2406.01465)
  - Data-driven global weather model; ECMWF has deployed an operational AIFS cycle (Feb 2025).
- **Prithvi-WxC** (NASA/IMPACT; arXiv:2409.13598)
  - Foundation model for weather and climate fields.
- **WeatherGFM** (XiChen et al., arXiv:2411.05420)
  - Benchmark/framework for evaluating global forecasting models.

## Tropical cyclone forecasting / benchmarks

- **TropiCycloneNet** (*Nature Communications* 2025)
  - Global TC benchmark for track + intensity.
- **TIFNet** (*npj Climate and Atmospheric Science* 2026)
  - Spatiotemporal transformer for TC intensity forecasting from satellite imagery.

## LLMs for meteorology

- **AI-Meteorologist** (arXiv:2511.23387)
  - LLM-generated meteorology reports with structured evidence (“proof”) for grounding.

## How this connects to our paper

- FMs (Aurora/AIFS/Prithvi) motivate a strong “climate FM baseline”.
- TC-specific work motivates a clear target task (track + intensity), and highlights the need for careful benchmark design.
- LLM meteorology/reporting work motivates LLMs as *analysis/decision layers* over structured state, rather than purely raw sequence prediction.
