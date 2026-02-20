# Rapid Intensification (RI) — Reading Pack

This folder is a lightweight “downloaded notes” cache: we store links + short summaries (not PDFs).

## Papers to read first (recent, high-signal)

1) Ko et al., *Weather and Forecasting* (2023) — HWRF → consensus machine learning RI probability forecasts (CML)
   - DOI: 10.1175/WAF-D-22-0217.1
   - PDF (NOAA repository): https://repository.library.noaa.gov/view/noaa/53692

2) Wang et al., *PNAS* (2025) — Contrastive learning improves RI forecasting and reduces false alarms
   - DOI: 10.1073/pnas.2415501122
   - PubMed/PMC entry is a good starting point for full text + figures.

3) Eusebi et al., *Environmental Research Letters* (2025) — sea surface salinity improves RI forecasts
   - DOI: 10.1088/1748-9326/adac7f

## Classic statistical baseline (worth skimming once)

- Kaplan et al., *Weather and Forecasting* (2010) — Revised SHIPS Rapid Intensification Index (RII)
  - DOI: 10.1175/2009WAF2222280.1
  - NOAA repository: https://repository.library.noaa.gov/view/noaa/15153

- Kaplan et al., *Weather and Forecasting* (2015) — SHIPS-RII verification + environmental predictability
  - DOI: 10.1175/WAF-D-15-0032.1
  - NOAA repository: https://repository.library.noaa.gov/view/noaa/15199

## How this repo uses the reading pack

- The paper draft references RI as an explicit metric: `docs/PAPER.md`.
- The “RI-aware baseline” lives in `src/ri_logit_baseline.py` (SHIPS-style logistic model + out-of-fold predictions).

## Notes in this folder

- `2010_kaplan_rii.md`
- `2015_kaplan_waf.md`
- `2023_ko_waf_cml.md`
- `2025_wang_pnas.md`
- `2025_eusebi_erl.md`
