# climate-scientist

Data prep script for creating small supervised fine-tuning (SFT) datasets from the HURDAT2 North Atlantic archive using `tropycal`. It pulls storm tracks, builds 48h-history → 24h-forecast samples, exports parquet/JSONL splits, and prints a simple 24h persistence baseline.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare_hurdat_v2.py
```

`tropycal` will automatically download the HURDAT2 data on first run and cache it under `~/.tropycal`.

## Outputs

- `hurdat2_llm_toy/all_samples.parquet` – full sample table
- `hurdat2_llm_toy/train_sft.jsonl`, `val_sft.jsonl`, `test_sft.jsonl` – prompts and targets for SFT

## Notes

- Seasons limited to 1980–2022; adjust `SEASON_MIN/SEASON_MAX` in `prepare_hurdat_v2.py` if needed.
- Persistence baseline uses great-circle distance and last-known intensity for a quick reference metric.
