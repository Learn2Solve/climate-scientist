from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SubsetMeta:
    n_total: int
    n_selected: int
    seed: int
    indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {"n_total": self.n_total, "n_selected": self.n_selected, "seed": self.seed, "indices": self.indices}


def subset_payloads(
    payloads_jsonl: Path,
    truth_jsonl: Path,
    out_payloads_jsonl: Path,
    out_truth_jsonl: Path,
    *,
    limit: int,
    seed: int = 42,
) -> SubsetMeta:
    payload_lines = payloads_jsonl.read_text(encoding="utf-8").splitlines()
    truth_lines = truth_jsonl.read_text(encoding="utf-8").splitlines()
    n = min(len(payload_lines), len(truth_lines))
    if n == 0:
        raise SystemExit("Empty payloads/truth JSONL")
    if limit <= 0 or limit > n:
        limit = n

    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:limit])

    out_payloads_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_truth_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_payloads_jsonl.write_text("\n".join(payload_lines[i] for i in chosen) + "\n", encoding="utf-8")
    out_truth_jsonl.write_text("\n".join(truth_lines[i] for i in chosen) + "\n", encoding="utf-8")

    return SubsetMeta(n_total=n, n_selected=len(chosen), seed=seed, indices=chosen)


def inject_baseline_guidance(
    payloads_jsonl: Path,
    baseline_predictions_jsonl: Path,
    out_guided_payloads_jsonl: Path,
    *,
    label: str,
) -> dict[str, Any]:
    guided = 0
    out_guided_payloads_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with payloads_jsonl.open("r", encoding="utf-8") as f_p, baseline_predictions_jsonl.open("r", encoding="utf-8") as f_b, out_guided_payloads_jsonl.open("w", encoding="utf-8") as f_out:
        for p_line, b_line in zip(f_p, f_b):
            payload = json.loads(p_line)
            base = json.loads(b_line)
            fc = base.get("forecast", [])

            parts = []
            for item in fc:
                if not isinstance(item, dict):
                    continue
                h = item.get("lead_hours")
                lat = item.get("lat")
                lon = item.get("lon")
                wind = item.get("wind")
                if h is None:
                    continue
                parts.append(f"{int(h)}h: lat {float(lat):.2f}, lon {float(lon):.2f}, wind {float(wind):.1f}")

            guidance_line = f"Baseline guidance ({label}): " + "; ".join(parts)
            g = payload.get("guidance")
            if not isinstance(g, list):
                g = []
            g.append(guidance_line)
            payload["guidance"] = g
            f_out.write(json.dumps(payload) + "\n")
            guided += 1

    return {"guided_samples": guided, "label": label, "out": str(out_guided_payloads_jsonl)}

