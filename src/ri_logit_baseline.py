#!/usr/bin/env python3
"""
SHIPS-style RI baseline: fit a small logistic model to predict RI@24h, then map to wind deltas.

This script is intentionally simple and auditable:
- Inputs are line-aligned payloads.jsonl + truth.jsonl
- Uses K-fold CV to produce out-of-fold predictions for every line
- Track is persistence (lat/lon unchanged); focuses on intensity + RI skill
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def parse_dt_hours_from_guidance(guidance: list[str]) -> float | None:
    for g in guidance:
        m = re.search(r"every\s+(\d+)\s+min", g)
        if not m:
            continue
        minutes = int(m.group(1))
        if minutes <= 0:
            continue
        return minutes / 60.0
    return None


def extract_winds_from_guidance(guidance: list[str]) -> list[float]:
    winds: list[float] = []
    for g in guidance:
        for m in re.finditer(r"wind\s+(-?\d+(?:\.\d+)?)", g):
            try:
                winds.append(float(m.group(1)))
            except Exception:
                continue
    return winds


def wind_change(winds_hist: list[float], *, dt_hours: float, hours_back: float) -> float:
    if dt_hours <= 0:
        return 0.0
    if len(winds_hist) < 2:
        return 0.0
    steps_back = int(round(hours_back / dt_hours))
    steps_back = max(1, min(steps_back, len(winds_hist) - 1))
    return float(winds_hist[-1] - winds_hist[-1 - steps_back])


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


FEATURE_NAMES = [
    "wind0",
    "pressure0",
    "lat0",
    "lon0",
    "shear",
    "rh",
    "t600",
    "u850",
    "v850",
    "vp",
    "p_env",
    "dwind_6h",
    "dwind_24h",
]


def extract_features(payload: dict) -> np.ndarray:
    storm = payload.get("storm", {}) if isinstance(payload, dict) else {}
    env = payload.get("environment", {}) if isinstance(payload, dict) else {}
    guidance = payload.get("guidance", []) or []
    if not isinstance(guidance, list):
        guidance = []

    wind0 = _safe_float(storm.get("wind"), 0.0)
    p0 = _safe_float(storm.get("pressure"), 0.0)
    lat0 = _safe_float(storm.get("lat"), 0.0)
    lon0 = _safe_float(storm.get("lon"), 0.0)

    shear = _safe_float(env.get("shearstore"), 0.0)
    rh = _safe_float(env.get("rhstore"), 0.0)
    t600 = _safe_float(env.get("T600store"), 0.0)
    u850 = _safe_float(env.get("u850store"), 0.0)
    v850 = _safe_float(env.get("v850store"), 0.0)
    vp = _safe_float(env.get("vpstore"), 0.0)
    p_env = _safe_float(env.get("pstore"), 0.0)

    winds_hist = extract_winds_from_guidance(guidance)
    dt_hours = parse_dt_hours_from_guidance(guidance) or 6.0
    dwind_6h = wind_change(winds_hist, dt_hours=dt_hours, hours_back=6.0)
    dwind_24h = wind_change(winds_hist, dt_hours=dt_hours, hours_back=24.0)

    return np.array(
        [
            wind0,
            p0,
            lat0,
            lon0,
            shear,
            rh,
            t600,
            u850,
            v850,
            vp,
            p_env,
            dwind_6h,
            dwind_24h,
        ],
        dtype=float,
    )


def apply_feature_mask(X: np.ndarray, keep_idx: list[int]) -> np.ndarray:
    if not keep_idx:
        return X
    return X[:, keep_idx]


def standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return (x - mu) / sigma, mu, sigma


def fit_logit_irls(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    reg: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    n, d = X.shape
    beta = np.zeros(d, dtype=float)
    # L2 on non-intercept
    reg_diag = np.ones(d, dtype=float) * float(reg)
    reg_diag[0] = 0.0
    sw = np.ones(n, dtype=float) if sample_weight is None else sample_weight.astype(float)

    for _ in range(max_iter):
        eta = X @ beta
        p = sigmoid(eta)
        w0 = p * (1.0 - p)
        w0 = np.clip(w0, 1e-6, 1.0)
        z = eta + (y - p) / w0
        w = sw * w0

        XTW = X.T * w  # (d,n)
        H = (XTW @ X) + np.diag(reg_diag)
        b = XTW @ z
        beta_new = np.linalg.solve(H, b)

        if float(np.linalg.norm(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def f1_for_threshold(probs: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float, float]:
    pred = probs >= float(thr)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_platt(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(probs) == 0:
        return 0.0, 1.0
    x = logit(probs).reshape(-1, 1)
    X = np.column_stack([np.ones(len(x)), x])
    beta = fit_logit_irls(X, y.astype(int), reg=1.0)
    return float(beta[0]), float(beta[1])


def apply_platt(probs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    x = logit(probs)
    return sigmoid(alpha + beta * x)


def pick_threshold_match_rate(probs: np.ndarray, target_rate: float) -> float:
    if len(probs) == 0:
        return 0.5
    target_rate = float(max(0.0, min(1.0, target_rate)))
    if target_rate <= 0.0:
        return 1.0
    if target_rate >= 1.0:
        return 0.0
    qs = 1.0 - target_rate
    return float(np.quantile(probs, qs))


def pick_threshold_max_f1(probs: np.ndarray, y: np.ndarray) -> float:
    if len(probs) == 0:
        return 0.5
    if int(np.sum(y)) == 0:
        return 1.0
    if int(np.sum(1 - y)) == 0:
        return 0.0

    candidates = sorted(set(float(x) for x in probs.tolist()))
    # Add a reasonable default candidate in case probabilities are degenerate.
    candidates.append(0.5)

    best = (0.0, 0.0, 0.0, 0.5)  # (f1, recall, precision, thr)
    for thr in candidates:
        precision, recall, f1 = f1_for_threshold(probs, y, thr)
        key = (f1, recall, precision, thr)
        if key > best:
            best = key
    return float(best[3])


def compute_deltas(wind0: np.ndarray, truth_wind: dict[int, np.ndarray], ri_y: np.ndarray, leads: list[int], thr_kt: float) -> tuple[dict[int, float], dict[int, float]]:
    ri = ri_y.astype(bool)
    non = ~ri
    deltas_ri: dict[int, float] = {}
    deltas_non: dict[int, float] = {}

    for h in leads:
        w = truth_wind.get(h)
        if w is None:
            continue
        d = w - wind0
        d_ri = float(np.mean(d[ri])) if int(np.sum(ri)) else float(thr_kt + 1.0)
        d_non = float(np.mean(d[non])) if int(np.sum(non)) else 0.0
        if h == 24:
            d_ri = max(d_ri, float(thr_kt + 1.0))
            d_non = min(d_non, float(thr_kt - 1.0))
        deltas_ri[h] = d_ri
        deltas_non[h] = d_non
    return deltas_ri, deltas_non


@dataclass(frozen=True)
class FoldMeta:
    fold: int
    train_n: int
    val_n: int
    test_n: int
    ri_rate_train: float
    threshold: float
    beta: list[float]
    mu: list[float]
    sigma: list[float]
    cal_alpha: float | None = None
    cal_beta: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="RI-aware logistic baseline with K-fold out-of-fold predictions.")
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-meta", type=Path, default=None)
    parser.add_argument("--leads", type=str, default="24,48,72")
    parser.add_argument("--ri-lead-hours", type=int, default=24)
    parser.add_argument("--ri-threshold-kt", type=float, default=30.0)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--drop-features", type=str, default="", help="Comma-separated feature names to drop.")
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="auto",
        choices=["auto", "rate", "f1", "fixed"],
        help="How to choose RI probability threshold.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (for threshold-mode=fixed).")
    parser.add_argument("--calibrate", action="store_true", help="Apply Platt scaling on validation probs.")
    args = parser.parse_args()

    leads = [int(x.strip()) for x in args.leads.split(",") if x.strip()]
    lead_ri = int(args.ri_lead_hours)
    thr_kt = float(args.ri_threshold_kt)
    k = int(args.kfold)
    if k < 2:
        raise SystemExit("--kfold must be >= 2")

    payload_lines = args.payloads.read_text(encoding="utf-8").strip().splitlines()
    truth_lines = args.truth.read_text(encoding="utf-8").strip().splitlines()
    n = min(len(payload_lines), len(truth_lines))
    if n == 0:
        raise SystemExit("empty input")

    payloads = [json.loads(payload_lines[i]) for i in range(n)]
    truth = [json.loads(truth_lines[i]) for i in range(n)]

    X_full = np.stack([extract_features(p) for p in payloads], axis=0)
    wind0 = np.array([_safe_float((p.get("storm", {}) or {}).get("wind"), 0.0) for p in payloads], dtype=float)

    truth_wind: dict[int, np.ndarray] = {}
    for h in leads:
        key = f"wind_{h}h"
        truth_wind[h] = np.array([_safe_float(t.get(key), float("nan")) for t in truth], dtype=float)

    if lead_ri not in truth_wind:
        raise SystemExit(f"truth missing required lead: {lead_ri}h")

    ri_y = (truth_wind[lead_ri] - wind0) >= thr_kt
    y = ri_y.astype(int)

    drop = [x.strip() for x in args.drop_features.split(",") if x.strip()]
    drop_set = {d for d in drop}
    keep_idx = [i for i, name in enumerate(FEATURE_NAMES) if name not in drop_set]
    feature_names = [FEATURE_NAMES[i] for i in keep_idx]
    if drop_set:
        missing = drop_set - set(FEATURE_NAMES)
        if missing:
            raise SystemExit(f"unknown feature(s) in --drop-features: {sorted(missing)}")

    X = apply_feature_mask(X_full, keep_idx)

    if args.threshold_mode == "fixed" and args.threshold is None:
        raise SystemExit("--threshold is required when --threshold-mode=fixed")

    rng = np.random.default_rng(int(args.seed))
    indices = np.arange(n, dtype=int)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    out_lines: list[str] = [""] * n
    fold_meta: list[FoldMeta] = []

    for fold_id, test_idx in enumerate(folds):
        test_idx = np.array(test_idx, dtype=int)
        test_set = set(test_idx.tolist())
        train_idx = np.array([i for i in indices if i not in test_set], dtype=int)

        # inner split for threshold selection
        inner = train_idx.copy()
        inner_rng = np.random.default_rng(int(args.seed) + 1000 + fold_id)
        inner_rng.shuffle(inner)
        val_n = max(1, int(round(0.2 * len(inner))))
        val_idx = inner[:val_n]
        fit_idx = inner[val_n:]
        if len(fit_idx) < 5:
            fit_idx = train_idx
            val_idx = train_idx[: max(1, min(10, len(train_idx)))]

        X_fit_raw = X[fit_idx]
        X_val_raw = X[val_idx]
        X_test_raw = X[test_idx]
        X_train_raw = X[train_idx]

        X_fit, mu, sigma = standardize(X_fit_raw, X_fit_raw)
        X_val, _, _ = standardize(X_fit_raw, X_val_raw)
        X_train, _, _ = standardize(X_fit_raw, X_train_raw)
        X_test, _, _ = standardize(X_fit_raw, X_test_raw)

        X_fit_i = np.column_stack([np.ones(len(X_fit)), X_fit])
        X_val_i = np.column_stack([np.ones(len(X_val)), X_val])
        X_train_i = np.column_stack([np.ones(len(X_train)), X_train])
        X_test_i = np.column_stack([np.ones(len(X_test)), X_test])

        # Simple class weighting for imbalance: weight positives ~ n_neg/n_pos (fit set).
        y_fit = y[fit_idx]
        n_pos = int(np.sum(y_fit))
        n_neg = int(len(y_fit) - n_pos)
        pos_w = (n_neg / n_pos) if n_pos else 1.0
        sample_w_fit = np.where(y_fit == 1, pos_w, 1.0).astype(float)

        beta_inner = fit_logit_irls(X_fit_i, y_fit, sample_weight=sample_w_fit, reg=float(args.reg))
        val_probs_raw = sigmoid(X_val_i @ beta_inner)
        cal_alpha = None
        cal_beta = None
        val_probs = val_probs_raw
        if args.calibrate:
            cal_alpha, cal_beta = fit_platt(val_probs_raw, y[val_idx])
            val_probs = apply_platt(val_probs_raw, cal_alpha, cal_beta)

        thr_f1 = pick_threshold_max_f1(val_probs, y[val_idx])
        thr_rate = pick_threshold_match_rate(val_probs, float(np.mean(y_fit)))
        if args.threshold_mode == "f1":
            thr = thr_f1
        elif args.threshold_mode == "rate":
            thr = thr_rate
        elif args.threshold_mode == "fixed":
            thr = float(args.threshold)
        else:
            p1, r1, f1_1 = f1_for_threshold(val_probs, y[val_idx], thr_f1)
            p2, r2, f1_2 = f1_for_threshold(val_probs, y[val_idx], thr_rate)
            thr = thr_f1 if (f1_1, r1, p1) >= (f1_2, r2, p2) else thr_rate

        y_train = y[train_idx]
        n_pos_t = int(np.sum(y_train))
        n_neg_t = int(len(y_train) - n_pos_t)
        pos_w_t = (n_neg_t / n_pos_t) if n_pos_t else 1.0
        sample_w_train = np.where(y_train == 1, pos_w_t, 1.0).astype(float)
        beta = fit_logit_irls(X_train_i, y_train, sample_weight=sample_w_train, reg=float(args.reg))
        test_probs_raw = sigmoid(X_test_i @ beta)
        test_probs = test_probs_raw
        if args.calibrate and cal_alpha is not None and cal_beta is not None:
            test_probs = apply_platt(test_probs_raw, cal_alpha, cal_beta)
        test_pred = (test_probs >= thr).astype(int)

        truth_wind_train = {h: w[train_idx] for h, w in truth_wind.items()}
        d_ri, d_non = compute_deltas(wind0[train_idx], truth_wind_train, ri_y[train_idx], leads, thr_kt)

        for j, idx in enumerate(test_idx.tolist()):
            storm = payloads[idx].get("storm", {}) if isinstance(payloads[idx], dict) else {}
            lat0 = _safe_float(storm.get("lat"), 0.0)
            lon0 = _safe_float(storm.get("lon"), 0.0)
            w0 = _safe_float(storm.get("wind"), 0.0)

            is_ri = bool(test_pred[j])
            fc = []
            for h in leads:
                delta = d_ri.get(h, 0.0) if is_ri else d_non.get(h, 0.0)
                fc.append({"lead_hours": int(h), "lat": lat0, "lon": lon0, "wind": float(max(0.0, w0 + float(delta)))})
            payload = {
                "forecast": fc,
                "reasoning": "ri_logit",
                "ri_prob_24h": float(test_probs_raw[j]),
                "ri_call_24h": int(is_ri),
                "ri_threshold": float(thr),
                "ri_threshold_mode": str(args.threshold_mode),
            }
            if args.calibrate:
                payload["ri_prob_24h_cal"] = float(test_probs[j])
                payload["ri_calibration"] = {"alpha": float(cal_alpha), "beta": float(cal_beta)} if cal_alpha is not None else {}
            out_lines[idx] = json.dumps(payload)

        fold_meta.append(
            FoldMeta(
                fold=int(fold_id),
                train_n=int(len(train_idx)),
                val_n=int(len(val_idx)),
                test_n=int(len(test_idx)),
                ri_rate_train=float(np.mean(ri_y[train_idx])),
                threshold=float(thr),
                beta=[float(x) for x in beta.tolist()],
                mu=[float(x) for x in mu.tolist()],
                sigma=[float(x) for x in sigma.tolist()],
                cal_alpha=float(cal_alpha) if cal_alpha is not None else None,
                cal_beta=float(cal_beta) if cal_beta is not None else None,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote predictions to {args.out}")

    if args.out_meta:
        meta = {
            "meta": {
                "kfold": k,
                "seed": int(args.seed),
                "ri_lead_hours": lead_ri,
                "ri_threshold_kt": thr_kt,
                "feature_names": feature_names,
                "dropped_features": drop,
                "reg": float(args.reg),
                "threshold_mode": str(args.threshold_mode),
                "threshold_fixed": float(args.threshold) if args.threshold is not None else None,
                "calibrate": bool(args.calibrate),
            },
            "folds": [asdict(m) for m in fold_meta],
        }
        args.out_meta.parent.mkdir(parents=True, exist_ok=True)
        args.out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
