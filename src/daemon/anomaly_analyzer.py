"""Anomaly Analyzer — finds systematic failures and surprises in experiment results.

This is the "discovery engine" of the swarm.  It looks at per-sample errors
from experiments and identifies patterns that suggest missing features,
model blind spots, or interesting physical regimes.

Anomaly types detected:
    1. HIGH_ERROR_CLUSTER   — group of samples with unusually large errors
    2. RI_BLIND_SPOT        — rapid intensification events systematically missed
    3. REGIME_SHIFT         — errors correlate with a physical regime change
    4. BIAS_PATTERN         — systematic over/under-prediction
    5. OUTLIER              — individual extreme outlier
    6. ERROR_TREND          — error grows with lead time faster than expected
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from typing import Any

log = logging.getLogger("daemon.anomaly_analyzer")


class AnomalyAnalyzer:
    """Detect anomalies in experiment results.

    Works in two modes:
    1. Statistical mode (no LLM): uses thresholds on per-sample metrics
    2. LLM-augmented mode: asks inference to explain anomaly patterns

    Phase 1 focuses on statistical mode for speed and cost.
    """

    def __init__(
        self,
        config: Any,
        db: Any,
        inference: Any | None = None,
        *,
        z_threshold: float = 2.0,
        ri_threshold_kt: float = 30.0,
        bias_threshold_pct: float = 20.0,
    ) -> None:
        self.config = config
        self.db = db
        self.inference = inference
        self.z_threshold = z_threshold
        self.ri_threshold_kt = ri_threshold_kt
        self.bias_threshold_pct = bias_threshold_pct

    def analyze(self, experiment_results: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze experiment results and return a list of anomalies.

        Each anomaly is a dict with:
            type: str — anomaly category
            description: str — human-readable description
            severity: float — 0-1 how important this is
            evidence: dict — supporting data
            suggested_investigation: str — what to do about it
        """
        per_sample = experiment_results.get("per_sample", [])
        metrics = experiment_results.get("metrics", {})

        anomalies: list[dict[str, Any]] = []

        if per_sample:
            anomalies.extend(self._check_outliers(per_sample))
            anomalies.extend(self._check_ri_blindspot(per_sample))
            anomalies.extend(self._check_bias_pattern(per_sample))
            anomalies.extend(self._check_regime_clusters(per_sample))
            anomalies.extend(self._check_error_trend(per_sample))

        if metrics:
            anomalies.extend(self._check_metric_anomalies(metrics))

        # If we have an LLM, ask it to interpret the anomalies
        if self.inference and anomalies:
            anomalies = self._llm_interpret(anomalies, experiment_results)

        # Sort by severity
        anomalies.sort(key=lambda a: a.get("severity", 0), reverse=True)

        log.info("[ANOMALY] Found %d anomalies", len(anomalies))
        for a in anomalies[:5]:
            log.info(
                "  [%.2f] %s: %s",
                a.get("severity", 0),
                a.get("type", "?"),
                a.get("description", "")[:80],
            )

        return anomalies

    # ------------------------------------------------------------------
    # Statistical checks
    # ------------------------------------------------------------------

    @staticmethod
    def _get_field(s: dict[str, Any], *candidates: str) -> float | None:
        """Try multiple field names, return the first non-None value."""
        for c in candidates:
            v = s.get(c)
            if v is not None:
                return float(v)
        return None

    def _check_outliers(self, per_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find samples with errors far above the mean (z-score)."""
        anomalies = []

        # Extract track errors — try multiple naming conventions
        track_errors = []
        for s in per_sample:
            v = self._get_field(s, "track_error_km", "track_mae_km",
                                "track_error_24h", "track_error_48h")
            if v is not None:
                track_errors.append(v)

        wind_errors = []
        for s in per_sample:
            v = self._get_field(s, "wind_error_kt", "wind_mae_kt",
                                "wind_error_24h", "wind_error_48h")
            if v is not None:
                wind_errors.append(abs(v))

        for label, errors in [("track", track_errors), ("wind", wind_errors)]:
            if len(errors) < 5:
                continue
            mean = statistics.mean(errors)
            stdev = statistics.stdev(errors) if len(errors) > 1 else 1.0
            if stdev < 1e-9:
                continue

            outliers = []
            for i, e in enumerate(errors):
                z = (e - mean) / stdev
                if z > self.z_threshold:
                    outliers.append({"index": i, "error": e, "z_score": round(z, 2)})

            if outliers:
                anomalies.append({
                    "type": "OUTLIER",
                    "description": (
                        f"{len(outliers)} samples have {label} errors > {self.z_threshold}σ "
                        f"above the mean ({mean:.1f}). "
                        f"Worst: index {outliers[0]['index']} with error {outliers[0]['error']:.1f} "
                        f"(z={outliers[0]['z_score']:.1f})"
                    ),
                    "severity": min(1.0, len(outliers) / len(errors) * 5),
                    "evidence": {
                        "metric": label,
                        "mean": round(mean, 2),
                        "stdev": round(stdev, 2),
                        "outlier_count": len(outliers),
                        "total_samples": len(errors),
                        "worst_outliers": outliers[:5],
                    },
                    "suggested_investigation": (
                        f"Examine the {len(outliers)} outlier samples. "
                        f"Check if they share physical characteristics (location, intensity, season). "
                        f"These may represent a missing feature or model blind spot."
                    ),
                })

        return anomalies

    def _check_ri_blindspot(self, per_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check if rapid intensification events are systematically missed."""
        anomalies = []

        ri_samples = []
        non_ri_samples = []

        for s in per_sample:
            # Try multiple field names for wind change
            dwind = self._get_field(s, "actual_dwind_24h", "delta_wind_24h", "dwind_24h")
            if dwind is None:
                # Try to compute from truth and predicted wind at different leads
                truth_wind = self._get_field(s, "truth_wind_24h", "wind_24h_truth",
                                             "wind_truth_24h")
                init_wind = self._get_field(s, "initial_wind", "wind_0h", "wind0",
                                            "wind_truth_0h")
                if truth_wind is not None and init_wind is not None:
                    dwind = truth_wind - init_wind

            if dwind is None:
                continue

            wind_error = abs(self._get_field(
                s, "wind_error_kt", "wind_mae_kt", "wind_error_24h"
            ) or 0)

            if dwind >= self.ri_threshold_kt:
                ri_samples.append({"sample": s, "dwind": dwind, "error": wind_error})
            else:
                non_ri_samples.append({"sample": s, "dwind": dwind, "error": wind_error})

        if len(ri_samples) >= 2 and non_ri_samples:
            ri_errors = [s["error"] for s in ri_samples]
            non_ri_errors = [s["error"] for s in non_ri_samples]
            ri_mean = statistics.mean(ri_errors)
            non_ri_mean = statistics.mean(non_ri_errors)
            ratio = ri_mean / max(non_ri_mean, 0.1)

            if ratio > 1.5:
                anomalies.append({
                    "type": "RI_BLIND_SPOT",
                    "description": (
                        f"Rapid intensification events (ΔV≥{self.ri_threshold_kt}kt/24h) have "
                        f"{ratio:.1f}x higher wind errors than non-RI events. "
                        f"RI MAE: {ri_mean:.1f}kt vs non-RI MAE: {non_ri_mean:.1f}kt "
                        f"({len(ri_samples)} RI events out of {len(ri_samples) + len(non_ri_samples)} total)"
                    ),
                    "severity": min(1.0, ratio / 3.0),
                    "evidence": {
                        "ri_count": len(ri_samples),
                        "non_ri_count": len(non_ri_samples),
                        "ri_wind_mae": round(ri_mean, 2),
                        "non_ri_wind_mae": round(non_ri_mean, 2),
                        "error_ratio": round(ratio, 2),
                    },
                    "suggested_investigation": (
                        "RI events are systematically under-predicted. Investigate: "
                        "1) Does adding ocean heat content (OHC) or sea surface temperature (SST) "
                        "as features improve RI detection? "
                        "2) Is the model's intensity change distribution too narrow (regression to mean)? "
                        "3) Would a hybrid approach (LLM + dedicated RI classifier) help?"
                    ),
                })

        return anomalies

    def _check_bias_pattern(self, per_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check for systematic over/under-prediction bias."""
        anomalies = []

        # Signed wind errors (predicted - truth)
        signed_errors = []
        for s in per_sample:
            pred = self._get_field(s, "predicted_wind", "wind_predicted",
                                   "wind_predicted_24h")
            truth = self._get_field(s, "truth_wind", "wind_truth",
                                    "truth_wind_24h", "wind_truth_24h")
            if pred is not None and truth is not None:
                signed_errors.append(pred - truth)

        if len(signed_errors) >= 10:
            mean_bias = statistics.mean(signed_errors)
            abs_mean = statistics.mean([abs(e) for e in signed_errors])

            bias_pct = abs(mean_bias) / max(abs_mean, 0.1) * 100

            if bias_pct > self.bias_threshold_pct:
                direction = "over-predicting" if mean_bias > 0 else "under-predicting"
                anomalies.append({
                    "type": "BIAS_PATTERN",
                    "description": (
                        f"Systematic {direction} bias detected. "
                        f"Mean signed error: {mean_bias:+.1f}kt "
                        f"(bias accounts for {bias_pct:.0f}% of total error). "
                        f"The model consistently {direction} wind intensity."
                    ),
                    "severity": min(1.0, bias_pct / 50.0),
                    "evidence": {
                        "mean_signed_error": round(mean_bias, 2),
                        "mean_abs_error": round(abs_mean, 2),
                        "bias_percentage": round(bias_pct, 1),
                        "direction": direction,
                        "n_samples": len(signed_errors),
                    },
                    "suggested_investigation": (
                        f"The model is systematically {direction}. "
                        f"Investigate: 1) Is the training data / prompt biased toward "
                        f"{'stronger' if mean_bias > 0 else 'weaker'} storms? "
                        f"2) Does calibration / bias correction improve forecasts? "
                        f"3) Are there specific regimes where the bias is worse?"
                    ),
                })

        return anomalies

    def _check_regime_clusters(self, per_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check if errors cluster by physical regime (lat, intensity, basin)."""
        anomalies = []

        # Group by latitude bands
        lat_bins: dict[str, list[float]] = {}
        for s in per_sample:
            lat = self._get_field(s, "lat", "latitude", "lat_0",
                                  "lat_truth_24h")
            error = self._get_field(s, "track_error_km", "track_mae_km",
                                    "track_error_24h")
            if lat is not None and error is not None:
                band = f"{int(lat // 10) * 10}-{int(lat // 10) * 10 + 10}"
                lat_bins.setdefault(band, []).append(error)

        if len(lat_bins) >= 2:
            bin_means = {k: statistics.mean(v) for k, v in lat_bins.items() if len(v) >= 3}
            if len(bin_means) >= 2:
                overall_mean = statistics.mean(
                    [e for errors in lat_bins.values() for e in errors]
                )
                worst_band = max(bin_means, key=bin_means.get)
                best_band = min(bin_means, key=bin_means.get)
                spread = bin_means[worst_band] / max(bin_means[best_band], 0.1)

                if spread > 2.0:
                    anomalies.append({
                        "type": "REGIME_SHIFT",
                        "description": (
                            f"Track errors vary {spread:.1f}x across latitude bands. "
                            f"Worst: {worst_band}° ({bin_means[worst_band]:.0f}km), "
                            f"Best: {best_band}° ({bin_means[best_band]:.0f}km). "
                            f"The model struggles more at {'higher' if int(worst_band.split('-')[0]) > 20 else 'lower'} latitudes."
                        ),
                        "severity": min(1.0, spread / 5.0),
                        "evidence": {
                            "bin_means": {k: round(v, 1) for k, v in bin_means.items()},
                            "bin_counts": {k: len(v) for k, v in lat_bins.items()},
                            "worst_band": worst_band,
                            "best_band": best_band,
                            "spread_ratio": round(spread, 2),
                        },
                        "suggested_investigation": (
                            f"Errors are significantly worse in the {worst_band}° latitude band. "
                            f"Investigate: 1) Are there different dynamics at play (recurvature, "
                            f"extratropical transition)? 2) Does adding steering flow or "
                            f"environmental data for this regime help?"
                        ),
                    })

        # Group by initial intensity
        intensity_bins: dict[str, list[float]] = {}
        for s in per_sample:
            wind = self._get_field(s, "initial_wind", "wind0", "wind_0h",
                                   "wind_truth_24h")  # fallback: use truth as proxy
            error = self._get_field(s, "wind_error_kt", "wind_mae_kt",
                                    "wind_error_24h")
            if wind is not None and error is not None:
                if wind < 34:
                    cat = "TD (<34kt)"
                elif wind < 64:
                    cat = "TS (34-63kt)"
                elif wind < 96:
                    cat = "Cat1-2 (64-95kt)"
                else:
                    cat = "Major (≥96kt)"
                intensity_bins.setdefault(cat, []).append(abs(error))

        if len(intensity_bins) >= 2:
            bin_means = {k: statistics.mean(v) for k, v in intensity_bins.items() if len(v) >= 3}
            if len(bin_means) >= 2:
                worst_cat = max(bin_means, key=bin_means.get)
                best_cat = min(bin_means, key=bin_means.get)
                spread = bin_means[worst_cat] / max(bin_means[best_cat], 0.1)

                if spread > 1.8:
                    anomalies.append({
                        "type": "REGIME_SHIFT",
                        "description": (
                            f"Wind errors vary {spread:.1f}x across intensity categories. "
                            f"Worst: {worst_cat} ({bin_means[worst_cat]:.1f}kt MAE), "
                            f"Best: {best_cat} ({bin_means[best_cat]:.1f}kt MAE)."
                        ),
                        "severity": min(1.0, spread / 4.0),
                        "evidence": {
                            "category_means": {k: round(v, 1) for k, v in bin_means.items()},
                            "category_counts": {k: len(v) for k, v in intensity_bins.items()},
                            "worst_category": worst_cat,
                            "spread_ratio": round(spread, 2),
                        },
                        "suggested_investigation": (
                            f"The model performs worst on {worst_cat} storms. "
                            f"Investigate intensity-dependent prompt engineering "
                            f"or separate models for different intensity regimes."
                        ),
                    })

        return anomalies

    def _check_error_trend(self, per_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check if error growth with lead time is abnormal."""
        anomalies = []

        lead_errors: dict[int, list[float]] = {}
        for s in per_sample:
            for lead in [24, 48, 72]:
                key_track = f"track_error_{lead}h"
                error = s.get(key_track)
                if error is not None:
                    lead_errors.setdefault(lead, []).append(error)

        if len(lead_errors) >= 2:
            lead_means = {k: statistics.mean(v) for k, v in lead_errors.items() if v}
            leads = sorted(lead_means.keys())

            if len(leads) >= 2:
                # Check if error growth is super-linear
                first = lead_means[leads[0]]
                last = lead_means[leads[-1]]
                time_ratio = leads[-1] / leads[0]
                error_ratio = last / max(first, 0.1)

                # For good forecasts, error grows roughly linearly with lead time
                # Super-linear growth suggests compounding errors
                if error_ratio > time_ratio * 1.5:
                    anomalies.append({
                        "type": "ERROR_TREND",
                        "description": (
                            f"Error growth is super-linear: {leads[0]}h→{leads[-1]}h "
                            f"errors grow {error_ratio:.1f}x while lead time grows {time_ratio:.1f}x. "
                            f"This suggests compounding forecast errors."
                        ),
                        "severity": min(1.0, error_ratio / (time_ratio * 3)),
                        "evidence": {
                            "lead_means": {str(k): round(v, 1) for k, v in lead_means.items()},
                            "error_ratio": round(error_ratio, 2),
                            "time_ratio": round(time_ratio, 2),
                        },
                        "suggested_investigation": (
                            "Forecast errors compound with lead time faster than expected. "
                            "Investigate: 1) Is the model ignoring temporal dynamics? "
                            "2) Would iterative (autoregressive) forecasting help? "
                            "3) Are specific storm phases (e.g., recurvature) causing late-lead blowup?"
                        ),
                    })

        return anomalies

    def _check_metric_anomalies(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Check summary metrics for red flags."""
        anomalies = []

        # Check if valid_json_rate is low
        vjr = metrics.get("valid_json_rate", metrics.get("valid_json", 1.0))
        if isinstance(vjr, (int, float)) and vjr < 0.9:
            anomalies.append({
                "type": "OUTLIER",
                "description": (
                    f"Low valid JSON rate: {vjr:.1%}. "
                    f"The model is producing malformed outputs."
                ),
                "severity": min(1.0, (1.0 - vjr) * 2),
                "evidence": {"valid_json_rate": vjr},
                "suggested_investigation": (
                    "Improve the output schema enforcement. Consider: "
                    "1) Stricter system prompt with JSON examples "
                    "2) Post-processing with JSON repair "
                    "3) Using structured output mode if available"
                ),
            })

        return anomalies

    # ------------------------------------------------------------------
    # LLM-augmented interpretation
    # ------------------------------------------------------------------

    def _llm_interpret(
        self,
        anomalies: list[dict[str, Any]],
        experiment_results: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Ask LLM to provide deeper interpretation of detected anomalies."""
        if not self.inference:
            return anomalies

        anomaly_text = "\n".join(
            f"{i+1}. [{a['type']}] {a['description']}"
            for i, a in enumerate(anomalies[:8])
        )

        metrics_text = json.dumps(
            experiment_results.get("metrics", {}), indent=2
        )[:2000]

        prompt = (
            "You are a climate science expert analyzing experiment results.\n\n"
            f"Summary metrics:\n{metrics_text}\n\n"
            f"Detected anomalies:\n{anomaly_text}\n\n"
            "For each anomaly, provide:\n"
            "1. A physical explanation (what climate process might cause this?)\n"
            "2. The most promising experiment to investigate it\n"
            "3. What data sources would help\n\n"
            "Respond as JSON array with objects: "
            '{"index": 1, "physical_explanation": "...", "experiment": "...", "data_needed": "..."}'
        )

        try:
            response = self.inference.chat([
                {"role": "system", "content": "You are a climate science research advisor. Respond only in JSON."},
                {"role": "user", "content": prompt},
            ])

            interpretations = json.loads(response.content or "[]")
            if isinstance(interpretations, list):
                for interp in interpretations:
                    idx = interp.get("index", 0) - 1
                    if 0 <= idx < len(anomalies):
                        anomalies[idx]["physical_explanation"] = interp.get("physical_explanation", "")
                        anomalies[idx]["suggested_experiment"] = interp.get("experiment", "")
                        anomalies[idx]["data_needed"] = interp.get("data_needed", "")

        except Exception as exc:
            log.warning("[ANOMALY] LLM interpretation failed: %s", exc)

        return anomalies
