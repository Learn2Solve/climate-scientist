"""External benchmark validation - ground-truth comparison for climate research."""

from __future__ import annotations

import logging
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.benchmark")


# Pre-populated known results for climate/RI domain
CLIMATE_BENCHMARKS: dict[str, dict[str, Any]] = {
    "hurdat2_ri_base_rate": {
        "metric": "ri_rate_fraction",
        "value": 0.043,
        "tolerance": 0.01,
        "source": "Kaplan & DeMaria 2003",
        "description": "Historical RI rate (30kt/24h) in Atlantic basin",
    },
    "persistence_model_rmse": {
        "metric": "intensity_rmse_kt",
        "value": 11.2,
        "tolerance": 2.0,
        "source": "DeMaria et al. 2005",
        "description": "Persistence model 24h intensity forecast RMSE",
    },
    "ships_ri_pod": {
        "metric": "probability_of_detection",
        "value": 0.35,
        "tolerance": 0.10,
        "source": "Kaplan et al. 2010",
        "description": "SHIPS-RI probability of detection baseline",
    },
    "climatology_brier_skill": {
        "metric": "brier_skill_score",
        "value": 0.0,
        "tolerance": 0.05,
        "source": "Climatological baseline",
        "description": "Climatology reference BSS (by definition 0)",
    },
    "atlantic_sst_climatology": {
        "metric": "mean_sst_mdi_celsius",
        "value": 27.5,
        "tolerance": 1.0,
        "source": "Reynolds et al. 2002",
        "description": "Atlantic MDI SST climatology (June-November)",
    },
    "ri_shear_threshold": {
        "metric": "wind_shear_threshold_kt",
        "value": 20.0,
        "tolerance": 5.0,
        "source": "DeMaria 1996",
        "description": "Wind shear threshold discriminating RI from non-RI",
    },
    "tc_pi_max": {
        "metric": "max_potential_intensity_kt",
        "value": 170.0,
        "tolerance": 20.0,
        "source": "Emanuel 1988",
        "description": "Maximum potential intensity for warmest SSTs",
    },
    "hurdat2_annual_count": {
        "metric": "annual_named_storm_count",
        "value": 12.1,
        "tolerance": 3.0,
        "source": "NHC climatology 1991-2020",
        "description": "Average annual named storm count in Atlantic",
    },
    "ri_false_alarm_ratio": {
        "metric": "false_alarm_ratio",
        "value": 0.70,
        "tolerance": 0.15,
        "source": "Kaplan et al. 2010",
        "description": "SHIPS-RI false alarm ratio baseline",
    },
    "lorenz63_max_lyapunov": {
        "metric": "max_lyapunov_exponent",
        "value": 0.906,
        "tolerance": 0.05,
        "source": "Lorenz 1963; Sparrow 1982",
        "description": "Maximum Lyapunov exponent of Lorenz 63 attractor",
    },
}


class BenchmarkValidator:
    def __init__(self, config: Any, db: Any) -> None:
        self.config = config
        self.db = db

    def register_benchmark(self, name: str, metric: str, ground_truth_value: float,
                           tolerance: float = 0.05, source: str = "",
                           description: str = "", domain: str = "climate") -> str:
        """Register a new benchmark."""
        bm_id = str(ULID())
        self.db.insert_benchmark({
            "id": bm_id,
            "name": name,
            "metric": metric,
            "ground_truth_value": ground_truth_value,
            "tolerance": tolerance,
            "source": source,
            "description": description,
            "domain": domain,
        })
        return bm_id

    def seed_defaults(self) -> int:
        """Seed CLIMATE_BENCHMARKS into DB if not already present."""
        count = 0
        for name, info in CLIMATE_BENCHMARKS.items():
            existing = self.db.get_benchmark_by_name(name)
            if existing:
                continue
            self.register_benchmark(
                name=name,
                metric=info["metric"],
                ground_truth_value=info["value"],
                tolerance=info["tolerance"],
                source=info.get("source", ""),
                description=info.get("description", ""),
            )
            count += 1
        return count

    def validate_against_benchmark(self, benchmark_name: str, measured_value: float,
                                   execution_id: str | None = None,
                                   notes: str = "") -> dict[str, Any]:
        """Compare a measured value against a registered benchmark."""
        bm = self.db.get_benchmark_by_name(benchmark_name)
        if not bm:
            return {"error": f"Benchmark not found: {benchmark_name}"}

        gt = bm["ground_truth_value"]
        tol = bm["tolerance"]
        delta = abs(measured_value - gt)
        passed = delta <= tol

        run_id = str(ULID())
        self.db.insert_benchmark_run({
            "id": run_id,
            "benchmark_id": bm["id"],
            "measured_value": measured_value,
            "passed": passed,
            "delta": round(delta, 6),
            "execution_id": execution_id,
            "notes": notes,
        })

        return {
            "run_id": run_id,
            "passed": passed,
            "benchmark_name": benchmark_name,
            "benchmark_value": gt,
            "measured": measured_value,
            "delta": round(delta, 6),
            "tolerance": tol,
            "within_tolerance": passed,
            "source": bm.get("source", ""),
        }

    def validate_execution(self, execution_id: str) -> dict[str, Any]:
        """Auto-match sandbox execution metrics against all relevant benchmarks."""
        execution = self.db.get_sandbox_execution(execution_id)
        if not execution:
            return {"error": f"Execution not found: {execution_id}"}

        metrics = execution.get("metrics", {})
        if not metrics:
            return {"checks": [], "passed_count": 0, "failed_count": 0,
                    "note": "No metrics in execution"}

        benchmarks = self.db.list_benchmarks()
        checks: list[dict[str, Any]] = []

        for bm in benchmarks:
            metric_name = bm["metric"]
            if metric_name in metrics:
                result = self.validate_against_benchmark(
                    bm["name"], float(metrics[metric_name]),
                    execution_id=execution_id,
                )
                checks.append(result)

        passed = sum(1 for c in checks if c.get("passed"))
        failed = sum(1 for c in checks if not c.get("passed") and "error" not in c)

        return {
            "checks": checks,
            "passed_count": passed,
            "failed_count": failed,
        }

    def get_benchmark_report(self) -> dict[str, Any]:
        """Aggregate report of all benchmark runs."""
        benchmarks = self.db.list_benchmarks()
        all_runs = self.db.get_all_benchmark_runs()

        total_runs = len(all_runs)
        passed_runs = sum(1 for r in all_runs if r.get("passed"))
        pass_rate = (passed_runs / total_runs) if total_runs > 0 else 0.0

        details: list[dict[str, Any]] = []
        for bm in benchmarks:
            runs = self.db.get_benchmark_runs(bm["id"], limit=10)
            recent_pass = sum(1 for r in runs if r.get("passed"))
            details.append({
                "name": bm["name"],
                "metric": bm["metric"],
                "ground_truth": bm["ground_truth_value"],
                "tolerance": bm["tolerance"],
                "total_runs": len(runs),
                "passed": recent_pass,
                "source": bm.get("source", ""),
            })

        return {
            "benchmarks_registered": len(benchmarks),
            "total_runs": total_runs,
            "pass_rate": round(pass_rate, 4),
            "details": details,
        }

    def list_benchmarks(self, domain: str | None = None) -> list[dict[str, Any]]:
        return self.db.list_benchmarks(domain=domain)
