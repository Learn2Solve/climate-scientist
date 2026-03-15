"""Closed-loop parameter optimization for experiments.

Manages iterative experiment loops: random exploration followed by
exploitation around best-known parameters. Tracks trials, detects
convergence, and persists state in the experiment_loops / loop_trials tables.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.experiment_loop")


class ExperimentLoop:
    """Parameter optimization loop with random-then-exploit strategy."""

    def __init__(self, config: Any, db: Any, sandbox_runner: Any) -> None:
        self._config = config
        self._db = db
        self._runner = sandbox_runner

    # -- Public API ----------------------------------------------------------

    def create_loop(
        self,
        name: str,
        parameter_space: dict[str, dict[str, Any]],
        target_metric: str,
        objective: str = "minimize",
        description: str = "",
        linked_hypothesis_id: str | None = None,
    ) -> str:
        """Create a new experiment loop. Returns loop_id."""
        loop_id = str(ULID())
        self._db.insert_experiment_loop({
            "id": loop_id,
            "name": name,
            "description": description,
            "parameter_space": parameter_space,
            "objective": objective,
            "target_metric": target_metric,
            "status": "active",
            "linked_hypothesis_id": linked_hypothesis_id,
        })
        log.info(
            "Created experiment loop %s (%s) optimizing %s (%s)",
            loop_id, name, target_metric, objective,
        )
        return loop_id

    def suggest_next_params(self, loop_id: str) -> dict[str, Any]:
        """Suggest the next set of parameters to try.

        Strategy:
        - First N trials (N = len(parameter_space) * 2): random sampling.
        - After that: gaussian perturbation around the best parameters.

        Returns: {"params": {...}, "trial_number": int, "strategy": str}
        """
        loop = self._db.get_experiment_loop(loop_id)
        if loop is None:
            raise ValueError(f"Experiment loop {loop_id} not found")

        param_space = loop["parameter_space"]
        trials = self._db.get_loop_trials(loop_id)
        trial_number = len(trials) + 1

        # Check if we hit the max
        max_trials = self._config.experiment_loop_max_trials
        if trial_number > max_trials:
            raise RuntimeError(
                f"Loop {loop_id} reached max trials ({max_trials})"
            )

        exploration_count = len(param_space) * 2
        completed = [t for t in trials if t["status"] == "completed"]

        if len(completed) < exploration_count:
            params = _random_sample(param_space)
            strategy = "random"
        else:
            best_params = self._find_best_params(loop, completed)
            params = _perturb_params(param_space, best_params)
            strategy = "exploitation"

        # Insert a pending trial
        trial_id = str(ULID())
        self._db.insert_loop_trial({
            "id": trial_id,
            "loop_id": loop_id,
            "trial_number": trial_number,
            "params": params,
            "metrics": {},
            "status": "pending",
        })

        log.info(
            "Loop %s trial %d: strategy=%s params=%s",
            loop_id, trial_number, strategy, params,
        )
        return {
            "trial_id": trial_id,
            "params": params,
            "trial_number": trial_number,
            "strategy": strategy,
        }

    def record_trial_result(
        self,
        loop_id: str,
        trial_id: str,
        metrics: dict[str, Any],
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Record metrics for a completed trial.

        Returns: {"improved": bool, "converged": bool, "best_value": float}
        """
        loop = self._db.get_experiment_loop(loop_id)
        if loop is None:
            raise ValueError(f"Experiment loop {loop_id} not found")

        target_metric = loop["target_metric"]
        target_value = metrics.get(target_metric)
        if target_value is None:
            raise ValueError(
                f"Target metric '{target_metric}' not found in metrics: "
                f"{list(metrics.keys())}"
            )

        # Update the trial
        self._db.update_loop_trial(
            trial_id,
            status="completed",
            target_value=target_value,
            metrics=metrics,
            execution_id=execution_id,
        )

        # Check if this improves on the current best
        improved = False
        objective = loop.get("objective", "minimize")
        current_best = loop.get("best_value")

        if current_best is None or _is_better(target_value, current_best, objective):
            improved = True
            # Get params from the trial
            trials = self._db.get_loop_trials(loop_id)
            trial = next((t for t in trials if t["id"] == trial_id), None)
            best_params = trial["params"] if trial else {}

            self._db.update_experiment_loop(
                loop_id,
                best_value=target_value,
                best_params=best_params,
                total_trials=len([t for t in trials if t["status"] == "completed"]),
            )
            current_best = target_value
            log.info(
                "Loop %s new best: %s = %s", loop_id, target_metric, target_value
            )

        # Update total_trials count even if not improved
        if not improved:
            trials = self._db.get_loop_trials(loop_id)
            completed_count = len([t for t in trials if t["status"] == "completed"])
            self._db.update_experiment_loop(loop_id, total_trials=completed_count)

        # Check convergence
        converged = self._check_convergence(loop_id)
        if converged:
            self._db.update_experiment_loop(loop_id, status="converged")
            log.info("Loop %s converged", loop_id)

        return {
            "improved": improved,
            "converged": converged,
            "best_value": current_best if current_best is not None else target_value,
        }

    def get_loop_status(self, loop_id: str) -> dict[str, Any]:
        """Get full loop status including trial history and convergence analysis."""
        loop = self._db.get_experiment_loop(loop_id)
        if loop is None:
            raise ValueError(f"Experiment loop {loop_id} not found")

        trials = self._db.get_loop_trials(loop_id)
        completed = [t for t in trials if t["status"] == "completed"]
        pending = [t for t in trials if t["status"] == "pending"]

        return {
            "loop": loop,
            "total_trials": len(trials),
            "completed_trials": len(completed),
            "pending_trials": len(pending),
            "trials": trials,
            "convergence": self._convergence_analysis(loop_id, completed),
        }

    def get_active_loops(self) -> list[dict[str, Any]]:
        """Get all active experiment loops."""
        return self._db.get_active_experiment_loops()

    # -- Private helpers -----------------------------------------------------

    def _find_best_params(
        self,
        loop: dict[str, Any],
        completed: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Find best params from completed trials, or fall back to loop's stored best."""
        objective = loop.get("objective", "minimize")
        if not completed:
            return loop.get("best_params", {})

        # Sort by target_value
        valid = [t for t in completed if t.get("target_value") is not None]
        if not valid:
            return loop.get("best_params", {})

        if objective == "minimize":
            best_trial = min(valid, key=lambda t: t["target_value"])
        else:
            best_trial = max(valid, key=lambda t: t["target_value"])

        return best_trial.get("params", {})

    def _check_convergence(self, loop_id: str) -> bool:
        """Check if the last 5 completed trials are within threshold of best."""
        loop = self._db.get_experiment_loop(loop_id)
        if loop is None:
            return False

        trials = self._db.get_loop_trials(loop_id)
        completed = [
            t for t in trials
            if t["status"] == "completed" and t.get("target_value") is not None
        ]

        if len(completed) < 5:
            return False

        best_value = loop.get("best_value")
        if best_value is None:
            return False

        threshold = self._config.experiment_loop_convergence_threshold
        last_five = completed[-5:]

        for trial in last_five:
            if abs(trial["target_value"] - best_value) > threshold:
                return False

        return True

    def _convergence_analysis(
        self,
        loop_id: str,
        completed: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Produce convergence analysis summary."""
        if not completed:
            return {"converged": False, "reason": "no completed trials"}

        loop = self._db.get_experiment_loop(loop_id)
        if loop is None:
            return {"converged": False, "reason": "loop not found"}

        best_value = loop.get("best_value")
        threshold = self._config.experiment_loop_convergence_threshold

        values = [
            t["target_value"] for t in completed
            if t.get("target_value") is not None
        ]
        if not values:
            return {"converged": False, "reason": "no target values"}

        last_five = values[-5:] if len(values) >= 5 else values
        spread = max(last_five) - min(last_five) if last_five else 0.0

        return {
            "converged": loop.get("status") == "converged",
            "best_value": best_value,
            "total_completed": len(completed),
            "last_5_spread": round(spread, 6),
            "threshold": threshold,
            "within_threshold": spread <= threshold if len(last_five) >= 5 else False,
        }


# -- Module-level helpers ----------------------------------------------------


def _random_sample(param_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample uniformly from the parameter space."""
    params: dict[str, Any] = {}
    for name, spec in param_space.items():
        ptype = spec.get("type", "float")
        if ptype == "float":
            params[name] = random.uniform(spec["low"], spec["high"])
        elif ptype == "int":
            params[name] = random.randint(spec["low"], spec["high"])
        elif ptype == "categorical":
            params[name] = random.choice(spec["choices"])
        else:
            log.warning("Unknown param type %s for %s, skipping", ptype, name)
    return params


def _perturb_params(
    param_space: dict[str, dict[str, Any]],
    best_params: dict[str, Any],
) -> dict[str, Any]:
    """Gaussian perturbation around best params (sigma = 0.1 * range)."""
    params: dict[str, Any] = {}
    for name, spec in param_space.items():
        ptype = spec.get("type", "float")
        base = best_params.get(name)

        if ptype == "float":
            low, high = spec["low"], spec["high"]
            if base is None:
                params[name] = random.uniform(low, high)
            else:
                sigma = 0.1 * (high - low)
                value = random.gauss(base, sigma)
                params[name] = max(low, min(high, value))

        elif ptype == "int":
            low, high = spec["low"], spec["high"]
            if base is None:
                params[name] = random.randint(low, high)
            else:
                sigma = 0.1 * (high - low)
                value = random.gauss(base, sigma)
                params[name] = max(low, min(high, round(value)))

        elif ptype == "categorical":
            # 80% keep best, 20% random
            if base is not None and random.random() < 0.8:
                params[name] = base
            else:
                params[name] = random.choice(spec["choices"])

        else:
            log.warning("Unknown param type %s for %s, skipping", ptype, name)

    return params


def _is_better(new: float, old: float, objective: str) -> bool:
    """Check if new value is better than old given the objective."""
    if objective == "minimize":
        return new < old
    return new > old
