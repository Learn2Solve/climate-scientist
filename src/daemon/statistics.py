"""Statistical rigor validation for experimental results.

Runs standard hypothesis tests (t-test, Mann-Whitney, chi-squared, etc.)
via sandboxed scipy scripts and validates that experiments meet minimum
rigor standards before findings are accepted.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.statistics")

# Supported test types and their required data keys
_TEST_DATA_KEYS: dict[str, set[str]] = {
    "t_test": {"group_a", "group_b"},
    "mann_whitney": {"group_a", "group_b"},
    "chi_squared": {"observed", "expected"},
    "bootstrap_ci": {"values"},
    "paired_t_test": {"before", "after"},
    "wilcoxon": {"before", "after"},
}


class StatisticalRigor:
    """Run and validate statistical tests for experimental rigor."""

    def __init__(self, config: Any, db: Any, sandbox_runner: Any) -> None:
        self._config = config
        self._db = db
        self._runner = sandbox_runner

    # -- Public API ----------------------------------------------------------

    def run_test(
        self,
        test_type: str,
        data: dict[str, Any],
        description: str = "",
        linked_finding_id: str | None = None,
        linked_execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Run a statistical test via sandbox and store results.

        Args:
            test_type: One of t_test, mann_whitney, chi_squared,
                       bootstrap_ci, paired_t_test, wilcoxon.
            data: Test-specific data dict (see _TEST_DATA_KEYS).
            description: Human-readable description.
            linked_finding_id: Optional finding to associate with.
            linked_execution_id: Optional execution to associate with.

        Returns:
            {"test_id": str, "passed": bool, "p_value": float, "result": dict}
        """
        if test_type not in _TEST_DATA_KEYS:
            raise ValueError(
                f"Unknown test type '{test_type}'. "
                f"Supported: {sorted(_TEST_DATA_KEYS)}"
            )

        required = _TEST_DATA_KEYS[test_type]
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Missing required data keys for {test_type}: {missing}"
            )

        # Generate and execute the test script
        script = self._generate_test_script(test_type, data)
        result = self._runner.execute(
            script=script,
            requirements=["scipy", "numpy"],
            timeout_s=60,
            description=f"Statistical test: {test_type}" + (
                f" - {description}" if description else ""
            ),
        )

        if result.returncode != 0:
            log.error(
                "Statistical test %s failed: %s", test_type, result.stderr
            )
            # Store failed test
            test_id = str(ULID())
            self._db.insert_statistical_test({
                "id": test_id,
                "test_type": test_type,
                "description": description,
                "input_data": data,
                "result": {"error": result.stderr},
                "passed": False,
                "p_value": None,
                "confidence_level": 1.0 - self._config.significance_threshold,
                "linked_finding_id": linked_finding_id,
                "linked_execution_id": linked_execution_id,
            })
            return {
                "test_id": test_id,
                "passed": False,
                "p_value": None,
                "result": {"error": result.stderr},
            }

        # Parse results from sandbox
        test_result = result.metrics or {}
        p_value = test_result.get("p_value")
        passed = (
            p_value is not None
            and p_value < self._config.significance_threshold
        )

        test_id = str(ULID())
        self._db.insert_statistical_test({
            "id": test_id,
            "test_type": test_type,
            "description": description,
            "input_data": data,
            "result": test_result,
            "passed": passed,
            "p_value": p_value,
            "confidence_level": 1.0 - self._config.significance_threshold,
            "linked_finding_id": linked_finding_id,
            "linked_execution_id": linked_execution_id,
        })

        log.info(
            "Statistical test %s (id=%s): p=%.4g passed=%s",
            test_type, test_id,
            p_value if p_value is not None else float("nan"),
            passed,
        )
        return {
            "test_id": test_id,
            "passed": passed,
            "p_value": p_value,
            "result": test_result,
        }

    def validate_experiment_rigor(self, execution_id: str) -> dict[str, Any]:
        """Check if an experiment meets statistical rigor standards.

        Checks:
        1. Has associated statistical test(s)?
        2. All tests passed (p < significance_threshold)?
        3. Sample sizes adequate (> 30)?

        Returns:
            {"rigorous": bool, "issues": list[str],
             "tests_run": int, "tests_passed": int}
        """
        # Get all verifications/tests linked to this execution
        # Statistical tests can be linked via finding or execution
        tests: list[dict[str, Any]] = []

        # Check tests linked directly to execution
        all_tests = self._db.get_tests_for_finding(execution_id)
        tests.extend(all_tests)

        # Also check by execution_id - get the test by querying
        # (get_tests_for_finding uses linked_finding_id, we also need execution-linked)
        exec_test = self._db.get_statistical_test(execution_id)
        if exec_test and exec_test["id"] not in {t["id"] for t in tests}:
            tests.append(exec_test)

        issues: list[str] = []
        tests_passed = sum(1 for t in tests if t.get("passed"))

        # Check 1: Has tests?
        if not tests:
            if self._config.require_statistical_test:
                issues.append(
                    "No statistical tests found for this execution. "
                    "Config requires at least one."
                )
            else:
                return {
                    "rigorous": True,
                    "issues": [],
                    "tests_run": 0,
                    "tests_passed": 0,
                }

        # Check 2: All tests passed?
        failed = [t for t in tests if not t.get("passed")]
        for t in failed:
            p = t.get("p_value")
            issues.append(
                f"Test '{t['test_type']}' failed: "
                f"p={p if p is not None else 'N/A'} "
                f"(threshold={self._config.significance_threshold})"
            )

        # Check 3: Sample size adequacy
        for t in tests:
            input_data = t.get("input_data", {})
            for key in ("group_a", "group_b", "before", "after", "values"):
                samples = input_data.get(key)
                if isinstance(samples, list) and len(samples) < 30:
                    issues.append(
                        f"Test '{t['test_type']}': {key} has only "
                        f"{len(samples)} samples (recommended > 30)"
                    )

        rigorous = len(issues) == 0
        return {
            "rigorous": rigorous,
            "issues": issues,
            "tests_run": len(tests),
            "tests_passed": tests_passed,
        }

    # -- Script generation ---------------------------------------------------

    def _generate_test_script(
        self, test_type: str, data: dict[str, Any]
    ) -> str:
        """Generate a Python script that runs a scipy.stats test.

        The script writes results.json with:
        {"test_type", "statistic", "p_value", "effect_size", "details"}
        """
        data_json = json.dumps(data)

        if test_type == "t_test":
            return _T_TEST_SCRIPT.format(data_json=data_json)
        elif test_type == "mann_whitney":
            return _MANN_WHITNEY_SCRIPT.format(data_json=data_json)
        elif test_type == "chi_squared":
            return _CHI_SQUARED_SCRIPT.format(data_json=data_json)
        elif test_type == "bootstrap_ci":
            return _BOOTSTRAP_CI_SCRIPT.format(data_json=data_json)
        elif test_type == "paired_t_test":
            return _PAIRED_T_TEST_SCRIPT.format(data_json=data_json)
        elif test_type == "wilcoxon":
            return _WILCOXON_SCRIPT.format(data_json=data_json)
        else:
            raise ValueError(f"No script template for test type: {test_type}")


# -- Script templates --------------------------------------------------------

_T_TEST_SCRIPT = textwrap.dedent("""\
    import json
    import numpy as np
    from scipy import stats

    data = json.loads('''{data_json}''')
    a = np.array(data["group_a"], dtype=float)
    b = np.array(data["group_b"], dtype=float)

    stat, p = stats.ttest_ind(a, b)

    # Cohen's d
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    effect_size = float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0

    result = {{
        "test_type": "t_test",
        "statistic": float(stat),
        "p_value": float(p),
        "effect_size": effect_size,
        "details": {{
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "std_a": float(np.std(a, ddof=1)),
            "std_b": float(np.std(b, ddof=1)),
            "n_a": len(a),
            "n_b": len(b),
        }},
    }}
    with open("results.json", "w") as f:
        json.dump(result, f)
""")

_MANN_WHITNEY_SCRIPT = textwrap.dedent("""\
    import json
    import numpy as np
    from scipy import stats

    data = json.loads('''{data_json}''')
    a = np.array(data["group_a"], dtype=float)
    b = np.array(data["group_b"], dtype=float)

    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")

    # Rank-biserial correlation as effect size
    n = len(a) * len(b)
    effect_size = float(1 - (2 * stat) / n) if n > 0 else 0.0

    result = {{
        "test_type": "mann_whitney",
        "statistic": float(stat),
        "p_value": float(p),
        "effect_size": effect_size,
        "details": {{
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "n_a": len(a),
            "n_b": len(b),
        }},
    }}
    with open("results.json", "w") as f:
        json.dump(result, f)
""")

_CHI_SQUARED_SCRIPT = textwrap.dedent("""\
    import json
    import numpy as np
    from scipy import stats

    data = json.loads('''{data_json}''')
    observed = np.array(data["observed"], dtype=float)
    expected = np.array(data["expected"], dtype=float)

    stat, p = stats.chisquare(observed, f_exp=expected)

    # Cramer's V as effect size (for 1D case, simplified)
    n = float(np.sum(observed))
    effect_size = float(np.sqrt(stat / n)) if n > 0 else 0.0

    result = {{
        "test_type": "chi_squared",
        "statistic": float(stat),
        "p_value": float(p),
        "effect_size": effect_size,
        "details": {{
            "observed": observed.tolist(),
            "expected": expected.tolist(),
            "n": int(n),
            "df": len(observed) - 1,
        }},
    }}
    with open("results.json", "w") as f:
        json.dump(result, f)
""")

_BOOTSTRAP_CI_SCRIPT = textwrap.dedent("""\
    import json
    import numpy as np

    data = json.loads('''{data_json}''')
    values = np.array(data["values"], dtype=float)
    statistic_name = data.get("statistic", "mean")
    n_bootstrap = data.get("n_bootstrap", 1000)

    rng = np.random.default_rng(42)

    if statistic_name == "mean":
        stat_fn = np.mean
    elif statistic_name == "median":
        stat_fn = np.median
    else:
        stat_fn = np.mean

    observed_stat = float(stat_fn(values))

    # Bootstrap resampling
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats[i] = stat_fn(sample)

    ci_lower = float(np.percentile(boot_stats, 2.5))
    ci_upper = float(np.percentile(boot_stats, 97.5))
    boot_se = float(np.std(boot_stats, ddof=1))

    # p-value: proportion of bootstrap samples on wrong side of zero
    p_value = float(np.mean(boot_stats < 0)) if observed_stat > 0 else float(np.mean(boot_stats > 0))
    p_value = min(p_value * 2, 1.0)  # two-sided

    result = {{
        "test_type": "bootstrap_ci",
        "statistic": observed_stat,
        "p_value": p_value,
        "effect_size": float(observed_stat / boot_se) if boot_se > 0 else 0.0,
        "details": {{
            "statistic_name": statistic_name,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "bootstrap_se": boot_se,
            "n_bootstrap": n_bootstrap,
            "n_samples": len(values),
        }},
    }}
    with open("results.json", "w") as f:
        json.dump(result, f)
""")

_PAIRED_T_TEST_SCRIPT = textwrap.dedent("""\
    import json
    import numpy as np
    from scipy import stats

    data = json.loads('''{data_json}''')
    before = np.array(data["before"], dtype=float)
    after = np.array(data["after"], dtype=float)

    stat, p = stats.ttest_rel(before, after)

    # Cohen's d for paired samples
    diff = after - before
    effect_size = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0

    result = {{
        "test_type": "paired_t_test",
        "statistic": float(stat),
        "p_value": float(p),
        "effect_size": effect_size,
        "details": {{
            "mean_before": float(np.mean(before)),
            "mean_after": float(np.mean(after)),
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "n_pairs": len(before),
        }},
    }}
    with open("results.json", "w") as f:
        json.dump(result, f)
""")

_WILCOXON_SCRIPT = textwrap.dedent("""\
    import json
    import numpy as np
    from scipy import stats

    data = json.loads('''{data_json}''')
    before = np.array(data["before"], dtype=float)
    after = np.array(data["after"], dtype=float)

    stat, p = stats.wilcoxon(after - before)

    # Matched-pairs rank-biserial correlation
    diff = after - before
    n = len(diff[diff != 0])
    effect_size = float(1 - (2 * stat) / (n * (n + 1) / 2)) if n > 0 else 0.0

    result = {{
        "test_type": "wilcoxon",
        "statistic": float(stat),
        "p_value": float(p),
        "effect_size": effect_size,
        "details": {{
            "median_before": float(np.median(before)),
            "median_after": float(np.median(after)),
            "median_diff": float(np.median(diff)),
            "n_pairs": len(before),
            "n_nonzero_diffs": int(n),
        }},
    }}
    with open("results.json", "w") as f:
        json.dump(result, f)
""")
