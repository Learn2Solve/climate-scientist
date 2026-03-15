"""Causal reasoning - DAG construction, Granger causality, do-calculus estimation."""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.causal_reasoning")


class CausalReasoner:
    def __init__(self, config: Any, db: Any, sandbox_runner: Any, inference: Any | None = None) -> None:
        self.config = config
        self.db = db
        self.sandbox = sandbox_runner
        self.inference = inference

    def build_dag(self, name: str, nodes: list[str], edges: list[list[str]],
                  confounders: list[str] | None = None,
                  linked_hypothesis_id: str | None = None) -> dict[str, Any]:
        """Create a causal DAG. Validates acyclicity via topological sort in sandbox."""
        graph_id = str(ULID())
        confounders = confounders or []

        # Validate DAG structure via sandbox
        script = textwrap.dedent(f"""\
            import json
            import sys

            nodes = {json.dumps(nodes)}
            edges = {json.dumps(edges)}

            # Build adjacency list and check acyclicity
            adj = {{n: [] for n in nodes}}
            in_degree = {{n: 0 for n in nodes}}
            warnings = []

            for e in edges:
                if len(e) < 2:
                    warnings.append(f"Invalid edge: {{e}}")
                    continue
                src, dst = e[0], e[1]
                if src not in adj:
                    warnings.append(f"Unknown source node: {{src}}")
                    continue
                if dst not in adj:
                    warnings.append(f"Unknown target node: {{dst}}")
                    continue
                adj[src].append(dst)
                in_degree[dst] = in_degree.get(dst, 0) + 1

            # Kahn's algorithm for topological sort (cycle detection)
            queue = [n for n in nodes if in_degree.get(n, 0) == 0]
            sorted_nodes = []
            while queue:
                node = queue.pop(0)
                sorted_nodes.append(node)
                for neighbor in adj.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            is_dag = len(sorted_nodes) == len(nodes)
            if not is_dag:
                cycle_nodes = [n for n in nodes if n not in sorted_nodes]
                warnings.append(f"Cycle detected involving: {{cycle_nodes}}")

            result = {{
                "is_valid": is_dag,
                "topological_order": sorted_nodes,
                "warnings": warnings,
                "node_count": len(nodes),
                "edge_count": len(edges),
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)

        result = self.sandbox.execute(
            script=script, requirements=[],
            description=f"Validate causal DAG: {name}",
        )

        is_valid = True
        warnings_list: list[str] = []
        if result.metrics:
            is_valid = result.metrics.get("is_valid", True)
            warnings_list = result.metrics.get("warnings", [])

        self.db.insert_causal_graph({
            "id": graph_id,
            "name": name,
            "description": "",
            "nodes": nodes,
            "edges": edges,
            "confounders": confounders,
            "status": "valid" if is_valid else "invalid",
            "linked_hypothesis_id": linked_hypothesis_id,
        })

        return {
            "graph_id": graph_id,
            "nodes": nodes,
            "edges": edges,
            "is_valid": is_valid,
            "warnings": warnings_list,
        }

    def run_granger_test(self, graph_id: str, cause: str, effect: str,
                         data: dict[str, Any], max_lag: int = 5) -> dict[str, Any]:
        """Run Granger causality test between two time series."""
        graph = self.db.get_causal_graph(graph_id)
        if not graph:
            return {"error": f"Graph not found: {graph_id}"}

        cause_data = data.get(cause, [])
        effect_data = data.get(effect, [])
        if not cause_data or not effect_data:
            return {"error": f"Data required for both '{cause}' and '{effect}'"}

        script = textwrap.dedent(f"""\
            import json
            import numpy as np
            from statsmodels.tsa.stattools import grangercausalitytests

            cause_data = {json.dumps(cause_data)}
            effect_data = {json.dumps(effect_data)}
            max_lag = {max_lag}

            data_array = np.column_stack([effect_data, cause_data])

            results = grangercausalitytests(data_array, maxlag=max_lag, verbose=False)

            p_values_by_lag = {{}}
            best_lag = 1
            best_p = 1.0
            for lag in range(1, max_lag + 1):
                if lag in results:
                    test_results = results[lag][0]
                    # Use the ssr_ftest p-value
                    p_val = test_results["ssr_ftest"][1]
                    p_values_by_lag[str(lag)] = round(p_val, 6)
                    if p_val < best_p:
                        best_p = p_val
                        best_lag = lag

            significant = best_p < {self.config.causal_significance_threshold}

            result = {{
                "p_values_by_lag": p_values_by_lag,
                "significant": significant,
                "best_lag": best_lag,
                "best_p_value": round(best_p, 6),
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)

        sandbox_result = self.sandbox.execute(
            script=script,
            requirements=["statsmodels", "numpy"],
            description=f"Granger test: {cause} -> {effect}",
        )

        metrics = sandbox_result.metrics or {}
        estimate_id = str(ULID())

        self.db.insert_causal_estimate({
            "id": estimate_id,
            "graph_id": graph_id,
            "method": "granger",
            "treatment": cause,
            "outcome": effect,
            "estimate": None,
            "p_value": metrics.get("best_p_value"),
            "interpretation": f"Granger test: best lag={metrics.get('best_lag', '?')}, "
                            f"p={metrics.get('best_p_value', '?')}",
            "execution_id": sandbox_result.execution_id,
        })

        return {
            "estimate_id": estimate_id,
            "p_values_by_lag": metrics.get("p_values_by_lag", {}),
            "significant": metrics.get("significant", False),
            "best_lag": metrics.get("best_lag", 1),
            "best_p_value": metrics.get("best_p_value"),
        }

    def estimate_causal_effect(self, graph_id: str, treatment: str, outcome: str,
                               data: dict[str, Any],
                               method: str | None = None) -> dict[str, Any]:
        """Estimate causal effect using backdoor adjustment or propensity score."""
        graph = self.db.get_causal_graph(graph_id)
        if not graph:
            return {"error": f"Graph not found: {graph_id}"}

        method = method or self.config.causal_default_method
        confounders = graph.get("confounders", [])
        treatment_data = data.get(treatment, [])
        outcome_data = data.get(outcome, [])

        if not treatment_data or not outcome_data:
            return {"error": f"Data required for '{treatment}' and '{outcome}'"}

        # Build confounder data columns
        confounder_data = {c: data.get(c, []) for c in confounders if c in data}

        if method == "backdoor":
            script = self._backdoor_script(treatment, outcome, treatment_data,
                                           outcome_data, confounder_data)
        elif method == "propensity_score":
            script = self._propensity_script(treatment, outcome, treatment_data,
                                             outcome_data, confounder_data)
        elif method == "iv":
            instrument = data.get("instrument", [])
            script = self._iv_script(treatment, outcome, treatment_data,
                                     outcome_data, instrument)
        else:
            script = self._backdoor_script(treatment, outcome, treatment_data,
                                           outcome_data, confounder_data)

        sandbox_result = self.sandbox.execute(
            script=script,
            requirements=["numpy", "scipy", "scikit-learn"],
            description=f"Causal effect ({method}): {treatment} -> {outcome}",
        )

        metrics = sandbox_result.metrics or {}
        estimate_id = str(ULID())

        self.db.insert_causal_estimate({
            "id": estimate_id,
            "graph_id": graph_id,
            "method": method,
            "treatment": treatment,
            "outcome": outcome,
            "estimate": metrics.get("ate"),
            "ci_lower": metrics.get("ci_lower"),
            "ci_upper": metrics.get("ci_upper"),
            "p_value": metrics.get("p_value"),
            "interpretation": metrics.get("interpretation", ""),
            "execution_id": sandbox_result.execution_id,
        })

        return {
            "estimate_id": estimate_id,
            "ate": metrics.get("ate"),
            "ci_lower": metrics.get("ci_lower"),
            "ci_upper": metrics.get("ci_upper"),
            "p_value": metrics.get("p_value"),
            "method": method,
        }

    def test_dag_fit(self, graph_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Test whether observed data is consistent with proposed DAG."""
        graph = self.db.get_causal_graph(graph_id)
        if not graph:
            return {"error": f"Graph not found: {graph_id}"}

        nodes = graph["nodes"]
        edges = graph["edges"]

        script = textwrap.dedent(f"""\
            import json
            import numpy as np
            from scipy import stats

            data = {json.dumps(data)}
            edges = {json.dumps(edges)}
            nodes = {json.dumps(nodes)}

            # Build adjacency and parent sets
            parents = {{n: set() for n in nodes}}
            children = {{n: set() for n in nodes}}
            for e in edges:
                if len(e) >= 2:
                    parents[e[1]].add(e[0])
                    children[e[0]].add(e[1])

            # Test conditional independence implications
            # For each non-adjacent pair, test if they are conditionally independent
            # given parents (simplified d-separation test)
            violations = []
            tests_run = 0
            for i, a in enumerate(nodes):
                for b in nodes[i+1:]:
                    # Check if a and b are directly connected
                    connected = any(
                        (e[0] == a and e[1] == b) or (e[0] == b and e[1] == a)
                        for e in edges if len(e) >= 2
                    )
                    if connected:
                        continue
                    # They should be conditionally independent given parents
                    if a not in data or b not in data:
                        continue
                    a_data = np.array(data[a])
                    b_data = np.array(data[b])
                    if len(a_data) != len(b_data) or len(a_data) < 10:
                        continue

                    tests_run += 1
                    # Simple partial correlation test
                    conditioning = list(parents[a] | parents[b])
                    if conditioning:
                        cond_data = []
                        skip = False
                        for c in conditioning:
                            if c in data and len(data[c]) == len(a_data):
                                cond_data.append(np.array(data[c]))
                            else:
                                skip = True
                                break
                        if skip or not cond_data:
                            corr, p = stats.pearsonr(a_data, b_data)
                        else:
                            # Residualize
                            X = np.column_stack(cond_data)
                            from numpy.linalg import lstsq
                            coef_a, _, _, _ = lstsq(X, a_data, rcond=None)
                            coef_b, _, _, _ = lstsq(X, b_data, rcond=None)
                            res_a = a_data - X @ coef_a
                            res_b = b_data - X @ coef_b
                            corr, p = stats.pearsonr(res_a, res_b)
                    else:
                        corr, p = stats.pearsonr(a_data, b_data)

                    if p < 0.05:
                        violations.append({{
                            "pair": [a, b],
                            "correlation": round(float(corr), 4),
                            "p_value": round(float(p), 6),
                        }})

            consistent = len(violations) == 0
            score = 1.0 - (len(violations) / max(tests_run, 1))

            result = {{
                "consistent": consistent,
                "violations": violations,
                "tests_run": tests_run,
                "score": round(score, 4),
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)

        sandbox_result = self.sandbox.execute(
            script=script,
            requirements=["numpy", "scipy"],
            description=f"DAG fit test for graph {graph_id}",
        )

        metrics = sandbox_result.metrics or {}
        return {
            "consistent": metrics.get("consistent", False),
            "violations": metrics.get("violations", []),
            "score": metrics.get("score", 0.0),
        }

    def suggest_confounders(self, graph_id: str) -> dict[str, Any]:
        """Use LLM to suggest potential confounders for edges in the DAG."""
        graph = self.db.get_causal_graph(graph_id)
        if not graph:
            return {"error": f"Graph not found: {graph_id}"}
        if not self.inference:
            return {"error": "Inference client required for confounder suggestions"}

        edges_desc = ", ".join(f"{e[0]} -> {e[1]}" for e in graph["edges"] if len(e) >= 2)
        existing = ", ".join(graph.get("confounders", [])) or "none"

        response = self.inference.chat(
            messages=[{
                "role": "user",
                "content": (
                    f"I have a causal DAG for climate research with these edges: {edges_desc}\n"
                    f"Existing confounders: {existing}\n"
                    f"Nodes: {', '.join(graph['nodes'])}\n\n"
                    f"Suggest potential unmeasured confounders that could bias the causal "
                    f"estimates. For each suggestion, specify which edge it affects and why.\n\n"
                    f"Return a JSON array of objects with keys: edge (array of [src, dst]), "
                    f"confounder (string), reasoning (string)."
                ),
            }],
        )

        # Parse response
        content = response.content.strip()
        suggestions: list[dict[str, Any]] = []
        try:
            # Try to extract JSON from response
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                suggestions = json.loads(content[start:end])
        except (json.JSONDecodeError, ValueError):
            suggestions = [{"edge": [], "confounder": "parse_error", "reasoning": content[:500]}]

        return {"suggestions": suggestions}

    def get_dag(self, graph_id: str) -> dict[str, Any] | None:
        return self.db.get_causal_graph(graph_id)

    def list_dags(self) -> list[dict[str, Any]]:
        return self.db.list_causal_graphs()

    # -- Script generators -----------------------------------------------

    def _backdoor_script(self, treatment: str, outcome: str,
                         treatment_data: list, outcome_data: list,
                         confounder_data: dict[str, list]) -> str:
        return textwrap.dedent(f"""\
            import json
            import numpy as np
            from scipy import stats

            treatment = np.array({json.dumps(treatment_data)})
            outcome = np.array({json.dumps(outcome_data)})
            confounders = {json.dumps(confounder_data)}

            # Backdoor adjustment via OLS with confounders
            X_cols = [treatment]
            col_names = ["{treatment}"]
            for name, vals in confounders.items():
                if len(vals) == len(treatment):
                    X_cols.append(np.array(vals))
                    col_names.append(name)

            X = np.column_stack(X_cols)
            X_with_intercept = np.column_stack([np.ones(len(X)), X])

            # OLS
            from numpy.linalg import lstsq
            coef, residuals, rank, sv = lstsq(X_with_intercept, outcome, rcond=None)

            ate = float(coef[1])  # coefficient of treatment variable

            # Bootstrap CI
            n = len(outcome)
            n_boot = 1000
            boot_ates = []
            for _ in range(n_boot):
                idx = np.random.choice(n, n, replace=True)
                c, _, _, _ = lstsq(X_with_intercept[idx], outcome[idx], rcond=None)
                boot_ates.append(float(c[1]))

            ci_lower = float(np.percentile(boot_ates, 2.5))
            ci_upper = float(np.percentile(boot_ates, 97.5))

            # Approximate p-value
            se = float(np.std(boot_ates))
            if se > 0:
                t_stat = ate / se
                p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
            else:
                p_value = 1.0

            result = {{
                "ate": round(ate, 6),
                "ci_lower": round(ci_lower, 6),
                "ci_upper": round(ci_upper, 6),
                "p_value": round(p_value, 6),
                "method": "backdoor_adjustment",
                "n_confounders": len(confounders),
                "n_obs": n,
                "interpretation": f"ATE={{ate:.4f}} (95% CI: [{{ci_lower:.4f}}, {{ci_upper:.4f}}]), p={{p_value:.4f}}",
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)

    def _propensity_script(self, treatment: str, outcome: str,
                           treatment_data: list, outcome_data: list,
                           confounder_data: dict[str, list]) -> str:
        return textwrap.dedent(f"""\
            import json
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from scipy import stats

            treatment = np.array({json.dumps(treatment_data)})
            outcome = np.array({json.dumps(outcome_data)})
            confounders = {json.dumps(confounder_data)}

            # Build confounder matrix
            X_cols = []
            for name, vals in confounders.items():
                if len(vals) == len(treatment):
                    X_cols.append(np.array(vals))

            if not X_cols:
                X_cols = [np.ones(len(treatment))]

            X = np.column_stack(X_cols)

            # Binarize treatment for propensity score
            t_binary = (treatment > np.median(treatment)).astype(int)

            # Fit propensity score model
            ps_model = LogisticRegression(max_iter=1000)
            ps_model.fit(X, t_binary)
            ps = ps_model.predict_proba(X)[:, 1]

            # IPW estimator
            weights_treated = t_binary / np.clip(ps, 0.01, 0.99)
            weights_control = (1 - t_binary) / np.clip(1 - ps, 0.01, 0.99)

            ate = float(np.mean(weights_treated * outcome) - np.mean(weights_control * outcome))

            # Bootstrap CI
            n = len(outcome)
            n_boot = 1000
            boot_ates = []
            for _ in range(n_boot):
                idx = np.random.choice(n, n, replace=True)
                wt = t_binary[idx] / np.clip(ps[idx], 0.01, 0.99)
                wc = (1 - t_binary[idx]) / np.clip(1 - ps[idx], 0.01, 0.99)
                boot_ates.append(float(np.mean(wt * outcome[idx]) - np.mean(wc * outcome[idx])))

            ci_lower = float(np.percentile(boot_ates, 2.5))
            ci_upper = float(np.percentile(boot_ates, 97.5))

            se = float(np.std(boot_ates))
            p_value = float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else 1.0

            result = {{
                "ate": round(ate, 6),
                "ci_lower": round(ci_lower, 6),
                "ci_upper": round(ci_upper, 6),
                "p_value": round(p_value, 6),
                "method": "propensity_score_ipw",
                "n_obs": n,
                "interpretation": f"ATE={{ate:.4f}} (95% CI: [{{ci_lower:.4f}}, {{ci_upper:.4f}}]), p={{p_value:.4f}}",
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)

    def _iv_script(self, treatment: str, outcome: str,
                   treatment_data: list, outcome_data: list,
                   instrument_data: list) -> str:
        return textwrap.dedent(f"""\
            import json
            import numpy as np
            from scipy import stats
            from numpy.linalg import lstsq

            treatment = np.array({json.dumps(treatment_data)})
            outcome = np.array({json.dumps(outcome_data)})
            instrument = np.array({json.dumps(instrument_data)})

            n = len(outcome)

            # Two-stage least squares (2SLS)
            # Stage 1: treatment = a + b*instrument
            Z = np.column_stack([np.ones(n), instrument])
            coef1, _, _, _ = lstsq(Z, treatment, rcond=None)
            treatment_hat = Z @ coef1

            # Stage 2: outcome = c + d*treatment_hat
            X2 = np.column_stack([np.ones(n), treatment_hat])
            coef2, _, _, _ = lstsq(X2, outcome, rcond=None)

            ate = float(coef2[1])

            # Bootstrap CI
            n_boot = 1000
            boot_ates = []
            for _ in range(n_boot):
                idx = np.random.choice(n, n, replace=True)
                c1, _, _, _ = lstsq(Z[idx], treatment[idx], rcond=None)
                t_hat = Z[idx] @ c1
                X2b = np.column_stack([np.ones(len(idx)), t_hat])
                c2, _, _, _ = lstsq(X2b, outcome[idx], rcond=None)
                boot_ates.append(float(c2[1]))

            ci_lower = float(np.percentile(boot_ates, 2.5))
            ci_upper = float(np.percentile(boot_ates, 97.5))

            se = float(np.std(boot_ates))
            p_value = float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else 1.0

            # First-stage F-statistic
            rss_reduced = float(np.sum((treatment - np.mean(treatment))**2))
            rss_full = float(np.sum((treatment - treatment_hat)**2))
            f_stat = ((rss_reduced - rss_full) / 1) / (rss_full / (n - 2))

            result = {{
                "ate": round(ate, 6),
                "ci_lower": round(ci_lower, 6),
                "ci_upper": round(ci_upper, 6),
                "p_value": round(p_value, 6),
                "method": "instrumental_variable_2sls",
                "first_stage_f": round(float(f_stat), 4),
                "n_obs": n,
                "interpretation": f"ATE={{ate:.4f}} (95% CI: [{{ci_lower:.4f}}, {{ci_upper:.4f}}]), p={{p_value:.4f}}, F={{f_stat:.1f}}",
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)
