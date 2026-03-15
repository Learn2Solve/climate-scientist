"""Dynamic tool learning - discover, test, and catalog Python packages."""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.tool_learning")


class ToolLearner:
    def __init__(self, config: Any, db: Any, sandbox_runner: Any,
                 inference: Any | None = None) -> None:
        self.config = config
        self.db = db
        self.sandbox = sandbox_runner
        self.inference = inference

    def discover_package(self, package_name: str, purpose: str = "") -> dict[str, Any]:
        """Install and introspect a Python package in sandbox."""
        if not self.config.tool_learning_enabled:
            return {"error": "Tool learning is disabled in config"}

        # Check if already known
        existing = self.db.get_learned_tool_by_package(package_name)
        if existing:
            return {
                "tool_id": existing["id"],
                "package": package_name,
                "capabilities": existing["capabilities"],
                "example_code": existing["example_code"],
                "note": "Already discovered",
            }

        # Check max learned tools limit
        all_tools = self.db.list_learned_tools()
        if len(all_tools) >= self.config.max_learned_tools:
            return {"error": f"Max learned tools ({self.config.max_learned_tools}) reached"}

        script = textwrap.dedent(f"""\
            import json
            import importlib
            import inspect
            import sys

            package_name = "{package_name}"

            try:
                mod = importlib.import_module(package_name)
            except ImportError:
                result = {{"error": f"Could not import {{package_name}}", "installed": False}}
                with open("results.json", "w") as f:
                    json.dump(result, f)
                print(json.dumps(result))
                sys.exit(0)

            # Introspect
            version = getattr(mod, "__version__", "unknown")
            doc = (mod.__doc__ or "")[:500]

            # Get public names
            public = [n for n in dir(mod) if not n.startswith("_")][:50]

            # Get classes and functions
            classes = []
            functions = []
            for name in public:
                obj = getattr(mod, name, None)
                if obj is None:
                    continue
                if inspect.isclass(obj):
                    methods = [m for m in dir(obj) if not m.startswith("_")][:10]
                    classes.append({{"name": name, "methods": methods, "doc": (obj.__doc__ or "")[:200]}})
                elif callable(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except (ValueError, TypeError):
                        sig = "(...)"
                    functions.append({{"name": name, "signature": sig, "doc": (obj.__doc__ or "")[:200]}})

            result = {{
                "installed": True,
                "version": version,
                "doc": doc,
                "public_names": public[:30],
                "classes": classes[:15],
                "functions": functions[:15],
            }}
            with open("results.json", "w") as f:
                json.dump(result, f)
            print(json.dumps(result, indent=2))
        """)

        sandbox_result = self.sandbox.execute(
            script=script,
            requirements=[package_name],
            description=f"Discover package: {package_name}",
        )

        metrics = sandbox_result.metrics or {}
        if metrics.get("error") or not metrics.get("installed"):
            return {
                "error": metrics.get("error", f"Failed to install {package_name}"),
                "stderr": sandbox_result.stderr[:500],
            }

        # Summarize capabilities using LLM if available
        capabilities: list[str] = []
        example_code = ""
        description = metrics.get("doc", "")[:300]

        if self.inference:
            try:
                classes_desc = json.dumps(metrics.get("classes", [])[:10], indent=1)
                funcs_desc = json.dumps(metrics.get("functions", [])[:10], indent=1)
                resp = self.inference.chat(
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Summarize the Python package '{package_name}' "
                            f"(version {metrics.get('version', '?')}).\n"
                            f"Purpose: {purpose or 'general'}\n"
                            f"Doc: {description}\n"
                            f"Classes: {classes_desc}\n"
                            f"Functions: {funcs_desc}\n\n"
                            f"Return JSON with keys: capabilities (list of 3-5 short strings), "
                            f"description (one sentence), example_code (5-10 line example)."
                        ),
                    }],
                )
                content = resp.content.strip()
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(content[start:end])
                    capabilities = parsed.get("capabilities", [])[:5]
                    description = parsed.get("description", description)
                    example_code = parsed.get("example_code", "")
            except Exception as exc:
                log.warning("LLM summarization failed: %s", exc)
                # Fall back to auto-extracted info
                capabilities = [
                    f"class: {c['name']}" for c in metrics.get("classes", [])[:3]
                ] + [
                    f"func: {f['name']}" for f in metrics.get("functions", [])[:2]
                ]

        if not capabilities:
            capabilities = [
                f"class: {c['name']}" for c in metrics.get("classes", [])[:3]
            ] + [
                f"func: {f['name']}" for f in metrics.get("functions", [])[:2]
            ]

        tool_id = str(ULID())
        self.db.insert_learned_tool({
            "id": tool_id,
            "package_name": package_name,
            "version": metrics.get("version", ""),
            "description": description,
            "capabilities": capabilities,
            "use_cases": [purpose] if purpose else [],
            "example_code": example_code,
        })

        return {
            "tool_id": tool_id,
            "package": package_name,
            "version": metrics.get("version", ""),
            "capabilities": capabilities,
            "example_code": example_code,
        }

    def test_package(self, package_name: str, test_script: str) -> dict[str, Any]:
        """Run a test script using a package to validate it works."""
        sandbox_result = self.sandbox.execute(
            script=test_script,
            requirements=[package_name],
            description=f"Test package: {package_name}",
        )

        passed = sandbox_result.returncode == 0
        self.db.update_learned_tool_stats(package_name, success=passed)

        return {
            "passed": passed,
            "stdout": sandbox_result.stdout[:2000],
            "stderr": sandbox_result.stderr[:1000],
            "metrics": sandbox_result.metrics,
        }

    def recommend_package(self, problem_description: str) -> dict[str, Any]:
        """Recommend a package from learned_tools for a given problem."""
        tools = self.db.list_learned_tools(min_success=0)
        if not tools:
            return {
                "recommended": None,
                "reasoning": "No learned tools available",
                "alternatives": [],
            }

        if self.inference:
            tools_summary = json.dumps([
                {
                    "package": t["package_name"],
                    "description": t.get("description", "")[:200],
                    "capabilities": t.get("capabilities", []),
                    "success_count": t.get("success_count", 0),
                    "failure_count": t.get("failure_count", 0),
                }
                for t in tools
            ], indent=1)

            resp = self.inference.chat(
                messages=[{
                    "role": "user",
                    "content": (
                        f"Given this problem: {problem_description}\n\n"
                        f"Which of these learned tools would be best?\n{tools_summary}\n\n"
                        f"Return JSON with: recommended (package name or null), "
                        f"reasoning (string), alternatives (list of package names)"
                    ),
                }],
            )
            content = resp.content.strip()
            try:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(content[start:end])
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: keyword matching
        desc_lower = problem_description.lower()
        scored: list[tuple[int, dict[str, Any]]] = []
        for t in tools:
            score = 0
            for cap in t.get("capabilities", []):
                if any(word in desc_lower for word in cap.lower().split()):
                    score += 1
            score += t.get("success_count", 0)
            scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1] if scored else None
        alts = [s[1]["package_name"] for s in scored[1:4]]

        return {
            "recommended": best["package_name"] if best else None,
            "reasoning": "Keyword match + success count",
            "alternatives": alts,
        }

    def list_learned_tools(self, min_success: int = 0) -> list[dict[str, Any]]:
        return self.db.list_learned_tools(min_success=min_success)

    def forget_package(self, package_name: str) -> None:
        """Remove a package from learned_tools."""
        self.db.delete_learned_tool(package_name)
