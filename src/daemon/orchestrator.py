"""Swarm Orchestrator — coordinates specialist agents in a discovery loop.

The orchestrator runs a continuous cycle:
    1. Experimenter runs experiments → produces metrics + residuals
    2. Anomaly Analyzer finds systematic failures / surprises
    3. Theorist generates hypotheses from anomalies
    4. Novelty Checker filters hypotheses
    5. Experimenter tests promising hypotheses
    6. Reviewer debates & validates findings
    7. Knowledge base updated → loop repeats

The orchestrator itself is NOT an LLM — it is a deterministic state machine
that routes messages between specialist children and the parent daemon.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ulid import ULID

log = logging.getLogger("daemon.orchestrator")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Phase(str, Enum):
    """Research cycle phases."""
    IDLE = "idle"
    LITERATURE_SCAN = "literature_scan"
    EXPERIMENT = "experiment"
    ANOMALY_ANALYSIS = "anomaly_analysis"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    NOVELTY_CHECK = "novelty_check"
    HYPOTHESIS_TEST = "hypothesis_test"
    REVIEW = "review"
    SYNTHESIS = "synthesis"
    FIGURE_GENERATION = "figure_generation"
    PAPER_DRAFT = "paper_draft"
    PAPER_REVIEW = "paper_review"
    PAPER_REVISION = "paper_revision"
    PAPER_FINALIZE = "paper_finalize"


@dataclass
class ResearchTask:
    """A unit of work assigned to a specialist."""
    id: str
    phase: Phase
    role: str  # experimenter | theorist | reviewer | data_curator
    description: str
    context: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | running | done | failed
    result: dict[str, Any] | None = None
    created_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase.value,
            "role": self.role,
            "description": self.description,
            "context": self.context,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class CycleState:
    """Persistent state for the current research cycle."""
    cycle_id: str
    phase: Phase = Phase.IDLE
    tasks: list[ResearchTask] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    validated_findings: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "phase": self.phase.value,
            "tasks": [t.to_dict() for t in self.tasks],
            "anomalies": self.anomalies,
            "hypotheses": self.hypotheses,
            "validated_findings": self.validated_findings,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class SwarmOrchestrator:
    """Deterministic state machine that coordinates the agent swarm.

    The orchestrator does NOT call LLMs directly.  It dispatches tasks to
    specialist modules (which may internally call LLMs) and routes their
    outputs to the next phase.
    """

    def __init__(
        self,
        config: Any,
        db: Any,
        inference: Any,
        tool_executor: Any,
        *,
        anomaly_analyzer: Any | None = None,
        idea_generator: Any | None = None,
        novelty_checker: Any | None = None,
        debate_engine: Any | None = None,
        on_phase_change: Callable[[Phase], None] | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.inference = inference
        self.tool_executor = tool_executor
        self._on_phase_change = on_phase_change

        # Lazy-import specialist modules so orchestrator can be used standalone
        from .anomaly_analyzer import AnomalyAnalyzer
        from .idea_generator import IdeaGenerator
        from .novelty import NoveltyChecker
        from .debate import DebateEngine
        from .paper_writer import PaperWriter

        self.anomaly_analyzer = anomaly_analyzer or AnomalyAnalyzer(
            config=config, db=db, inference=inference,
        )
        self.idea_generator = idea_generator or IdeaGenerator(
            inference=inference, db=db, config=config,
        )
        self.novelty_checker = novelty_checker or NoveltyChecker(
            inference=inference, db=db, config=config,
        )
        self.debate_engine = debate_engine or DebateEngine(
            inference=inference, db=db, config=config,
        )
        self.paper_writer = PaperWriter(
            inference=inference, db=db, config=config,
        )

        self._cycle: CycleState | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        research_goal: str,
        *,
        max_hypotheses: int = 3,
        skip_literature: bool = False,
        experiment_context: dict[str, Any] | None = None,
        generate_paper: bool = False,
        paper_title: str | None = None,
        max_revisions: int = 2,
    ) -> dict[str, Any]:
        """Execute one full research cycle and return a summary.

        This is the main entry point.  It drives the swarm through all phases
        and returns when synthesis is complete (or on error).

        If generate_paper=True, the cycle continues with:
            figure generation → paper draft → peer review → revision → finalize
        """
        cycle_id = str(ULID())
        self._cycle = CycleState(
            cycle_id=cycle_id,
            started_at=_utc(),
        )
        self._persist_cycle()

        log.info("[ORCHESTRATOR] Starting cycle %s — goal: %s", cycle_id[:8], research_goal[:120])

        try:
            # Phase 1: Literature scan (optional)
            if not skip_literature:
                self._transition(Phase.LITERATURE_SCAN)
                lit_context = self._phase_literature(research_goal)
            else:
                lit_context = {}

            # Phase 2: Run baseline / current experiments
            self._transition(Phase.EXPERIMENT)
            experiment_results = self._phase_experiment(
                research_goal, experiment_context or {}, lit_context,
            )

            # Phase 3: Analyze anomalies in experiment results
            self._transition(Phase.ANOMALY_ANALYSIS)
            anomalies = self._phase_anomaly_analysis(experiment_results)
            self._cycle.anomalies = anomalies

            # Phase 4: Generate hypotheses from anomalies
            self._transition(Phase.HYPOTHESIS_GENERATION)
            hypotheses = self._phase_hypothesis_generation(
                research_goal, anomalies, max_ideas=max_hypotheses,
            )
            self._cycle.hypotheses = hypotheses

            # Phase 5: Novelty check — filter out well-trodden ideas
            self._transition(Phase.NOVELTY_CHECK)
            novel_hypotheses = self._phase_novelty_check(hypotheses)

            # Phase 6: Test promising hypotheses
            self._transition(Phase.HYPOTHESIS_TEST)
            test_results = self._phase_hypothesis_test(
                research_goal, novel_hypotheses, experiment_context or {},
            )

            # Phase 7: Adversarial review of findings
            self._transition(Phase.REVIEW)
            validated = self._phase_review(test_results)
            self._cycle.validated_findings = validated

            # Phase 8: Synthesis — persist knowledge, update paper
            self._transition(Phase.SYNTHESIS)
            summary = self._phase_synthesis(research_goal, validated)

            # Paper generation pipeline (optional)
            # Generate paper if we have any content: validated findings, anomalies, or hypotheses
            paper_result = {}
            has_content = validated or anomalies or novel_hypotheses or hypotheses
            if generate_paper and has_content:
                paper_result = self._run_paper_pipeline(
                    research_goal=research_goal,
                    validated_findings=validated,
                    anomalies=anomalies,
                    experiment_results=experiment_results,
                    paper_title=paper_title,
                    max_revisions=max_revisions,
                )

            self._cycle.completed_at = _utc()
            self._persist_cycle()

            log.info(
                "[ORCHESTRATOR] Cycle %s complete — %d findings validated",
                cycle_id[:8], len(validated),
            )
            result = {
                "ok": True,
                "cycle_id": cycle_id,
                "anomalies_found": len(anomalies),
                "hypotheses_generated": len(hypotheses),
                "hypotheses_novel": len(novel_hypotheses),
                "findings_validated": len(validated),
                "summary": summary,
            }
            if paper_result:
                result["paper"] = paper_result
            return result

        except Exception as exc:
            log.error("[ORCHESTRATOR] Cycle %s failed: %s", cycle_id[:8], exc)
            self._cycle.completed_at = _utc()
            self._persist_cycle()
            return {
                "ok": False,
                "cycle_id": cycle_id,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_literature(self, goal: str) -> dict[str, Any]:
        """Scan recent literature for relevant papers."""
        log.info("[LIT] Scanning literature for: %s", goal[:80])

        # Use the tool executor to search papers
        result = self.tool_executor.execute("web_search_papers", {
            "query": goal,
            "max_results": 5,
        })
        papers = []
        if isinstance(result, dict) and result.get("result"):
            try:
                papers = json.loads(result["result"])
                if isinstance(papers, dict):
                    papers = papers.get("papers", [])
            except (json.JSONDecodeError, TypeError):
                pass

        log.info("[LIT] Found %d papers", len(papers))
        return {"papers": papers}

    def _phase_experiment(
        self,
        goal: str,
        experiment_context: dict[str, Any],
        lit_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Run or gather current experiment results.

        If experiment_context contains 'predictions_path' and 'truth_path',
        evaluate those.  Otherwise, look for the most recent experiment
        outputs in the data directory.
        """
        log.info("[EXP] Running experiments...")

        pred_path = experiment_context.get("predictions_path")
        truth_path = experiment_context.get("truth_path")

        if not pred_path:
            # Try to find existing predictions
            candidates = [
                self.config.repo_root / "sim_outputs" / "predictions_200.jsonl",
                self.config.repo_root / "sim_outputs" / "predictions.jsonl",
                self.config.repo_root / "sim_outputs_agent" / "predictions.jsonl",
            ]
            for c in candidates:
                if c.exists():
                    pred_path = str(c)
                    truth_path = str(c.parent / "truth.jsonl")
                    break

        if not pred_path or not truth_path:
            log.warning("[EXP] No predictions found — returning empty results.")
            return {"metrics": {}, "per_sample": []}

        # Run evaluate_jsonl directly via subprocess for reliable JSON parsing
        import subprocess
        eval_cmd = [
            "python3", "src/evaluate_jsonl.py",
            "--predictions", pred_path,
            "--truth", truth_path,
            "--per-sample-json",
        ]
        try:
            proc = subprocess.run(
                eval_cmd,
                capture_output=True, text=True, timeout=60,
                cwd=str(self.config.repo_root),
            )
            if proc.returncode == 0 and proc.stdout.strip():
                parsed = json.loads(proc.stdout)
                metrics = parsed.get("summary", {})
                per_sample = parsed.get("per_sample", [])
                log.info(
                    "[EXP] Evaluated %d samples — %d per-sample records",
                    metrics.get("samples", 0), len(per_sample),
                )
                return {
                    "metrics": metrics,
                    "per_sample": per_sample,
                    "predictions_path": pred_path,
                    "truth_path": truth_path,
                }
            else:
                log.warning("[EXP] evaluate_jsonl failed: %s", proc.stderr[:300])
        except Exception as exc:
            log.warning("[EXP] Evaluation error: %s", exc)

        return {
            "metrics": {},
            "per_sample": [],
            "predictions_path": pred_path,
            "truth_path": truth_path,
        }

    def _phase_anomaly_analysis(self, experiment_results: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze experiment results for anomalies and systematic failures."""
        log.info("[ANOMALY] Analyzing experiment results...")
        return self.anomaly_analyzer.analyze(experiment_results)

    def _phase_hypothesis_generation(
        self,
        goal: str,
        anomalies: list[dict[str, Any]],
        *,
        max_ideas: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate hypotheses from anomalies."""
        log.info("[HYPOTHESIZE] Generating hypotheses from %d anomalies...", len(anomalies))

        if not anomalies:
            log.info("[HYPOTHESIZE] No anomalies — using general idea generation.")
            result = self.idea_generator.generate(goal, max_ideas=max_ideas)
            return result.get("ideas", [])

        # Build a constraint string from anomalies
        anomaly_summary = "\n".join(
            f"- [{a.get('type', 'unknown')}] {a.get('description', '')}"
            for a in anomalies[:10]
        )
        constraint = (
            f"The following anomalies were found in recent experiments:\n{anomaly_summary}\n\n"
            f"Generate hypotheses that EXPLAIN these anomalies and suggest "
            f"concrete experiments to test them."
        )

        result = self.idea_generator.generate(
            goal, max_ideas=max_ideas, constraint=constraint,
        )
        hypotheses = result.get("ideas", [])

        # Record each hypothesis in the DB
        for h in hypotheses:
            try:
                self.tool_executor.execute("propose_hypothesis", {
                    "content": f"{h.get('title', 'Untitled')}: {h.get('description', '')}",
                    "source": f"anomaly-driven (cycle {self._cycle.cycle_id[:8]})",
                })
            except Exception:
                pass

        return hypotheses

    def _phase_novelty_check(self, hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter hypotheses through the novelty checker."""
        if not hypotheses:
            return []

        novel = []
        for h in hypotheses:
            title = h.get("title", "")
            desc = h.get("description", "")
            idea = f"{title}: {desc}"

            log.info("[NOVELTY] Checking: %s", title[:60])
            try:
                verdict = self.novelty_checker.check(idea, context=desc)
                score = verdict.novelty_score if hasattr(verdict, "novelty_score") else 0.0
                # Use judge's verdict but accept borderline ideas (>0.3) for exploration
                is_novel = score >= 0.3

                h["novelty_score"] = score
                h["novelty_verdict"] = "novel" if is_novel else "not_novel"

                if is_novel:
                    novel.append(h)
                    log.info("[NOVELTY] ✓ NOVEL (%.2f): %s", score, title[:60])
                else:
                    log.info("[NOVELTY] ✗ Not novel (%.2f): %s", score, title[:60])
            except Exception as exc:
                log.warning("[NOVELTY] Check failed for '%s': %s — keeping it", title[:40], exc)
                h["novelty_score"] = -1
                h["novelty_verdict"] = "check_failed"
                novel.append(h)  # keep on error

        log.info("[NOVELTY] %d/%d hypotheses passed novelty check", len(novel), len(hypotheses))
        return novel

    def _phase_hypothesis_test(
        self,
        goal: str,
        hypotheses: list[dict[str, Any]],
        experiment_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Design and run experiments for each novel hypothesis.

        For Phase 1, we use the inference client to generate a test plan
        and execute it via sandbox.
        """
        if not hypotheses:
            return []

        results = []
        for h in hypotheses:
            title = h.get("title", "Untitled")
            desc = h.get("description", "")
            log.info("[TEST] Testing hypothesis: %s", title[:60])

            # Ask LLM to design an experiment
            experiment_prompt = (
                f"Research goal: {goal}\n\n"
                f"Hypothesis to test: {title}\n{desc}\n\n"
                f"Available data: predictions JSONL, truth JSONL, HURDAT2 samples, sim_outputs.\n"
                f"Available tools: Python scripts in src/, sandbox execution.\n\n"
                f"Design a MINIMAL experiment (one Python script) that tests this hypothesis. "
                f"The script should:\n"
                f"1. Load relevant data\n"
                f"2. Compute a specific metric that would support or refute the hypothesis\n"
                f"3. Write results.json with clear evidence\n\n"
                f"Return ONLY the Python script, no explanation."
            )

            try:
                response = self.inference.chat([
                    {"role": "system", "content": "You are an experiment designer. Return only executable Python code."},
                    {"role": "user", "content": experiment_prompt},
                ])

                script = _extract_code(response.content or "")
                if script:
                    # Run in sandbox
                    exec_result = self.tool_executor.execute("sandbox_run", {
                        "script": script,
                        "description": f"Test: {title[:60]}",
                        "timeout_s": 120,
                    })

                    result_data = {}
                    if isinstance(exec_result, dict):
                        try:
                            result_data = json.loads(exec_result.get("result", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            result_data = {"raw": exec_result.get("result", "")}

                    results.append({
                        "hypothesis": h,
                        "experiment_script": script[:500],
                        "result": result_data,
                        "status": "completed",
                    })
                else:
                    results.append({
                        "hypothesis": h,
                        "status": "failed",
                        "error": "Could not extract executable script from LLM response",
                    })

            except Exception as exc:
                log.warning("[TEST] Failed to test '%s': %s", title[:40], exc)
                results.append({
                    "hypothesis": h,
                    "status": "failed",
                    "error": str(exc),
                })

        return results

    def _phase_review(self, test_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Adversarial review of experimental findings via debate."""
        if not test_results:
            return []

        validated = []
        for tr in test_results:
            if tr.get("status") != "completed":
                continue

            h = tr.get("hypothesis", {})
            title = h.get("title", "Untitled")
            result = tr.get("result", {})

            finding = (
                f"Hypothesis: {title}\n"
                f"Description: {h.get('description', '')}\n"
                f"Experimental result: {json.dumps(result, indent=2)[:1000]}"
            )

            log.info("[REVIEW] Debating: %s", title[:60])
            try:
                debate_result = self.debate_engine.start_debate(
                    topic=finding,
                    debate_type="verification",
                    max_rounds=2,  # keep it fast for phase 1
                )

                verdict = debate_result.get("verdict", "inconclusive")
                confidence = debate_result.get("confidence", 0.0)

                tr["debate_verdict"] = verdict
                tr["debate_confidence"] = confidence

                if verdict == "valid" and confidence >= 0.5:
                    validated.append(tr)
                    log.info("[REVIEW] ✓ VALIDATED (%.2f): %s", confidence, title[:60])
                else:
                    log.info("[REVIEW] ✗ %s (%.2f): %s", verdict, confidence, title[:60])

            except Exception as exc:
                log.warning("[REVIEW] Debate failed for '%s': %s — keeping it", title[:40], exc)
                tr["debate_verdict"] = "debate_failed"
                validated.append(tr)  # keep on error

        return validated

    def _phase_synthesis(
        self,
        goal: str,
        validated_findings: list[dict[str, Any]],
    ) -> str:
        """Persist validated findings and anomalies to knowledge base and generate summary."""

        # Always persist anomalies as observations (even without validated findings)
        if self._cycle and self._cycle.anomalies:
            for a in self._cycle.anomalies:
                try:
                    self.tool_executor.execute("save_knowledge", {
                        "topic": f"Anomaly: {a.get('type', 'unknown')}",
                        "content": (
                            f"{a.get('description', '')}\n\n"
                            f"Suggested investigation: {a.get('suggested_investigation', '')}\n"
                            f"Physical explanation: {a.get('physical_explanation', '')}"
                        ),
                        "confidence": a.get("severity", 0.5),
                        "entry_type": "observation",
                        "source": f"anomaly_analyzer (cycle {self._cycle.cycle_id[:8]})",
                    })
                except Exception:
                    pass

        if not validated_findings:
            n_anomalies = len(self._cycle.anomalies) if self._cycle else 0
            return f"No validated findings in this cycle. {n_anomalies} anomalies persisted as observations."

        summaries = []
        for vf in validated_findings:
            h = vf.get("hypothesis", {})
            title = h.get("title", "Untitled")
            result = vf.get("result", {})

            # Persist to knowledge base
            try:
                self.tool_executor.execute("save_knowledge", {
                    "topic": title,
                    "content": json.dumps({
                        "hypothesis": h.get("description", ""),
                        "evidence": result,
                        "debate_verdict": vf.get("debate_verdict", ""),
                        "debate_confidence": vf.get("debate_confidence", 0),
                        "novelty_score": h.get("novelty_score", -1),
                    }),
                    "confidence": vf.get("debate_confidence", 0.5),
                    "entry_type": "finding",
                    "source": f"swarm_cycle_{self._cycle.cycle_id[:8]}",
                })
            except Exception:
                pass

            summaries.append(
                f"• {title} — verdict: {vf.get('debate_verdict', '?')}, "
                f"confidence: {vf.get('debate_confidence', 0):.2f}"
            )

        summary = (
            f"Research cycle {self._cycle.cycle_id[:8]} complete.\n"
            f"Goal: {goal}\n"
            f"Anomalies found: {len(self._cycle.anomalies)}\n"
            f"Hypotheses generated: {len(self._cycle.hypotheses)}\n"
            f"Validated findings:\n" + "\n".join(summaries)
        )

        log.info("[SYNTHESIS]\n%s", summary)
        return summary

    # ------------------------------------------------------------------
    # Paper generation pipeline
    # ------------------------------------------------------------------

    def _run_paper_pipeline(
        self,
        research_goal: str,
        validated_findings: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        experiment_results: dict[str, Any],
        paper_title: str | None = None,
        max_revisions: int = 2,
    ) -> dict[str, Any]:
        """Run the full paper generation pipeline: figures → draft → review → revise → finalize."""

        # Phase 9: Generate figures from experiment data
        self._transition(Phase.FIGURE_GENERATION)
        figures = self._phase_figure_generation(
            experiment_results, anomalies, validated_findings,
        )

        # Phase 10: Draft paper
        self._transition(Phase.PAPER_DRAFT)
        draft_result = self._phase_paper_draft(paper_title)
        draft_id = draft_result.get("draft_id")

        if not draft_id:
            log.error("[PAPER] Failed to generate draft")
            return {"ok": False, "error": "Draft generation failed"}

        # Phase 11-12: Review and revise loop
        final_review = None
        for revision_round in range(1, max_revisions + 1):
            # Review
            self._transition(Phase.PAPER_REVIEW)
            review_result = self._phase_paper_review(draft_id)

            decision = review_result.get("decision", "revise")
            score = review_result.get("overall_score", 0)
            final_review = review_result

            log.info(
                "[PAPER] Review round %d: score=%.1f, decision=%s",
                revision_round, score, decision,
            )

            # If accepted or score high enough, stop revising
            if decision == "accept" or score >= 8.0:
                log.info("[PAPER] Paper accepted after %d revision(s)", revision_round - 1)
                break

            if decision == "reject" and score < 3.0:
                log.warning("[PAPER] Paper rejected — major rewrite needed")
                # Still try one revision
                pass

            # Revise
            self._transition(Phase.PAPER_REVISION)
            revision_result = self._phase_paper_revision(
                draft_id, review_result.get("review_id"),
            )
            log.info(
                "[PAPER] Revised to v%d: %s",
                revision_result.get("version", "?"),
                revision_result.get("changes_summary", "")[:100],
            )

        # Phase 13: Finalize — export to files
        self._transition(Phase.PAPER_FINALIZE)
        final_result = self._phase_paper_finalize(draft_id, figures)

        return {
            "ok": True,
            "draft_id": draft_id,
            "title": draft_result.get("title", ""),
            "final_score": final_review.get("overall_score", 0) if final_review else 0,
            "final_decision": final_review.get("decision", "unknown") if final_review else "unknown",
            "figures_generated": len(figures),
            "output_path": final_result.get("output_path", ""),
        }

    def _phase_figure_generation(
        self,
        experiment_results: dict[str, Any],
        anomalies: list[dict[str, Any]],
        findings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate publication-quality figures from experiment data."""
        log.info("[FIGURES] Generating figures...")

        figures = []
        metrics = experiment_results.get("metrics", {})
        per_sample = experiment_results.get("per_sample", [])

        # Figure 1: Error distribution by lead time
        if per_sample:
            script = self._build_figure_script(
                "error_distribution",
                "Error Distribution by Lead Time",
                per_sample, metrics,
            )
            if script:
                fig_result = self._run_figure_script(script, "fig_error_dist.png")
                if fig_result:
                    figures.append(fig_result)

        # Figure 2: RI vs non-RI comparison (if anomaly detected)
        ri_anomaly = next((a for a in anomalies if a.get("type") == "RI_BLIND_SPOT"), None)
        if ri_anomaly:
            script = self._build_figure_script(
                "ri_comparison",
                "Rapid Intensification Detection Performance",
                per_sample, ri_anomaly.get("evidence", {}),
            )
            if script:
                fig_result = self._run_figure_script(script, "fig_ri_comparison.png")
                if fig_result:
                    figures.append(fig_result)

        # Figure 3: Regime analysis (if detected)
        regime_anomaly = next((a for a in anomalies if a.get("type") == "REGIME_SHIFT"), None)
        if regime_anomaly:
            script = self._build_figure_script(
                "regime_analysis",
                "Performance by Physical Regime",
                per_sample, regime_anomaly.get("evidence", {}),
            )
            if script:
                fig_result = self._run_figure_script(script, "fig_regime.png")
                if fig_result:
                    figures.append(fig_result)

        log.info("[FIGURES] Generated %d figures", len(figures))
        return figures

    def _build_figure_script(
        self,
        fig_type: str,
        title: str,
        per_sample: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str:
        """Ask LLM to generate a matplotlib figure script."""
        prompt = (
            f"Generate a Python matplotlib script that creates a publication-quality figure.\n\n"
            f"Figure type: {fig_type}\n"
            f"Title: {title}\n"
            f"Data format: list of dicts, each with keys like track_error_24h, wind_error_24h, "
            f"initial_wind, lat, actual_dwind_24h, etc.\n"
            f"Sample data (first 3): {json.dumps(per_sample[:3], indent=2)[:1500]}\n"
            f"Context: {json.dumps(context, indent=2)[:1000]}\n\n"
            f"Requirements:\n"
            f"- Use matplotlib with a clean style (plt.style.use('seaborn-v0_8-paper') or similar)\n"
            f"- High DPI (300), tight layout\n"
            f"- Clear axis labels, legend, title\n"
            f"- Save to 'output.png'\n"
            f"- The data variable is called `data` and is already loaded as a list of dicts\n"
            f"- Add: import json; data = json.load(open('input_data.json'))\n"
            f"- Return ONLY the Python script, no explanation"
        )

        try:
            response = self.inference.chat([
                {"role": "system", "content": "You are a scientific visualization expert. Return only executable Python code."},
                {"role": "user", "content": prompt},
            ])
            return _extract_code(response.content or "")
        except Exception as exc:
            log.warning("[FIGURES] Failed to generate script for %s: %s", fig_type, exc)
            return ""

    def _run_figure_script(self, script: str, filename: str) -> dict[str, Any] | None:
        """Execute a figure generation script in sandbox."""
        try:
            result = self.tool_executor.execute("sandbox_run", {
                "script": script,
                "description": f"Generate figure: {filename}",
                "timeout_s": 60,
                "requirements": ["matplotlib", "numpy"],
            })
            if isinstance(result, dict) and not result.get("error"):
                return {
                    "filename": filename,
                    "script": script[:200],
                    "status": "generated",
                }
        except Exception as exc:
            log.warning("[FIGURES] Script execution failed for %s: %s", filename, exc)
        return None

    def _phase_paper_draft(self, title: str | None = None) -> dict[str, Any]:
        """Draft a full paper from the knowledge base."""
        log.info("[PAPER] Drafting paper...")
        try:
            return self.paper_writer.draft_paper(title=title)
        except Exception as exc:
            log.error("[PAPER] Draft failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def _phase_paper_review(self, draft_id: str) -> dict[str, Any]:
        """Run peer review on the draft."""
        log.info("[PAPER] Reviewing draft %s...", draft_id[:8])
        try:
            return self.paper_writer.review_paper(draft_id)
        except Exception as exc:
            log.error("[PAPER] Review failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def _phase_paper_revision(self, draft_id: str, review_id: str | None) -> dict[str, Any]:
        """Revise the paper based on review feedback."""
        log.info("[PAPER] Revising draft %s...", draft_id[:8])
        try:
            return self.paper_writer.revise_paper(draft_id, review_id)
        except Exception as exc:
            log.error("[PAPER] Revision failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def _phase_paper_finalize(
        self,
        draft_id: str,
        figures: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Export the final paper to markdown and optionally LaTeX."""
        log.info("[PAPER] Finalizing paper...")

        try:
            draft = self.db.get_paper_draft(draft_id) if hasattr(self.db, 'get_paper_draft') else None
        except Exception:
            draft = None

        if not draft:
            return {"ok": False, "error": "Could not retrieve final draft"}

        content = draft.get("content", "")
        title = draft.get("title", "Untitled")

        # Determine output directory
        cycle_id = self._cycle.cycle_id if self._cycle else "unknown"
        out_dir = self.config.repo_root / "runs" / f"paper_{cycle_id[:8]}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write markdown
        md_path = out_dir / "paper.md"
        md_path.write_text(content, encoding="utf-8")

        # Write figure list
        if figures:
            fig_manifest = out_dir / "figures.json"
            fig_manifest.write_text(
                json.dumps(figures, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Generate LaTeX version
        latex_content = self._markdown_to_latex(title, content)
        pdf_path = None
        if latex_content:
            tex_path = out_dir / "paper.tex"
            tex_path.write_text(latex_content, encoding="utf-8")

            # Compile LaTeX to PDF
            pdf_path = self._compile_latex(tex_path)

        log.info("[PAPER] Final paper written to %s", out_dir)

        result = {
            "ok": True,
            "output_path": str(out_dir),
            "markdown_path": str(md_path),
            "title": title,
            "version": draft.get("version", 1),
        }
        if pdf_path:
            result["pdf_path"] = str(pdf_path)
        return result

    def _markdown_to_latex(self, title: str, content: str) -> str:
        """Convert paper markdown to LaTeX format via LLM."""
        try:
            response = self.inference.chat([
                {
                    "role": "system",
                    "content": (
                        "You are a LaTeX expert. Convert the given markdown paper to a clean LaTeX document "
                        "using the article class. Use standard packages (amsmath, graphicx, natbib, hyperref). "
                        "Output ONLY the LaTeX source, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Convert this paper to LaTeX:\n\n"
                        f"Title: {title}\n\n"
                        f"{content[:8000]}"
                    ),
                },
            ])
            latex = response.content or ""
            # Extract from code fences if present
            if "\\begin{document}" in latex:
                return latex.strip()
            code = _extract_code(latex)
            if code and "\\begin{document}" in code:
                return code
            return latex.strip()
        except Exception as exc:
            log.warning("[PAPER] LaTeX conversion failed: %s", exc)
            return ""

    def _compile_latex(self, tex_path: Path) -> Path | None:
        """Compile LaTeX to PDF using pdflatex (2 passes for references)."""
        import subprocess

        out_dir = tex_path.parent
        log.info("[PAPER] Compiling LaTeX → PDF...")

        for pass_num in (1, 2):
            try:
                proc = subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-output-directory", str(out_dir),
                        str(tex_path),
                    ],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(out_dir),
                )
                if proc.returncode != 0 and pass_num == 2:
                    log.warning("[PAPER] pdflatex pass %d warnings: %s", pass_num, proc.stdout[-500:] if proc.stdout else "")
            except FileNotFoundError:
                log.warning("[PAPER] pdflatex not found — skipping PDF generation")
                return None
            except subprocess.TimeoutExpired:
                log.warning("[PAPER] pdflatex timed out on pass %d", pass_num)
                return None
            except Exception as exc:
                log.warning("[PAPER] pdflatex error: %s", exc)
                return None

        pdf_path = out_dir / "paper.pdf"
        if pdf_path.exists():
            log.info("[PAPER] PDF generated: %s (%.1f KB)", pdf_path, pdf_path.stat().st_size / 1024)
            return pdf_path
        else:
            log.warning("[PAPER] PDF not found after compilation")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transition(self, phase: Phase) -> None:
        """Transition to a new phase."""
        if self._cycle:
            self._cycle.phase = phase
            self._persist_cycle()
        if self._on_phase_change:
            self._on_phase_change(phase)
        log.info("[PHASE] → %s", phase.value)

    def _persist_cycle(self) -> None:
        """Save cycle state to DB."""
        if not self._cycle:
            return
        try:
            self.db.set_kv(
                f"swarm_cycle_{self._cycle.cycle_id}",
                json.dumps(self._cycle.to_dict()),
            )
            self.db.set_kv("swarm_latest_cycle", self._cycle.cycle_id)
        except Exception as exc:
            log.warning("[PERSIST] Failed to save cycle state: %s", exc)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response (handles markdown fences)."""
    import re
    # Try ```python ... ``` blocks
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[0].strip()
    # If no fences, assume the whole thing is code
    if text.strip().startswith(("import ", "from ", "#!", "def ", "class ")):
        return text.strip()
    return ""
