"""Tool adapter - bridges existing tool_registry.py handlers + daemon-specific tools.

Port of automaton/src/agent/tools.ts adapted for research context.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ulid import ULID


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tool definition format for inference API
# ---------------------------------------------------------------------------


def tool_def(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


# ---------------------------------------------------------------------------
# Daemon-specific tool specs (not from existing tool_registry)
# ---------------------------------------------------------------------------


DAEMON_TOOL_SPECS: list[dict[str, Any]] = [
    tool_def(
        "sleep",
        "Sleep for N seconds (max 300). The agent loop will pause.",
        {
            "type": "object",
            "properties": {"seconds": {"type": "integer", "default": 60}},
            "additionalProperties": False,
        },
    ),
    tool_def(
        "save_knowledge",
        "Persist a research finding to the knowledge base. For structured research, "
        "prefer propose_hypothesis/record_experiment/record_finding/conclude_hypothesis.",
        {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "content": {"type": "string"},
                "source": {"type": "string", "default": ""},
                "confidence": {"type": "number", "default": 0.5},
                "entry_type": {
                    "type": "string",
                    "enum": ["observation", "hypothesis", "experiment", "finding", "conclusion"],
                    "default": "observation",
                },
                "parent_id": {"type": "string", "description": "ID of parent knowledge entry"},
                "status": {
                    "type": "string",
                    "enum": ["active", "superseded", "refuted", "confirmed"],
                    "default": "active",
                },
            },
            "required": ["topic", "content"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "query_knowledge",
        "Search accumulated knowledge. Default: FTS5 keyword search. Set semantic=true for embedding-based similarity search.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "entry_type": {
                    "type": "string",
                    "enum": ["observation", "hypothesis", "experiment", "finding", "conclusion"],
                    "description": "Filter results to this entry type only",
                },
                "semantic": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use embedding-based semantic search instead of FTS5",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "propose_hypothesis",
        "Articulate a testable hypothesis before experimenting. Returns hypothesis ID for linking.",
        {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The hypothesis statement"},
                "source": {"type": "string", "default": "agent reasoning"},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "record_experiment",
        "Log an experiment linked to a hypothesis. Describes methodology and parameters.",
        {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Experiment description and methodology"},
                "hypothesis_id": {"type": "string", "description": "ID of the hypothesis being tested"},
                "source": {"type": "string", "default": "experiment"},
            },
            "required": ["content", "hypothesis_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "record_finding",
        "Log a finding from an experiment, with optional quantitative metrics.",
        {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Finding description and interpretation"},
                "experiment_id": {"type": "string", "description": "ID of the experiment that produced this finding"},
                "source": {"type": "string", "default": "analysis"},
                "metrics": {
                    "type": "object",
                    "description": "Optional quantitative metrics (e.g. {\"rmse\": 0.42, \"r2\": 0.87})",
                },
            },
            "required": ["content", "experiment_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "conclude_hypothesis",
        "Assess a hypothesis based on accumulated findings. Updates hypothesis status.",
        {
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string", "description": "ID of the hypothesis to conclude"},
                "verdict": {
                    "type": "string",
                    "enum": ["confirmed", "refuted", "inconclusive"],
                    "description": "Assessment of the hypothesis",
                },
                "reasoning": {"type": "string", "description": "Justification for the verdict"},
            },
            "required": ["hypothesis_id", "verdict", "reasoning"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "search_literature",
        "Search stored papers. Default: FTS5 keyword search. Set semantic=true for embedding-based similarity search.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms for paper content"},
                "limit": {"type": "integer", "default": 10},
                "semantic": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use embedding-based semantic search instead of FTS5",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "get_hypothesis_status",
        "Show a hypothesis and all its linked experiments, findings, and conclusion.",
        {
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string"},
            },
            "required": ["hypothesis_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "modify_heartbeat",
        "Add, update, or disable a heartbeat task.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "schedule": {"type": "string", "description": "Cron expression"},
                "task": {"type": "string"},
                "enabled": {"type": "boolean", "default": True},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "system_status",
        "Get full system status: state, costs, turns, knowledge count, children.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    tool_def(
        "list_research_cycles",
        "List past and current research cycles.",
        {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 10}},
            "additionalProperties": False,
        },
    ),
    tool_def(
        "spawn_child",
        "Create a child agent with a sub-topic and budget.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "topic": {"type": "string"},
                "budget_cents": {"type": "integer", "default": 500},
                "genesis_prompt": {"type": "string", "default": ""},
            },
            "required": ["name", "topic"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "check_child",
        "Check child agent status and read its recent knowledge.",
        {
            "type": "object",
            "properties": {"child_id": {"type": "string"}},
            "required": ["child_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "send_message",
        "Send a message to a child or peer agent.",
        {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Child ID or peer name"},
                "content": {"type": "string"},
            },
            "required": ["to", "content"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "read_inbox",
        "Read unprocessed messages from inbox.",
        {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 10}},
            "additionalProperties": False,
        },
    ),
    tool_def(
        "list_peers",
        "List known remote peers.",
        {
            "type": "object",
            "properties": {"topic": {"type": "string", "default": ""}},
            "additionalProperties": False,
        },
    ),
    tool_def(
        "register_peer",
        "Register a remote peer for federation.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "url": {"type": "string"},
                "topic": {"type": "string", "default": ""},
            },
            "required": ["name", "url"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "broadcast_finding",
        "Broadcast a knowledge entry to all peers.",
        {
            "type": "object",
            "properties": {"knowledge_id": {"type": "string"}},
            "required": ["knowledge_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "cite_paper",
        "Link a knowledge entry to a paper as a citation. Use to track which papers support which findings/hypotheses.",
        {
            "type": "object",
            "properties": {
                "knowledge_id": {"type": "string", "description": "ID of the knowledge entry"},
                "paper_id": {"type": "string", "description": "ID of the paper being cited"},
                "context": {
                    "type": "string",
                    "default": "",
                    "description": "How the paper is cited (e.g. 'supports', 'contradicts', 'methodology')",
                },
            },
            "required": ["knowledge_id", "paper_id"],
            "additionalProperties": False,
        },
    ),
    # -- Phase 3: Sandbox + Data + Plans + Novelty + Reports ----------------
    tool_def(
        "sandbox_run",
        "Execute a Python script in an isolated sandbox (Docker preferred, local fallback). "
        "Write results.json from your script for structured metric output.",
        {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "Python script source code"},
                "requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "pip packages to install (e.g. ['numpy', 'pandas'])",
                },
                "data_files": {
                    "type": "object",
                    "default": {},
                    "description": "Additional files to write to workspace: {filename: content}",
                },
                "timeout_s": {"type": "integer", "default": 120},
                "description": {"type": "string", "default": ""},
            },
            "required": ["script"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "sandbox_list_results",
        "List recent sandbox execution results.",
        {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 10}},
            "additionalProperties": False,
        },
    ),
    tool_def(
        "acquire_dataset",
        "Download and cache a dataset from a URL. Cached by URL - second call returns cached copy.",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to download from"},
                "name": {"type": "string", "description": "Short name for the dataset"},
                "format": {"type": "string", "default": "csv", "description": "File format hint"},
                "max_bytes": {"type": "integer", "description": "Max download size in bytes"},
                "description": {"type": "string", "default": ""},
            },
            "required": ["url", "name"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "search_datasets",
        "Search known climate data sources and cached datasets.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "create_research_plan",
        "Create a multi-step research plan with dependencies between steps.",
        {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string", "default": ""},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string", "default": ""},
                            "step_type": {
                                "type": "string",
                                "enum": ["literature_review", "data_acquisition", "experiment", "analysis", "writing", "custom"],
                                "default": "custom",
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "default": [],
                                "description": "Indices of steps this depends on (0-based)",
                            },
                        },
                        "required": ["title"],
                    },
                    "description": "Ordered list of plan steps",
                },
            },
            "required": ["title", "steps"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "update_plan_step",
        "Update a plan step: change status, add notes, link knowledge or execution.",
        {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
                "step_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "blocked", "skipped"],
                },
                "notes": {"type": "string"},
                "link_knowledge_id": {"type": "string", "description": "Link a knowledge entry to this step"},
                "link_execution_id": {"type": "string", "description": "Link a sandbox execution to this step"},
            },
            "required": ["plan_id", "step_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "get_plan_status",
        "Get plan progress, blockers, and suggested next action.",
        {
            "type": "object",
            "properties": {"plan_id": {"type": "string"}},
            "required": ["plan_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "list_plans",
        "List research plans with status.",
        {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "draft", "completed"],
                    "description": "Filter by status",
                },
                "limit": {"type": "integer", "default": 10},
            },
            "additionalProperties": False,
        },
    ),
    tool_def(
        "check_novelty",
        "Adversarial novelty check: advocate argues FOR, critic argues AGAINST, judge decides. "
        "Costs 3 API calls. Use for significant hypotheses before investing in experiments.",
        {
            "type": "object",
            "properties": {
                "idea": {"type": "string", "description": "The idea or hypothesis to check"},
                "context": {"type": "string", "default": "", "description": "Additional context"},
                "type": {
                    "type": "string",
                    "enum": ["hypothesis", "approach"],
                    "default": "hypothesis",
                },
            },
            "required": ["idea"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "generate_report",
        "Generate a research report from the knowledge base. Outputs markdown.",
        {
            "type": "object",
            "properties": {
                "title": {"type": "string", "default": "Research Report"},
                "plan_id": {"type": "string", "description": "Scope report to a specific plan"},
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sections to include (default: all)",
                },
                "output_path": {
                    "type": "string",
                    "default": "docs/research_report.md",
                    "description": "Output file path relative to repo root",
                },
            },
            "additionalProperties": False,
        },
    ),
    tool_def(
        "update_paper_section",
        "Update a specific section of a working paper using marker-based replacement.",
        {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": ["summary", "hypotheses", "experiments", "findings", "conclusions", "references"],
                },
                "content": {"type": "string", "description": "New content for the section"},
                "paper_path": {
                    "type": "string",
                    "default": "docs/paper.md",
                    "description": "Path to the paper file",
                },
            },
            "required": ["section", "content"],
            "additionalProperties": False,
        },
    ),
    # -- Phase 4: Close all gaps vs top-lab AI scientists -------------------
    tool_def(
        "start_experiment_loop",
        "Create an optimization loop over a parameter space. Automatically suggests next params.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "parameter_space": {
                    "type": "object",
                    "description": 'Map of param name to spec: {"lr": {"type": "float", "low": 0.001, "high": 0.1}}',
                },
                "target_metric": {"type": "string", "description": "Metric to optimize from results.json"},
                "objective": {"type": "string", "enum": ["minimize", "maximize"], "default": "minimize"},
                "description": {"type": "string", "default": ""},
                "linked_hypothesis_id": {"type": "string"},
            },
            "required": ["name", "parameter_space", "target_metric"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "run_next_trial",
        "Get suggested params and run next trial in an experiment loop.",
        {
            "type": "object",
            "properties": {
                "loop_id": {"type": "string"},
                "script_template": {"type": "string", "description": "Python script with {param_name} placeholders"},
                "requirements": {"type": "array", "items": {"type": "string"}, "default": []},
            },
            "required": ["loop_id", "script_template"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "get_loop_status",
        "Get experiment loop status: trials, best params, convergence.",
        {
            "type": "object",
            "properties": {"loop_id": {"type": "string"}},
            "required": ["loop_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "run_statistical_test",
        "Run a statistical significance test on data. Outputs p-value and pass/fail.",
        {
            "type": "object",
            "properties": {
                "test_type": {
                    "type": "string",
                    "enum": ["t_test", "mann_whitney", "chi_squared", "bootstrap_ci", "paired_t_test", "wilcoxon"],
                },
                "data": {"type": "object", "description": "Test data: {group_a: [...], group_b: [...]} or {values: [...]}"},
                "description": {"type": "string", "default": ""},
                "linked_finding_id": {"type": "string"},
                "linked_execution_id": {"type": "string"},
            },
            "required": ["test_type", "data"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "validate_experiment_rigor",
        "Check if an experiment meets statistical rigor standards.",
        {
            "type": "object",
            "properties": {"execution_id": {"type": "string"}},
            "required": ["execution_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "verify_result",
        "Re-run a sandbox execution to check reproducibility.",
        {
            "type": "object",
            "properties": {"execution_id": {"type": "string"}},
            "required": ["execution_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "run_climate_analysis",
        "Run a structured climate analysis (HURDAT2 parse, SST, RI detection, climate index).",
        {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": ["hurdat2_parse", "sst_extraction", "ri_detection", "wind_shear_calc", "climate_index"],
                },
                "data_path": {"type": "string", "description": "Path to input data file"},
                "parameters": {"type": "object", "default": {}, "description": "Analysis-specific parameters"},
            },
            "required": ["analysis_type", "data_path"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "fetch_era5",
        "Fetch ERA5 reanalysis data via CDS API (requires era5_api_key in config).",
        {
            "type": "object",
            "properties": {
                "variable": {"type": "string", "description": "ERA5 variable name (e.g. sea_surface_temperature)"},
                "year": {"type": "string"},
                "month": {"type": "string"},
                "area": {"type": "array", "items": {"type": "number"}, "description": "[N, W, S, E] bounding box"},
            },
            "required": ["variable", "year", "month"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "analyze_figure",
        "Send a generated plot/figure to a vision model for interpretation.",
        {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to image file (png, jpg)"},
                "prompt": {"type": "string", "description": "What to analyze in the figure"},
                "analysis_type": {"type": "string", "enum": ["interpret", "quality", "compare"], "default": "interpret"},
                "linked_execution_id": {"type": "string"},
            },
            "required": ["image_path"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "compare_figures",
        "Send two figures to a vision model for comparison.",
        {
            "type": "object",
            "properties": {
                "image_path_1": {"type": "string"},
                "image_path_2": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["image_path_1", "image_path_2"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "draft_paper",
        "Generate a full research paper from the knowledge base.",
        {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "linked_plan_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    ),
    tool_def(
        "review_paper",
        "Simulate peer review of a paper draft. Returns scores and feedback.",
        {
            "type": "object",
            "properties": {"draft_id": {"type": "string"}},
            "required": ["draft_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "revise_paper",
        "Revise a paper draft based on review feedback.",
        {
            "type": "object",
            "properties": {
                "draft_id": {"type": "string"},
                "review_id": {"type": "string", "description": "Specific review to address"},
            },
            "required": ["draft_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "add_plan_branch",
        "Add a conditional branch to a research plan (if condition then X else Y).",
        {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
                "from_step_id": {"type": "string", "description": "Step after which to evaluate condition"},
                "condition": {"type": "string", "description": 'e.g. "rmse < 0.5" or "hypothesis_id:confirmed"'},
                "condition_type": {"type": "string", "enum": ["metric_threshold", "hypothesis_verdict", "custom"], "default": "metric_threshold"},
                "then_steps": {"type": "array", "items": {"type": "object"}, "description": "Steps if condition true"},
                "else_steps": {"type": "array", "items": {"type": "object"}, "description": "Steps if condition false"},
            },
            "required": ["plan_id", "from_step_id", "condition"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "evaluate_branch",
        "Evaluate a plan branch condition and create the appropriate steps.",
        {
            "type": "object",
            "properties": {"branch_id": {"type": "string"}},
            "required": ["branch_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "start_debate",
        "Start an adversarial debate to verify a finding. Two models argue, judge decides.",
        {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The finding or claim to debate"},
                "debate_type": {"type": "string", "enum": ["verification", "methodology", "interpretation"], "default": "verification"},
                "linked_finding_id": {"type": "string"},
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "get_debate_verdict",
        "Get the result of a completed debate.",
        {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "log_strategy_outcome",
        "Record what research strategy was used and whether it worked.",
        {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "description": "What approach was taken"},
                "context": {"type": "string", "description": "What situation prompted this"},
                "outcome": {"type": "string", "description": "What happened"},
                "success": {"type": "boolean"},
                "lessons": {"type": "string", "default": ""},
            },
            "required": ["strategy", "context", "outcome", "success"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "suggest_strategy",
        "Get a meta-learned strategy suggestion based on past successes and failures.",
        {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "Current research situation"},
            },
            "required": ["context"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "get_research_insights",
        "Get aggregated meta-learning insights: success rates, patterns, recommendations.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    # -- Phase 5: Causal, Benchmarks, Tool Learning, Safety, Roles, Physics
    tool_def(
        "build_causal_dag",
        "Create a causal DAG (directed acyclic graph). Validates acyclicity via sandbox.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nodes": {"type": "array", "items": {"type": "string"}, "description": "Variable names"},
                "edges": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "description": "Directed edges as [cause, effect] pairs",
                },
                "confounders": {"type": "array", "items": {"type": "string"}, "default": []},
                "linked_hypothesis_id": {"type": "string"},
            },
            "required": ["name", "nodes", "edges"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "run_granger_test",
        "Run Granger causality test between two time series variables.",
        {
            "type": "object",
            "properties": {
                "graph_id": {"type": "string"},
                "cause": {"type": "string", "description": "Name of potential cause variable"},
                "effect": {"type": "string", "description": "Name of potential effect variable"},
                "data": {"type": "object", "description": "Dict of variable_name -> list of values"},
                "max_lag": {"type": "integer", "default": 5},
            },
            "required": ["graph_id", "cause", "effect", "data"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "estimate_causal_effect",
        "Estimate causal effect using backdoor adjustment, IV, or propensity score.",
        {
            "type": "object",
            "properties": {
                "graph_id": {"type": "string"},
                "treatment": {"type": "string"},
                "outcome": {"type": "string"},
                "data": {"type": "object", "description": "Dict of variable_name -> list of values"},
                "method": {
                    "type": "string",
                    "enum": ["backdoor", "iv", "frontdoor", "propensity_score"],
                    "default": "backdoor",
                },
            },
            "required": ["graph_id", "treatment", "outcome", "data"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "test_dag_fit",
        "Test whether observed data is consistent with proposed causal DAG.",
        {
            "type": "object",
            "properties": {
                "graph_id": {"type": "string"},
                "data": {"type": "object", "description": "Dict of variable_name -> list of values"},
            },
            "required": ["graph_id", "data"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "suggest_confounders",
        "Use LLM to suggest potential confounders for edges in a causal DAG.",
        {
            "type": "object",
            "properties": {"graph_id": {"type": "string"}},
            "required": ["graph_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "register_benchmark",
        "Register an external benchmark (ground-truth value) for validation.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "metric": {"type": "string"},
                "ground_truth_value": {"type": "number"},
                "tolerance": {"type": "number", "default": 0.05},
                "source": {"type": "string", "default": ""},
                "description": {"type": "string", "default": ""},
            },
            "required": ["name", "metric", "ground_truth_value"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "validate_against_benchmark",
        "Compare a measured value against a registered benchmark.",
        {
            "type": "object",
            "properties": {
                "benchmark_name": {"type": "string"},
                "measured_value": {"type": "number"},
                "execution_id": {"type": "string"},
                "notes": {"type": "string", "default": ""},
            },
            "required": ["benchmark_name", "measured_value"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "list_benchmarks",
        "List registered benchmarks with ground-truth values.",
        {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Filter by domain (e.g. 'climate')"},
            },
            "additionalProperties": False,
        },
    ),
    tool_def(
        "get_benchmark_report",
        "Get aggregate benchmark validation report: pass rates, details.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    tool_def(
        "discover_package",
        "Install and introspect a Python package in sandbox. Stores capabilities for future use.",
        {
            "type": "object",
            "properties": {
                "package_name": {"type": "string"},
                "purpose": {"type": "string", "default": "", "description": "What you want to use it for"},
            },
            "required": ["package_name"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "test_package",
        "Run a test script using a learned package to validate it works.",
        {
            "type": "object",
            "properties": {
                "package_name": {"type": "string"},
                "test_script": {"type": "string", "description": "Python script to test the package"},
            },
            "required": ["package_name", "test_script"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "recommend_package",
        "Get a recommendation for which learned package best fits a problem.",
        {
            "type": "object",
            "properties": {
                "problem_description": {"type": "string"},
            },
            "required": ["problem_description"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "list_learned_tools",
        "List all dynamically discovered packages with success/failure stats.",
        {
            "type": "object",
            "properties": {
                "min_success_count": {"type": "integer", "default": 0},
            },
            "additionalProperties": False,
        },
    ),
    tool_def(
        "run_safety_audit",
        "Run full safety audit: detect p-hacking, HARKing, cherry-picking, data dredging.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    tool_def(
        "get_audit_trail",
        "Full provenance trace for a knowledge entity: hypothesis -> experiments -> findings.",
        {
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "list_safety_flags",
        "List active safety/alignment flags from audits.",
        {
            "type": "object",
            "properties": {
                "include_dismissed": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
    ),
    tool_def(
        "dismiss_safety_flag",
        "Dismiss a safety flag with a documented reason.",
        {
            "type": "object",
            "properties": {
                "flag_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["flag_id", "reason"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "spawn_specialist",
        "Spawn a specialist child agent with a defined role (experimenter, theorist, reviewer, data_curator).",
        {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["experimenter", "theorist", "reviewer", "data_curator"],
                },
                "topic": {"type": "string"},
                "name": {"type": "string"},
                "budget_cents": {"type": "integer"},
            },
            "required": ["role", "topic"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "assign_task_to_specialist",
        "Assign a typed task to a specialist child agent.",
        {
            "type": "object",
            "properties": {
                "child_id": {"type": "string"},
                "task_type": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["child_id", "task_type", "description"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "check_specialist_task",
        "Check status and results of a task assigned to a specialist.",
        {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "synthesize_specialist_results",
        "Use LLM to synthesize results from multiple specialist tasks.",
        {
            "type": "object",
            "properties": {
                "task_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["task_ids"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "get_team_status",
        "Overview of all specialist children: roles, statuses, task progress.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    tool_def(
        "run_physics_sim",
        "Run a toy physics simulation (lorenz63, shallow_water, barotropic_vorticity, tc_potential_intensity, simple_gcm).",
        {
            "type": "object",
            "properties": {
                "sim_type": {
                    "type": "string",
                    "enum": ["lorenz63", "shallow_water", "barotropic_vorticity",
                             "tc_potential_intensity", "simple_gcm"],
                },
                "parameters": {"type": "object", "default": {}, "description": "Override default parameters"},
                "description": {"type": "string", "default": ""},
            },
            "required": ["sim_type"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "list_physics_sims",
        "List past physics simulation runs.",
        {
            "type": "object",
            "properties": {
                "sim_type": {"type": "string", "description": "Filter by simulation type"},
            },
            "additionalProperties": False,
        },
    ),
    tool_def(
        "list_available_sim_types",
        "List available physics simulation types with descriptions and default parameters.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    # -- Phase 6: Full paper reading + literature-aware novelty -----------
    tool_def(
        "search_papers_multi",
        "Search multiple sources (arXiv, Semantic Scholar, OpenAlex) for papers. "
        "Results are auto-stored in the local DB for future FTS queries.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5, "description": "Max results per source"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["arxiv", "semantic_scholar", "openalex"]},
                    "default": ["arxiv", "semantic_scholar", "openalex"],
                    "description": "Which sources to search",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "read_paper",
        "Read a paper's full text. Auto-fetches and extracts PDF if needed. "
        "Returns markdown-formatted text from the paper.",
        {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "ID of the paper to read"},
                "max_chars": {"type": "integer", "default": 50000, "description": "Max characters to return"},
            },
            "required": ["paper_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "fetch_paper_pdf",
        "Download a paper's PDF and extract its text. Stores the text in the DB "
        "for future read_paper / FTS queries.",
        {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "ID of the paper"},
            },
            "required": ["paper_id"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "generate_research_ideas",
        "Proactively generate novel research ideas via a 2-stage LLM pipeline. "
        "Combines literature search, existing knowledge, and multi-stage reasoning "
        "to produce ranked ideas with novelty/feasibility/relevance scores. "
        "Costs 2 API calls. Use before propose_hypothesis to identify promising directions.",
        {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Research topic to generate ideas for"},
                "max_ideas": {"type": "integer", "default": 5, "description": "Maximum number of ideas to generate"},
                "constraint": {"type": "string", "default": "", "description": "Optional constraint or focus area"},
                "use_literature": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to search external literature for context",
                },
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "run_swarm_cycle",
        "Run a full autonomous research cycle via the swarm orchestrator. "
        "This coordinates: experiment → anomaly analysis → hypothesis generation → "
        "novelty check → hypothesis testing → adversarial review → synthesis. "
        "Use this for discovery-driven research where you want the system to find "
        "anomalies and generate novel hypotheses automatically.",
        {
            "type": "object",
            "properties": {
                "research_goal": {
                    "type": "string",
                    "description": "The research question or goal for this cycle",
                },
                "max_hypotheses": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum hypotheses to generate per cycle",
                },
                "skip_literature": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip literature scan phase (faster)",
                },
                "predictions_path": {
                    "type": "string",
                    "description": "Path to predictions JSONL (auto-detected if omitted)",
                },
                "truth_path": {
                    "type": "string",
                    "description": "Path to truth JSONL (auto-detected if omitted)",
                },
                "generate_paper": {
                    "type": "boolean",
                    "default": False,
                    "description": "Generate a full scientific paper from findings",
                },
                "paper_title": {
                    "type": "string",
                    "description": "Paper title (auto-generated if omitted)",
                },
                "max_revisions": {
                    "type": "integer",
                    "default": 2,
                    "description": "Max paper review/revision rounds",
                },
            },
            "required": ["research_goal"],
            "additionalProperties": False,
        },
    ),
    tool_def(
        "analyze_anomalies",
        "Run anomaly analysis on experiment results without a full swarm cycle. "
        "Detects: outliers, RI blind spots, bias patterns, regime shifts, error trends.",
        {
            "type": "object",
            "properties": {
                "predictions_path": {
                    "type": "string",
                    "description": "Path to predictions JSONL",
                },
                "truth_path": {
                    "type": "string",
                    "description": "Path to truth JSONL",
                },
            },
            "required": ["predictions_path", "truth_path"],
            "additionalProperties": False,
        },
    ),
]


# ---------------------------------------------------------------------------
# Build full tool list: existing + daemon
# ---------------------------------------------------------------------------


def get_all_tool_specs() -> list[dict[str, Any]]:
    """Return all tool definitions for the inference API."""
    from agent.tool_registry import tool_specs as existing_specs

    return existing_specs() + DAEMON_TOOL_SPECS


def get_all_tool_descriptions() -> list[dict[str, Any]]:
    """Return simplified tool descriptions for the system prompt."""
    specs = get_all_tool_specs()
    return [
        {
            "name": s["function"]["name"],
            "description": s["function"]["description"],
        }
        for s in specs
    ]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Executes tools with context from the daemon."""

    def __init__(
        self,
        config: Any,
        db: Any,
        run_dir: Path,
        run_id: str,
        inference: Any | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.run_dir = run_dir
        self.run_id = run_id
        self.inference = inference
        self._existing_handlers: dict[str, Any] | None = None
        self._sandbox: Any | None = None
        self._data_acq: Any | None = None
        self._plans: Any | None = None
        self._novelty: Any | None = None
        self._reports: Any | None = None
        self._exp_loop: Any | None = None
        self._stats: Any | None = None
        self._verifier: Any | None = None
        self._climate: Any | None = None
        self._data_apis: Any | None = None
        self._vision: Any | None = None
        self._paper_writer: Any | None = None
        self._hier_planner: Any | None = None
        self._debate: Any | None = None
        self._meta: Any | None = None
        self._causal: Any | None = None
        self._benchmark: Any | None = None
        self._tool_learner: Any | None = None
        self._safety: Any | None = None
        self._roles: Any | None = None
        self._physics: Any | None = None
        self._literature: Any | None = None
        self._idea_generator: Any | None = None

    @property
    def sandbox(self) -> Any:
        if self._sandbox is None:
            from .sandbox import SandboxRunner
            self._sandbox = SandboxRunner(self.config, self.db)
        return self._sandbox

    @property
    def data_acq(self) -> Any:
        if self._data_acq is None:
            from .data_acquisition import DataAcquisition
            self._data_acq = DataAcquisition(self.config, self.db)
        return self._data_acq

    @property
    def plans(self) -> Any:
        if self._plans is None:
            from .plans import PlanManager
            self._plans = PlanManager(self.db)
        return self._plans

    @property
    def novelty(self) -> Any:
        if self._novelty is None:
            from .novelty import NoveltyChecker
            self._novelty = NoveltyChecker(
                self.inference, self.db, self.config, literature=self.literature
            )
        return self._novelty

    @property
    def reports(self) -> Any:
        if self._reports is None:
            from .report_generator import ReportGenerator
            self._reports = ReportGenerator(self.db, self.config)
        return self._reports

    @property
    def exp_loop(self) -> Any:
        if self._exp_loop is None:
            from .experiment_loop import ExperimentLoop
            self._exp_loop = ExperimentLoop(self.config, self.db, self.sandbox)
        return self._exp_loop

    @property
    def stats(self) -> Any:
        if self._stats is None:
            from .statistics import StatisticalRigor
            self._stats = StatisticalRigor(self.config, self.db, self.sandbox)
        return self._stats

    @property
    def verifier(self) -> Any:
        if self._verifier is None:
            from .verification import ResultVerifier
            self._verifier = ResultVerifier(self.config, self.db, self.sandbox)
        return self._verifier

    @property
    def climate(self) -> Any:
        if self._climate is None:
            from .climate_tools import ClimateToolkit
            self._climate = ClimateToolkit(self.config, self.db, self.sandbox)
        return self._climate

    @property
    def data_apis(self) -> Any:
        if self._data_apis is None:
            from .data_apis import AuthenticatedDataAccess
            self._data_apis = AuthenticatedDataAccess(self.config, self.db)
        return self._data_apis

    @property
    def vision(self) -> Any:
        if self._vision is None:
            from .vision import VisionAnalyzer
            self._vision = VisionAnalyzer(self.inference, self.db, self.config)
        return self._vision

    @property
    def paper_writer(self) -> Any:
        if self._paper_writer is None:
            from .paper_writer import PaperWriter
            self._paper_writer = PaperWriter(self.inference, self.db, self.config)
        return self._paper_writer

    @property
    def hier_planner(self) -> Any:
        if self._hier_planner is None:
            from .hierarchical_planner import HierarchicalPlanner
            self._hier_planner = HierarchicalPlanner(self.db, self.config)
        return self._hier_planner

    @property
    def debate(self) -> Any:
        if self._debate is None:
            from .debate import DebateEngine
            self._debate = DebateEngine(self.inference, self.db, self.config)
        return self._debate

    @property
    def meta(self) -> Any:
        if self._meta is None:
            from .meta_learning import MetaLearner
            self._meta = MetaLearner(self.db, self.config)
        return self._meta

    @property
    def causal(self) -> Any:
        if self._causal is None:
            from .causal_reasoning import CausalReasoner
            self._causal = CausalReasoner(self.config, self.db, self.sandbox, self.inference)
        return self._causal

    @property
    def benchmark(self) -> Any:
        if self._benchmark is None:
            from .benchmark import BenchmarkValidator
            self._benchmark = BenchmarkValidator(self.config, self.db)
        return self._benchmark

    @property
    def tool_learner(self) -> Any:
        if self._tool_learner is None:
            from .tool_learning import ToolLearner
            self._tool_learner = ToolLearner(self.config, self.db, self.sandbox, self.inference)
        return self._tool_learner

    @property
    def safety(self) -> Any:
        if self._safety is None:
            from .safety import SafetyMonitor
            self._safety = SafetyMonitor(self.config, self.db)
        return self._safety

    @property
    def roles(self) -> Any:
        if self._roles is None:
            from .agent_roles import AgentRoleManager
            self._roles = AgentRoleManager(self.config, self.db, self.inference)
        return self._roles

    @property
    def physics(self) -> Any:
        if self._physics is None:
            from .physics_sim import PhysicsSimulator
            self._physics = PhysicsSimulator(self.config, self.db, self.sandbox)
        return self._physics

    @property
    def literature(self) -> Any:
        if self._literature is None:
            from .literature import LiteratureManager
            self._literature = LiteratureManager(self.config, self.db)
        return self._literature

    @property
    def idea_generator(self) -> Any:
        if self._idea_generator is None:
            from .idea_generator import IdeaGenerator
            self._idea_generator = IdeaGenerator(
                self.inference, self.db, self.config, literature=self.literature
            )
        return self._idea_generator

    def _get_existing_handlers(self) -> dict[str, Any]:
        if self._existing_handlers is None:
            from agent.tool_registry import handlers, make_agent_config_from_daemon, ToolContext
            from agent.memory import RunMemory

            agent_config = make_agent_config_from_daemon(
                self.config, self.run_dir, self.run_id
            )
            memory = RunMemory(self.run_dir)
            self._tool_context = ToolContext(
                cfg=agent_config, memory=memory, touched_paths=set()
            )
            self._existing_handlers = handlers()
        return self._existing_handlers

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name. Returns result dict."""
        start = time.monotonic()
        try:
            result = self._dispatch(name, args)
        except Exception as exc:
            return {
                "id": str(ULID()),
                "name": name,
                "arguments": args,
                "result": "",
                "duration_ms": int((time.monotonic() - start) * 1000),
                "error": str(exc),
            }

        elapsed = int((time.monotonic() - start) * 1000)
        if isinstance(result, dict):
            text = json.dumps(result, ensure_ascii=False, default=str)
        else:
            text = str(result)
        # Truncate large results
        if len(text) > 8000:
            text = text[:8000] + "...(truncated)"

        return {
            "id": str(ULID()),
            "name": name,
            "arguments": args,
            "result": text,
            "duration_ms": elapsed,
            "error": None,
        }

    def _dispatch(self, name: str, args: dict[str, Any]) -> Any:
        # Daemon-specific tools
        daemon_handlers = {
            "sleep": self._tool_sleep,
            "save_knowledge": self._tool_save_knowledge,
            "query_knowledge": self._tool_query_knowledge,
            "propose_hypothesis": self._tool_propose_hypothesis,
            "record_experiment": self._tool_record_experiment,
            "record_finding": self._tool_record_finding,
            "conclude_hypothesis": self._tool_conclude_hypothesis,
            "search_literature": self._tool_search_literature,
            "get_hypothesis_status": self._tool_get_hypothesis_status,
            "modify_heartbeat": self._tool_modify_heartbeat,
            "system_status": self._tool_system_status,
            "list_research_cycles": self._tool_list_research_cycles,
            "spawn_child": self._tool_spawn_child,
            "check_child": self._tool_check_child,
            "send_message": self._tool_send_message,
            "read_inbox": self._tool_read_inbox,
            "list_peers": self._tool_list_peers,
            "register_peer": self._tool_register_peer,
            "broadcast_finding": self._tool_broadcast_finding,
            "cite_paper": self._tool_cite_paper,
            # Phase 3
            "sandbox_run": self._tool_sandbox_run,
            "sandbox_list_results": self._tool_sandbox_list_results,
            "acquire_dataset": self._tool_acquire_dataset,
            "search_datasets": self._tool_search_datasets,
            "create_research_plan": self._tool_create_research_plan,
            "update_plan_step": self._tool_update_plan_step,
            "get_plan_status": self._tool_get_plan_status,
            "list_plans": self._tool_list_plans,
            "check_novelty": self._tool_check_novelty,
            "generate_report": self._tool_generate_report,
            "update_paper_section": self._tool_update_paper_section,
            # Phase 4
            "start_experiment_loop": self._tool_start_experiment_loop,
            "run_next_trial": self._tool_run_next_trial,
            "get_loop_status": self._tool_get_loop_status,
            "run_statistical_test": self._tool_run_statistical_test,
            "validate_experiment_rigor": self._tool_validate_experiment_rigor,
            "verify_result": self._tool_verify_result,
            "run_climate_analysis": self._tool_run_climate_analysis,
            "fetch_era5": self._tool_fetch_era5,
            "analyze_figure": self._tool_analyze_figure,
            "compare_figures": self._tool_compare_figures,
            "draft_paper": self._tool_draft_paper,
            "review_paper": self._tool_review_paper,
            "revise_paper": self._tool_revise_paper,
            "add_plan_branch": self._tool_add_plan_branch,
            "evaluate_branch": self._tool_evaluate_branch,
            "start_debate": self._tool_start_debate,
            "get_debate_verdict": self._tool_get_debate_verdict,
            "log_strategy_outcome": self._tool_log_strategy_outcome,
            "suggest_strategy": self._tool_suggest_strategy,
            "get_research_insights": self._tool_get_research_insights,
            # Phase 5
            "build_causal_dag": self._tool_build_causal_dag,
            "run_granger_test": self._tool_run_granger_test,
            "estimate_causal_effect": self._tool_estimate_causal_effect,
            "test_dag_fit": self._tool_test_dag_fit,
            "suggest_confounders": self._tool_suggest_confounders,
            "register_benchmark": self._tool_register_benchmark,
            "validate_against_benchmark": self._tool_validate_against_benchmark,
            "list_benchmarks": self._tool_list_benchmarks,
            "get_benchmark_report": self._tool_get_benchmark_report,
            "discover_package": self._tool_discover_package,
            "test_package": self._tool_test_package,
            "recommend_package": self._tool_recommend_package,
            "list_learned_tools": self._tool_list_learned_tools,
            "run_safety_audit": self._tool_run_safety_audit,
            "get_audit_trail": self._tool_get_audit_trail,
            "list_safety_flags": self._tool_list_safety_flags,
            "dismiss_safety_flag": self._tool_dismiss_safety_flag,
            "spawn_specialist": self._tool_spawn_specialist,
            "assign_task_to_specialist": self._tool_assign_task_to_specialist,
            "check_specialist_task": self._tool_check_specialist_task,
            "synthesize_specialist_results": self._tool_synthesize_specialist_results,
            "get_team_status": self._tool_get_team_status,
            "run_physics_sim": self._tool_run_physics_sim,
            "list_physics_sims": self._tool_list_physics_sims,
            "list_available_sim_types": self._tool_list_available_sim_types,
            # Phase 6
            "search_papers_multi": self._tool_search_papers_multi,
            "read_paper": self._tool_read_paper,
            "fetch_paper_pdf": self._tool_fetch_paper_pdf,
            # Phase 7
            "generate_research_ideas": self._tool_generate_research_ideas,
            # Swarm orchestrator
            "run_swarm_cycle": self._tool_run_swarm_cycle,
            "analyze_anomalies": self._tool_analyze_anomalies,
        }

        if name in daemon_handlers:
            return daemon_handlers[name](args)

        # Existing tool_registry handlers
        existing = self._get_existing_handlers()
        if name in existing:
            return existing[name](self._tool_context, args)

        return {"ok": False, "error": f"unknown tool: {name}"}

    # -- Daemon tool implementations -------------------------------------

    def _tool_sleep(self, args: dict[str, Any]) -> dict[str, Any]:
        seconds = min(int(args.get("seconds", 60)), 300)
        wake_at = datetime.now(timezone.utc).timestamp() + seconds
        from datetime import datetime as dt

        wake_iso = dt.fromtimestamp(wake_at, tz=timezone.utc).isoformat()
        self.db.set_kv("sleep_until", wake_iso)
        return {"ok": True, "sleep_seconds": seconds, "wake_at": wake_iso}

    def _tool_save_knowledge(self, args: dict[str, Any]) -> dict[str, Any]:
        from .knowledge import save_knowledge

        kid = save_knowledge(
            self.db,
            topic=str(args.get("topic", "")),
            content=str(args.get("content", "")),
            source=str(args.get("source", "")),
            confidence=float(args.get("confidence", 0.5)),
            entry_type=str(args.get("entry_type", "observation")),
            parent_id=args.get("parent_id"),
            status=str(args.get("status", "active")),
        )
        return {"ok": True, "id": kid}

    def _tool_query_knowledge(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        limit = int(args.get("limit", 10))
        entry_type = args.get("entry_type")
        use_semantic = bool(args.get("semantic", False))

        if use_semantic:
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return {"ok": False, "error": "OPENAI_API_KEY not set for semantic search"}
            from . import embeddings
            candidates = self.db.get_all_knowledge_embeddings()
            if not candidates:
                return {"ok": True, "count": 0, "entries": [], "note": "No embeddings computed yet"}
            ranked = embeddings.search_by_embedding(query, candidates, api_key, limit=limit * 2)
            # Fetch full entries and filter
            entries_out = []
            for kid, score in ranked:
                row = self.db.get_knowledge_by_id(kid)
                if not row:
                    continue
                if entry_type and row.get("entry_type") != entry_type:
                    continue
                entries_out.append({
                    "id": row["id"],
                    "topic": row["topic"],
                    "content": row["content"][:500],
                    "source": row.get("source", ""),
                    "confidence": row.get("confidence", 0.5),
                    "entry_type": row.get("entry_type", "observation"),
                    "status": row.get("status", "active"),
                    "parent_id": row.get("parent_id"),
                    "similarity": round(score, 4),
                })
                if len(entries_out) >= limit:
                    break
            return {"ok": True, "count": len(entries_out), "entries": entries_out}

        from .knowledge import search_knowledge

        entries = search_knowledge(
            self.db,
            query=query,
            limit=limit,
            entry_type=entry_type,
        )
        return {
            "ok": True,
            "count": len(entries),
            "entries": [
                {
                    "id": e.id,
                    "topic": e.topic,
                    "content": e.content[:500],
                    "source": e.source,
                    "confidence": e.confidence,
                    "entry_type": e.entry_type,
                    "status": e.status,
                    "parent_id": e.parent_id,
                }
                for e in entries
            ],
        }

    def _tool_propose_hypothesis(self, args: dict[str, Any]) -> dict[str, Any]:
        from .knowledge import save_hypothesis

        content = str(args.get("content", ""))
        kid = save_hypothesis(
            self.db,
            content=content,
            source=str(args.get("source", "agent reasoning")),
        )

        result: dict[str, Any] = {
            "ok": True, "id": kid, "entry_type": "hypothesis", "status": "active",
        }

        # Auto novelty check if enabled
        auto_check = getattr(self.config, "novelty_auto_check", False)
        if auto_check and self.inference:
            try:
                verdict = self.novelty.check(
                    idea=content,
                    idea_type="hypothesis",
                    linked_hypothesis_id=kid,
                )
                result["novelty_check"] = {
                    "check_id": verdict.id,
                    "is_novel": verdict.is_novel,
                    "novelty_score": verdict.novelty_score,
                    "recommendation": verdict.recommendation,
                }
            except Exception as exc:
                result["novelty_check_error"] = str(exc)

        return result

    def _tool_record_experiment(self, args: dict[str, Any]) -> dict[str, Any]:
        from .knowledge import save_experiment

        hypothesis_id = str(args.get("hypothesis_id", ""))
        if not hypothesis_id:
            return {"ok": False, "error": "hypothesis_id required"}
        # Verify hypothesis exists
        h = self.db.get_knowledge_by_id(hypothesis_id)
        if not h:
            return {"ok": False, "error": f"hypothesis not found: {hypothesis_id}"}

        kid = save_experiment(
            self.db,
            content=str(args.get("content", "")),
            source=str(args.get("source", "experiment")),
            hypothesis_id=hypothesis_id,
        )
        return {"ok": True, "id": kid, "entry_type": "experiment", "hypothesis_id": hypothesis_id}

    def _tool_record_finding(self, args: dict[str, Any]) -> dict[str, Any]:
        from .knowledge import save_finding

        experiment_id = str(args.get("experiment_id", ""))
        if not experiment_id:
            return {"ok": False, "error": "experiment_id required"}
        # Verify experiment exists
        e = self.db.get_knowledge_by_id(experiment_id)
        if not e:
            return {"ok": False, "error": f"experiment not found: {experiment_id}"}

        kid = save_finding(
            self.db,
            content=str(args.get("content", "")),
            source=str(args.get("source", "analysis")),
            experiment_id=experiment_id,
            metrics=args.get("metrics"),
        )
        return {"ok": True, "id": kid, "entry_type": "finding", "experiment_id": experiment_id}

    def _tool_conclude_hypothesis(self, args: dict[str, Any]) -> dict[str, Any]:
        from .knowledge import save_conclusion

        hypothesis_id = str(args.get("hypothesis_id", ""))
        verdict = str(args.get("verdict", ""))
        reasoning = str(args.get("reasoning", ""))
        if not hypothesis_id or not verdict or not reasoning:
            return {"ok": False, "error": "hypothesis_id, verdict, and reasoning required"}

        h = self.db.get_knowledge_by_id(hypothesis_id)
        if not h:
            return {"ok": False, "error": f"hypothesis not found: {hypothesis_id}"}

        kid = save_conclusion(
            self.db,
            content=reasoning,
            source="agent evaluation",
            hypothesis_id=hypothesis_id,
            verdict=verdict,
        )
        return {
            "ok": True,
            "id": kid,
            "entry_type": "conclusion",
            "hypothesis_id": hypothesis_id,
            "verdict": verdict,
        }

    def _tool_search_literature(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        limit = int(args.get("limit", 10))
        use_semantic = bool(args.get("semantic", False))
        if not query:
            return {"ok": False, "error": "query required"}

        if use_semantic:
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return {"ok": False, "error": "OPENAI_API_KEY not set for semantic search"}
            from . import embeddings
            candidates = self.db.get_all_paper_embeddings()
            if not candidates:
                return {"ok": True, "count": 0, "papers": [], "note": "No embeddings computed yet"}
            ranked = embeddings.search_by_embedding(query, candidates, api_key, limit=limit)
            papers_out = []
            for pid, score in ranked:
                p = self.db.get_paper_by_id(pid)
                if not p:
                    continue
                papers_out.append({
                    "id": p["id"],
                    "title": p.get("title", ""),
                    "authors": p.get("authors", "")[:200],
                    "year": p.get("year"),
                    "doi": p.get("doi"),
                    "abstract": p.get("abstract", "")[:400],
                    "cited_by_count": p.get("cited_by_count", 0),
                    "similarity": round(score, 4),
                })
            return {"ok": True, "count": len(papers_out), "papers": papers_out}

        papers = self.db.search_papers_fts(query, limit)
        return {
            "ok": True,
            "count": len(papers),
            "papers": [
                {
                    "id": p["id"],
                    "title": p.get("title", ""),
                    "authors": p.get("authors", "")[:200],
                    "year": p.get("year"),
                    "doi": p.get("doi"),
                    "abstract": p.get("abstract", "")[:400],
                    "cited_by_count": p.get("cited_by_count", 0),
                }
                for p in papers
            ],
        }

    def _tool_get_hypothesis_status(self, args: dict[str, Any]) -> dict[str, Any]:
        from .knowledge import get_hypothesis_tree

        hypothesis_id = str(args.get("hypothesis_id", ""))
        if not hypothesis_id:
            return {"ok": False, "error": "hypothesis_id required"}

        tree = get_hypothesis_tree(self.db, hypothesis_id)
        if "error" in tree:
            return {"ok": False, "error": tree["error"]}

        h = tree["hypothesis"]

        # Collect citations for all entries in the tree
        all_entries = (
            [h]
            + tree.get("experiments", [])
            + tree.get("findings", [])
            + tree.get("conclusions", [])
        )
        citations_by_entry: dict[str, list[dict[str, Any]]] = {}
        for entry in all_entries:
            cites = self.db.get_citations_for_knowledge(entry["id"])
            if cites:
                citations_by_entry[entry["id"]] = [
                    {
                        "paper_id": c["paper_id"],
                        "title": c.get("title", "")[:100],
                        "context": c.get("context", ""),
                    }
                    for c in cites
                ]

        result: dict[str, Any] = {
            "ok": True,
            "hypothesis": {
                "id": h["id"],
                "content": h["content"],
                "status": h.get("status", "active"),
                "created_at": h.get("created_at", ""),
            },
            "experiments": [
                {"id": e["id"], "content": e["content"][:300]}
                for e in tree["experiments"]
            ],
            "findings": [
                {"id": f["id"], "content": f["content"][:300]}
                for f in tree["findings"]
            ],
            "conclusions": [
                {"id": c["id"], "content": c["content"][:300]}
                for c in tree["conclusions"]
            ],
        }
        if citations_by_entry:
            result["citations"] = citations_by_entry
        return result

    def _tool_modify_heartbeat(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name", ""))
        if not name:
            return {"ok": False, "error": "name required"}
        entry: dict[str, Any] = {"name": name}
        if "schedule" in args:
            entry["schedule"] = args["schedule"]
        if "task" in args:
            entry["task"] = args["task"]
        if "enabled" in args:
            entry["enabled"] = args["enabled"]
        # If updating existing, merge
        existing = None
        for e in self.db.get_heartbeat_entries():
            if e["name"] == name:
                existing = e
                break
        if existing:
            existing.update(entry)
            self.db.upsert_heartbeat_entry(existing)
        else:
            if "schedule" not in entry or "task" not in entry:
                return {"ok": False, "error": "new entries need schedule and task"}
            self.db.upsert_heartbeat_entry(entry)

        self.db.insert_modification({
            "id": str(ULID()),
            "timestamp": _utc_now(),
            "type": "heartbeat_change",
            "description": f"Modified heartbeat: {name}",
            "reversible": True,
        })
        return {"ok": True, "name": name}

    def _tool_system_status(self, args: dict[str, Any]) -> dict[str, Any]:
        from .cost_tracker import get_total_spent, get_budget_tier

        return {
            "ok": True,
            "state": self.db.get_agent_state(),
            "turn_count": self.db.get_turn_count(),
            "cost_spent_cents": get_total_spent(self.db),
            "budget_tier": get_budget_tier(self.db, self.config),
            "budget_max_cents": self.config.max_api_cost_cents,
            "knowledge_count": len(self.db.get_recent_knowledge(limit=1000)),
            "children_alive": len([
                c for c in self.db.get_children()
                if c.get("status") not in ("dead", "stopped")
            ]),
            "peers": len(self.db.get_peers()),
            "unread_inbox": len(self.db.get_unprocessed_inbox(limit=100)),
            "model": self.config.inference_model,
        }

    def _tool_list_research_cycles(self, args: dict[str, Any]) -> dict[str, Any]:
        cycles = self.db.get_research_cycles(limit=int(args.get("limit", 10)))
        return {
            "ok": True,
            "count": len(cycles),
            "cycles": [
                {
                    "id": c["id"],
                    "topic": c["topic"],
                    "phase": c["phase"],
                    "status": c["status"],
                    "results_summary": (c.get("results_summary") or "")[:200],
                }
                for c in cycles
            ],
        }

    def _tool_spawn_child(self, args: dict[str, Any]) -> dict[str, Any]:
        from .children import spawn_child

        try:
            child = spawn_child(
                parent_config=self.config,
                parent_db=self.db,
                name=str(args.get("name", "")),
                topic=str(args.get("topic", "")),
                budget_cents=int(args.get("budget_cents", 500)),
                genesis_prompt=str(args.get("genesis_prompt", "")),
            )
            return {"ok": True, "child_id": child["id"], "name": child["name"]}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_check_child(self, args: dict[str, Any]) -> dict[str, Any]:
        from .children import check_child_status, read_child_knowledge

        child_id = str(args.get("child_id", ""))
        child = self.db.get_child_by_id(child_id)
        if not child:
            return {"ok": False, "error": f"child not found: {child_id}"}

        status = check_child_status(self.db, child_id)
        knowledge = read_child_knowledge(Path(child["config_path"]))
        return {
            "ok": True,
            "child_id": child_id,
            "name": child["name"],
            "status": status.get("status", "unknown"),
            "knowledge_count": len(knowledge),
            "recent_knowledge": [
                {"topic": k.get("topic", ""), "content": k.get("content", "")[:200]}
                for k in knowledge[:5]
            ],
        }

    def _tool_send_message(self, args: dict[str, Any]) -> dict[str, Any]:
        to = str(args.get("to", ""))
        content = str(args.get("content", ""))
        if not to or not content:
            return {"ok": False, "error": "to and content required"}

        # Check if target is a child
        child = self.db.get_child_by_id(to)
        if child:
            from .children import send_to_child

            send_to_child(self.db, to, content, self.config.name)
            return {"ok": True, "sent_to": to, "type": "child"}

        # Check if target is a peer
        peer = self.db.get_peer_by_name(to)
        if peer:
            from .social import send_local, send_remote

            if peer.get("url", "").startswith("local:"):
                # Local peer - direct SQLite write
                db_path = peer["url"].removeprefix("local:")
                send_local(self.config.name, Path(db_path), content)
            else:
                send_remote(self.config.name, peer["url"], to, content)
            return {"ok": True, "sent_to": to, "type": "peer"}

        return {"ok": False, "error": f"unknown recipient: {to}"}

    def _tool_read_inbox(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit", 10))
        messages = self.db.get_unprocessed_inbox(limit=limit)
        for m in messages:
            self.db.mark_inbox_processed(m["id"])
        return {
            "ok": True,
            "count": len(messages),
            "messages": [
                {
                    "id": m["id"],
                    "from": m["from_agent"],
                    "content": m["content"][:500],
                    "received_at": m.get("received_at", ""),
                }
                for m in messages
            ],
        }

    def _tool_list_peers(self, args: dict[str, Any]) -> dict[str, Any]:
        topic = str(args.get("topic", "")) or None
        peers = self.db.get_peers(topic=topic)
        return {
            "ok": True,
            "count": len(peers),
            "peers": [
                {
                    "name": p["name"],
                    "url": p["url"],
                    "topic": p.get("topic", ""),
                    "trust_score": p.get("trust_score", 0.5),
                }
                for p in peers
            ],
        }

    def _tool_register_peer(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name", ""))
        url = str(args.get("url", ""))
        if not name or not url:
            return {"ok": False, "error": "name and url required"}
        self.db.upsert_peer({
            "name": name,
            "url": url,
            "topic": str(args.get("topic", "")),
            "last_seen": _utc_now(),
        })
        return {"ok": True, "name": name, "url": url}

    def _tool_broadcast_finding(self, args: dict[str, Any]) -> dict[str, Any]:
        kid = str(args.get("knowledge_id", ""))
        entry = self.db.get_knowledge_by_id(kid)
        if not entry:
            return {"ok": False, "error": f"knowledge not found: {kid}"}

        from .social import broadcast_to_peers

        sent = broadcast_to_peers(
            self.config.name, self.db, entry
        )
        return {"ok": True, "knowledge_id": kid, "sent_to": sent}

    def _tool_cite_paper(self, args: dict[str, Any]) -> dict[str, Any]:
        knowledge_id = str(args.get("knowledge_id", ""))
        paper_id = str(args.get("paper_id", ""))
        context = str(args.get("context", ""))

        if not knowledge_id or not paper_id:
            return {"ok": False, "error": "knowledge_id and paper_id required"}

        # Validate both exist
        k = self.db.get_knowledge_by_id(knowledge_id)
        if not k:
            return {"ok": False, "error": f"knowledge entry not found: {knowledge_id}"}
        p = self.db.get_paper_by_id(paper_id)
        if not p:
            return {"ok": False, "error": f"paper not found: {paper_id}"}

        self.db.add_citation(knowledge_id, paper_id, context)
        return {
            "ok": True,
            "knowledge_id": knowledge_id,
            "paper_id": paper_id,
            "paper_title": p.get("title", "")[:100],
            "context": context,
        }

    # -- Phase 3: Sandbox + Data + Plans + Novelty + Reports ---------------

    def _tool_sandbox_run(self, args: dict[str, Any]) -> dict[str, Any]:
        script = str(args.get("script", ""))
        if not script:
            return {"ok": False, "error": "script required"}

        requirements = args.get("requirements", [])
        data_files = args.get("data_files", {})
        timeout_s = int(args.get("timeout_s", 120))
        description = str(args.get("description", ""))

        try:
            result = self.sandbox.execute(
                script=script,
                requirements=requirements,
                data_files=data_files,
                timeout_s=timeout_s,
                description=description,
            )
            return {
                "ok": result.returncode == 0,
                "execution_id": result.execution_id,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_s": result.duration_s,
                "metrics": result.metrics,
                "artifacts": result.artifacts,
                "backend": result.backend,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_sandbox_list_results(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit", 10))
        executions = self.db.get_recent_sandbox_executions(limit=limit)
        return {
            "ok": True,
            "count": len(executions),
            "executions": [
                {
                    "id": e["id"],
                    "description": e.get("description", "")[:100],
                    "backend": e["backend"],
                    "returncode": e["returncode"],
                    "duration_s": e["duration_s"],
                    "metrics": e.get("metrics", {}),
                    "created_at": e.get("created_at", ""),
                }
                for e in executions
            ],
        }

    def _tool_acquire_dataset(self, args: dict[str, Any]) -> dict[str, Any]:
        url = str(args.get("url", ""))
        name = str(args.get("name", ""))
        if not url or not name:
            return {"ok": False, "error": "url and name required"}

        try:
            record = self.data_acq.fetch_dataset(
                url=url,
                name=name,
                format_hint=str(args.get("format", "csv")),
                max_bytes=args.get("max_bytes"),
                description=str(args.get("description", "")),
            )
            return {
                "ok": True,
                "dataset_id": record["id"],
                "name": record["name"],
                "local_path": record["local_path"],
                "size_bytes": record.get("size_bytes", 0),
                "sha256": record.get("sha256", "")[:16],
                "cached": bool(self.db.get_dataset_by_url(url)),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_search_datasets(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        if not query:
            return {"ok": False, "error": "query required"}

        results = self.data_acq.search_known_sources(query)
        return {
            "ok": True,
            "count": len(results),
            "sources": results,
        }

    def _tool_create_research_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title", ""))
        if not title:
            return {"ok": False, "error": "title required"}

        steps = args.get("steps", [])
        description = str(args.get("description", ""))

        try:
            plan_id = self.plans.create_plan(
                title=title,
                description=description,
                steps=steps,
            )
            plan = self.plans.get_plan(plan_id)
            return {
                "ok": True,
                "plan_id": plan_id,
                "title": title,
                "step_count": len(plan.get("steps", []) if plan else []),
                "step_ids": [s["id"] for s in plan.get("steps", [])] if plan else [],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_update_plan_step(self, args: dict[str, Any]) -> dict[str, Any]:
        plan_id = str(args.get("plan_id", ""))
        step_id = str(args.get("step_id", ""))
        if not plan_id or not step_id:
            return {"ok": False, "error": "plan_id and step_id required"}

        result = self.plans.update_step(
            plan_id=plan_id,
            step_id=step_id,
            status=args.get("status"),
            notes=args.get("notes"),
        )

        # Handle links
        if result.get("ok"):
            if args.get("link_knowledge_id"):
                self.plans.link_to_step(step_id, "knowledge", args["link_knowledge_id"])
            if args.get("link_execution_id"):
                self.plans.link_to_step(step_id, "execution", args["link_execution_id"])

        return result

    def _tool_get_plan_status(self, args: dict[str, Any]) -> dict[str, Any]:
        plan_id = str(args.get("plan_id", ""))
        if not plan_id:
            return {"ok": False, "error": "plan_id required"}

        progress = self.plans.get_progress(plan_id)
        if "error" in progress:
            return {"ok": False, "error": progress["error"]}

        # Also get step details
        plan = self.plans.get_plan(plan_id)
        steps_summary = []
        if plan:
            for s in plan.get("steps", []):
                steps_summary.append({
                    "id": s["id"],
                    "title": s["title"],
                    "status": s["status"],
                    "step_type": s["step_type"],
                    "notes": s.get("notes", "")[:200],
                })

        return {
            "ok": True,
            **progress,
            "steps": steps_summary,
        }

    def _tool_list_plans(self, args: dict[str, Any]) -> dict[str, Any]:
        status = args.get("status")
        limit = int(args.get("limit", 10))
        plans = self.db.get_research_plans_by_status(status, limit)
        return {
            "ok": True,
            "count": len(plans),
            "plans": [
                {
                    "id": p["id"],
                    "title": p["title"],
                    "status": p["status"],
                    "description": p.get("description", "")[:200],
                    "updated_at": p.get("updated_at", ""),
                }
                for p in plans
            ],
        }

    def _tool_check_novelty(self, args: dict[str, Any]) -> dict[str, Any]:
        idea = str(args.get("idea", ""))
        if not idea:
            return {"ok": False, "error": "idea required"}
        if not self.inference:
            return {"ok": False, "error": "inference client not available for novelty check"}

        context = str(args.get("context", ""))
        idea_type = str(args.get("type", "hypothesis"))

        try:
            verdict = self.novelty.check(
                idea=idea,
                idea_type=idea_type,
                context=context,
            )
            return {
                "ok": True,
                "check_id": verdict.id,
                "is_novel": verdict.is_novel,
                "novelty_score": verdict.novelty_score,
                "recommendation": verdict.recommendation,
                "advocate_summary": verdict.advocate_argument[:500],
                "critic_summary": verdict.critic_argument[:500],
                "prior_art": verdict.prior_art[:5],
                "models_used": verdict.models,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_generate_report(self, args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title", "Research Report"))
        plan_id = args.get("plan_id")
        sections = args.get("sections")
        output_path = str(args.get("output_path", "docs/research_report.md"))

        try:
            content = self.reports.generate_research_report(
                plan_id=plan_id,
                title=title,
                sections=sections,
            )
            path = self.reports.render_to_file(content, output_path)
            return {
                "ok": True,
                "path": str(path),
                "size_chars": len(content),
                "preview": content[:1000],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_update_paper_section(self, args: dict[str, Any]) -> dict[str, Any]:
        section = str(args.get("section", ""))
        content = str(args.get("content", ""))
        paper_path = str(args.get("paper_path", "docs/paper.md"))

        if not section or not content:
            return {"ok": False, "error": "section and content required"}

        try:
            self.reports.update_section(section, content, paper_path)
            return {"ok": True, "section": section, "paper_path": paper_path}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # -- Phase 4: Close all gaps ------------------------------------------

    def _tool_start_experiment_loop(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name", ""))
        param_space = args.get("parameter_space", {})
        target = str(args.get("target_metric", ""))
        if not name or not param_space or not target:
            return {"ok": False, "error": "name, parameter_space, and target_metric required"}
        try:
            loop_id = self.exp_loop.create_loop(
                name=name, parameter_space=param_space, target_metric=target,
                objective=str(args.get("objective", "minimize")),
                description=str(args.get("description", "")),
                linked_hypothesis_id=args.get("linked_hypothesis_id"),
            )
            return {"ok": True, "loop_id": loop_id, "name": name}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_run_next_trial(self, args: dict[str, Any]) -> dict[str, Any]:
        loop_id = str(args.get("loop_id", ""))
        template = str(args.get("script_template", ""))
        if not loop_id or not template:
            return {"ok": False, "error": "loop_id and script_template required"}
        try:
            suggestion = self.exp_loop.suggest_next_params(loop_id)
            if "error" in suggestion:
                return {"ok": False, "error": suggestion["error"]}
            params = suggestion["params"]
            script = template
            for k, v in params.items():
                script = script.replace(f"{{{k}}}", str(v))
            reqs = args.get("requirements", [])
            result = self.sandbox.execute(
                script=script, requirements=reqs,
                description=f"Trial {suggestion['trial_number']} for loop {loop_id}",
            )
            trial_result = self.exp_loop.record_trial_result(
                loop_id, suggestion.get("trial_id", ""),
                metrics=result.metrics or {}, execution_id=result.execution_id,
            )
            return {
                "ok": result.returncode == 0,
                "params": params, "trial_number": suggestion["trial_number"],
                "execution_id": result.execution_id,
                "metrics": result.metrics, "stdout": result.stdout[:2000],
                "improved": trial_result.get("improved", False),
                "converged": trial_result.get("converged", False),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_get_loop_status(self, args: dict[str, Any]) -> dict[str, Any]:
        loop_id = str(args.get("loop_id", ""))
        if not loop_id:
            return {"ok": False, "error": "loop_id required"}
        try:
            return {"ok": True, **self.exp_loop.get_loop_status(loop_id)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_run_statistical_test(self, args: dict[str, Any]) -> dict[str, Any]:
        test_type = str(args.get("test_type", ""))
        data = args.get("data", {})
        if not test_type or not data:
            return {"ok": False, "error": "test_type and data required"}
        try:
            return {"ok": True, **self.stats.run_test(
                test_type=test_type, data=data,
                description=str(args.get("description", "")),
                linked_finding_id=args.get("linked_finding_id"),
                linked_execution_id=args.get("linked_execution_id"),
            )}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_validate_experiment_rigor(self, args: dict[str, Any]) -> dict[str, Any]:
        execution_id = str(args.get("execution_id", ""))
        if not execution_id:
            return {"ok": False, "error": "execution_id required"}
        try:
            return {"ok": True, **self.stats.validate_experiment_rigor(execution_id)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_verify_result(self, args: dict[str, Any]) -> dict[str, Any]:
        execution_id = str(args.get("execution_id", ""))
        if not execution_id:
            return {"ok": False, "error": "execution_id required"}
        try:
            return {"ok": True, **self.verifier.verify_result(execution_id)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_run_climate_analysis(self, args: dict[str, Any]) -> dict[str, Any]:
        analysis_type = str(args.get("analysis_type", ""))
        data_path = str(args.get("data_path", ""))
        if not analysis_type or not data_path:
            return {"ok": False, "error": "analysis_type and data_path required"}
        try:
            params = args.get("parameters", {})
            if analysis_type == "hurdat2_parse":
                return {"ok": True, **self.climate.parse_hurdat2(data_path)}
            script = self.climate.generate_analysis_script(
                analysis_type, data_path=data_path, **params
            )
            result = self.sandbox.execute(
                script=script, requirements=["pandas", "numpy"],
                description=f"Climate analysis: {analysis_type}",
            )
            return {
                "ok": result.returncode == 0, "metrics": result.metrics,
                "stdout": result.stdout[:2000], "stderr": result.stderr[:1000],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_fetch_era5(self, args: dict[str, Any]) -> dict[str, Any]:
        variable = str(args.get("variable", ""))
        year = str(args.get("year", ""))
        month = str(args.get("month", ""))
        if not variable or not year or not month:
            return {"ok": False, "error": "variable, year, and month required"}
        try:
            return {"ok": True, **self.data_apis.fetch_era5(
                variable=variable, year=year, month=month,
                area=args.get("area"),
            )}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_analyze_figure(self, args: dict[str, Any]) -> dict[str, Any]:
        image_path = str(args.get("image_path", ""))
        if not image_path:
            return {"ok": False, "error": "image_path required"}
        if not self.inference:
            return {"ok": False, "error": "inference client needed for vision"}
        try:
            return {"ok": True, **self.vision.analyze_figure(
                image_path=image_path, prompt=args.get("prompt"),
                analysis_type=str(args.get("analysis_type", "interpret")),
                linked_execution_id=args.get("linked_execution_id"),
            )}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_compare_figures(self, args: dict[str, Any]) -> dict[str, Any]:
        p1 = str(args.get("image_path_1", ""))
        p2 = str(args.get("image_path_2", ""))
        if not p1 or not p2:
            return {"ok": False, "error": "image_path_1 and image_path_2 required"}
        if not self.inference:
            return {"ok": False, "error": "inference client needed for vision"}
        try:
            return {"ok": True, **self.vision.compare_figures(
                image_path_1=p1, image_path_2=p2, prompt=args.get("prompt"),
            )}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_draft_paper(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.inference:
            return {"ok": False, "error": "inference client needed for paper writing"}
        try:
            return {"ok": True, **self.paper_writer.draft_paper(
                title=args.get("title"), linked_plan_id=args.get("linked_plan_id"),
            )}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_review_paper(self, args: dict[str, Any]) -> dict[str, Any]:
        draft_id = str(args.get("draft_id", ""))
        if not draft_id:
            return {"ok": False, "error": "draft_id required"}
        if not self.inference:
            return {"ok": False, "error": "inference client needed for review"}
        try:
            return {"ok": True, **self.paper_writer.review_paper(draft_id)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_revise_paper(self, args: dict[str, Any]) -> dict[str, Any]:
        draft_id = str(args.get("draft_id", ""))
        if not draft_id:
            return {"ok": False, "error": "draft_id required"}
        if not self.inference:
            return {"ok": False, "error": "inference client needed for revision"}
        try:
            return {"ok": True, **self.paper_writer.revise_paper(
                draft_id, review_id=args.get("review_id"),
            )}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_add_plan_branch(self, args: dict[str, Any]) -> dict[str, Any]:
        plan_id = str(args.get("plan_id", ""))
        from_step = str(args.get("from_step_id", ""))
        condition = str(args.get("condition", ""))
        if not plan_id or not from_step or not condition:
            return {"ok": False, "error": "plan_id, from_step_id, and condition required"}
        try:
            branch_id = self.hier_planner.add_branch(
                plan_id=plan_id, from_step_id=from_step,
                condition=condition,
                condition_type=str(args.get("condition_type", "metric_threshold")),
                then_steps=args.get("then_steps"),
                else_steps=args.get("else_steps"),
            )
            return {"ok": True, "branch_id": branch_id}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_evaluate_branch(self, args: dict[str, Any]) -> dict[str, Any]:
        branch_id = str(args.get("branch_id", ""))
        if not branch_id:
            return {"ok": False, "error": "branch_id required"}
        try:
            return {"ok": True, **self.hier_planner.evaluate_branch(branch_id)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_start_debate(self, args: dict[str, Any]) -> dict[str, Any]:
        topic = str(args.get("topic", ""))
        if not topic:
            return {"ok": False, "error": "topic required"}
        if not self.inference:
            return {"ok": False, "error": "inference client needed for debate"}
        try:
            return {"ok": True, **self.debate.start_debate(
                topic=topic,
                debate_type=str(args.get("debate_type", "verification")),
                linked_finding_id=args.get("linked_finding_id"),
            )}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_get_debate_verdict(self, args: dict[str, Any]) -> dict[str, Any]:
        session_id = str(args.get("session_id", ""))
        if not session_id:
            return {"ok": False, "error": "session_id required"}
        try:
            return {"ok": True, **self.debate.get_debate_result(session_id)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_log_strategy_outcome(self, args: dict[str, Any]) -> dict[str, Any]:
        strategy = str(args.get("strategy", ""))
        context = str(args.get("context", ""))
        outcome = str(args.get("outcome", ""))
        if not strategy or not context or not outcome:
            return {"ok": False, "error": "strategy, context, and outcome required"}
        try:
            log_id = self.meta.log_strategy_outcome(
                strategy=strategy, context=context, outcome=outcome,
                success=bool(args.get("success", False)),
                lessons=str(args.get("lessons", "")),
            )
            return {"ok": True, "log_id": log_id}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_suggest_strategy(self, args: dict[str, Any]) -> dict[str, Any]:
        context = str(args.get("context", ""))
        if not context:
            return {"ok": False, "error": "context required"}
        try:
            return {"ok": True, **self.meta.suggest_strategy(context)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_get_research_insights(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, **self.meta.get_research_insights()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # -- Phase 5: Causal, Benchmarks, Tool Learning, Safety, Roles, Physics

    def _tool_build_causal_dag(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name", ""))
        nodes = args.get("nodes", [])
        edges = args.get("edges", [])
        if not name or not nodes or not edges:
            return {"ok": False, "error": "name, nodes, and edges required"}
        try:
            return {"ok": True, **self.causal.build_dag(
                name=name, nodes=nodes, edges=edges,
                confounders=args.get("confounders"),
                linked_hypothesis_id=args.get("linked_hypothesis_id"),
            )}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_run_granger_test(self, args: dict[str, Any]) -> dict[str, Any]:
        graph_id = str(args.get("graph_id", ""))
        cause = str(args.get("cause", ""))
        effect = str(args.get("effect", ""))
        data = args.get("data", {})
        if not graph_id or not cause or not effect or not data:
            return {"ok": False, "error": "graph_id, cause, effect, and data required"}
        try:
            result = self.causal.run_granger_test(
                graph_id=graph_id, cause=cause, effect=effect,
                data=data, max_lag=int(args.get("max_lag", 5)),
            )
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_estimate_causal_effect(self, args: dict[str, Any]) -> dict[str, Any]:
        graph_id = str(args.get("graph_id", ""))
        treatment = str(args.get("treatment", ""))
        outcome = str(args.get("outcome", ""))
        data = args.get("data", {})
        if not graph_id or not treatment or not outcome or not data:
            return {"ok": False, "error": "graph_id, treatment, outcome, and data required"}
        try:
            result = self.causal.estimate_causal_effect(
                graph_id=graph_id, treatment=treatment, outcome=outcome,
                data=data, method=args.get("method"),
            )
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_test_dag_fit(self, args: dict[str, Any]) -> dict[str, Any]:
        graph_id = str(args.get("graph_id", ""))
        data = args.get("data", {})
        if not graph_id or not data:
            return {"ok": False, "error": "graph_id and data required"}
        try:
            result = self.causal.test_dag_fit(graph_id=graph_id, data=data)
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_suggest_confounders(self, args: dict[str, Any]) -> dict[str, Any]:
        graph_id = str(args.get("graph_id", ""))
        if not graph_id:
            return {"ok": False, "error": "graph_id required"}
        if not self.inference:
            return {"ok": False, "error": "inference client required for confounder suggestions"}
        try:
            result = self.causal.suggest_confounders(graph_id=graph_id)
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_register_benchmark(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name", ""))
        metric = str(args.get("metric", ""))
        gt_value = args.get("ground_truth_value")
        if not name or not metric or gt_value is None:
            return {"ok": False, "error": "name, metric, and ground_truth_value required"}
        try:
            bm_id = self.benchmark.register_benchmark(
                name=name, metric=metric,
                ground_truth_value=float(gt_value),
                tolerance=float(args.get("tolerance", 0.05)),
                source=str(args.get("source", "")),
                description=str(args.get("description", "")),
            )
            return {"ok": True, "benchmark_id": bm_id, "name": name}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_validate_against_benchmark(self, args: dict[str, Any]) -> dict[str, Any]:
        bm_name = str(args.get("benchmark_name", ""))
        measured = args.get("measured_value")
        if not bm_name or measured is None:
            return {"ok": False, "error": "benchmark_name and measured_value required"}
        try:
            result = self.benchmark.validate_against_benchmark(
                benchmark_name=bm_name,
                measured_value=float(measured),
                execution_id=args.get("execution_id"),
                notes=str(args.get("notes", "")),
            )
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_list_benchmarks(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            benchmarks = self.benchmark.list_benchmarks(
                domain=args.get("domain"),
            )
            return {"ok": True, "count": len(benchmarks), "benchmarks": benchmarks}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_get_benchmark_report(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, **self.benchmark.get_benchmark_report()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_discover_package(self, args: dict[str, Any]) -> dict[str, Any]:
        package_name = str(args.get("package_name", ""))
        if not package_name:
            return {"ok": False, "error": "package_name required"}
        try:
            result = self.tool_learner.discover_package(
                package_name=package_name,
                purpose=str(args.get("purpose", "")),
            )
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_test_package(self, args: dict[str, Any]) -> dict[str, Any]:
        package_name = str(args.get("package_name", ""))
        test_script = str(args.get("test_script", ""))
        if not package_name or not test_script:
            return {"ok": False, "error": "package_name and test_script required"}
        try:
            return {"ok": True, **self.tool_learner.test_package(
                package_name=package_name, test_script=test_script,
            )}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_recommend_package(self, args: dict[str, Any]) -> dict[str, Any]:
        desc = str(args.get("problem_description", ""))
        if not desc:
            return {"ok": False, "error": "problem_description required"}
        try:
            return {"ok": True, **self.tool_learner.recommend_package(desc)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_list_learned_tools(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            tools = self.tool_learner.list_learned_tools(
                min_success=int(args.get("min_success_count", 0)),
            )
            return {"ok": True, "count": len(tools), "tools": tools}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_run_safety_audit(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, **self.safety.run_audit()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_get_audit_trail(self, args: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(args.get("entity_id", ""))
        if not entity_id:
            return {"ok": False, "error": "entity_id required"}
        try:
            result = self.safety.get_audit_trail(entity_id=entity_id)
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_list_safety_flags(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            flags = self.safety.list_flags(
                include_dismissed=bool(args.get("include_dismissed", False)),
            )
            return {"ok": True, "count": len(flags), "flags": flags}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_dismiss_safety_flag(self, args: dict[str, Any]) -> dict[str, Any]:
        flag_id = str(args.get("flag_id", ""))
        reason = str(args.get("reason", ""))
        if not flag_id or not reason:
            return {"ok": False, "error": "flag_id and reason required"}
        try:
            self.safety.dismiss_flag(flag_id=flag_id, reason=reason)
            return {"ok": True, "flag_id": flag_id, "dismissed": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_spawn_specialist(self, args: dict[str, Any]) -> dict[str, Any]:
        role = str(args.get("role", ""))
        topic = str(args.get("topic", ""))
        if not role or not topic:
            return {"ok": False, "error": "role and topic required"}
        try:
            result = self.roles.spawn_specialist(
                role=role, topic=topic,
                name=args.get("name"),
                budget_cents=args.get("budget_cents"),
            )
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_assign_task_to_specialist(self, args: dict[str, Any]) -> dict[str, Any]:
        child_id = str(args.get("child_id", ""))
        task_type = str(args.get("task_type", ""))
        description = str(args.get("description", ""))
        if not child_id or not task_type or not description:
            return {"ok": False, "error": "child_id, task_type, and description required"}
        try:
            task_id = self.roles.assign_task(
                child_id=child_id, task_type=task_type, description=description,
            )
            return {"ok": True, "task_id": task_id, "child_id": child_id}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_check_specialist_task(self, args: dict[str, Any]) -> dict[str, Any]:
        task_id = str(args.get("task_id", ""))
        if not task_id:
            return {"ok": False, "error": "task_id required"}
        try:
            result = self.roles.check_task_status(task_id=task_id)
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_synthesize_specialist_results(self, args: dict[str, Any]) -> dict[str, Any]:
        task_ids = args.get("task_ids", [])
        if not task_ids:
            return {"ok": False, "error": "task_ids required"}
        try:
            result = self.roles.synthesize_results(task_ids=task_ids)
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_get_team_status(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, **self.roles.get_team_status()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_run_physics_sim(self, args: dict[str, Any]) -> dict[str, Any]:
        sim_type = str(args.get("sim_type", ""))
        if not sim_type:
            return {"ok": False, "error": "sim_type required"}
        try:
            result = self.physics.run_simulation(
                sim_type=sim_type,
                parameters=args.get("parameters"),
                description=str(args.get("description", "")),
            )
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_list_physics_sims(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            sims = self.physics.list_simulations(sim_type=args.get("sim_type"))
            return {"ok": True, "count": len(sims), "simulations": sims}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_list_available_sim_types(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, "sim_types": self.physics.list_available_sims()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # -- Phase 6: Literature tools ---------------------------------------

    def _tool_search_papers_multi(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        if not query:
            return {"ok": False, "error": "query required"}
        max_results = int(args.get("max_results", 5))
        sources = args.get("sources") or ["arxiv", "semantic_scholar", "openalex"]
        try:
            return self.literature.search_all(
                query, max_per_source=max_results, sources=sources
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_read_paper(self, args: dict[str, Any]) -> dict[str, Any]:
        paper_id = str(args.get("paper_id", ""))
        if not paper_id:
            return {"ok": False, "error": "paper_id required"}
        max_chars = int(args.get("max_chars", 50000))
        try:
            return self.literature.read_paper(paper_id, max_chars=max_chars)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_fetch_paper_pdf(self, args: dict[str, Any]) -> dict[str, Any]:
        paper_id = str(args.get("paper_id", ""))
        if not paper_id:
            return {"ok": False, "error": "paper_id required"}
        try:
            return self.literature.fetch_paper_pdf(paper_id)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _tool_generate_research_ideas(self, args: dict[str, Any]) -> dict[str, Any]:
        topic = str(args.get("topic", ""))
        if not topic:
            return {"ok": False, "error": "topic required"}
        if not self.inference:
            return {"ok": False, "error": "inference client not available for idea generation"}
        max_ideas = int(args.get("max_ideas", 5))
        constraint = str(args.get("constraint", ""))
        use_literature = bool(args.get("use_literature", True))
        try:
            return self.idea_generator.generate(
                topic=topic,
                max_ideas=max_ideas,
                constraint=constraint,
                use_literature=use_literature,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # -- Swarm orchestrator tools ----------------------------------------

    def _tool_run_swarm_cycle(self, args: dict[str, Any]) -> dict[str, Any]:
        goal = args.get("research_goal", "")
        if not goal:
            return {"ok": False, "error": "research_goal is required"}

        from .orchestrator import SwarmOrchestrator

        orchestrator = SwarmOrchestrator(
            config=self.config,
            db=self.db,
            inference=self.inference,
            tool_executor=self,
        )

        experiment_context = {}
        if args.get("predictions_path"):
            experiment_context["predictions_path"] = args["predictions_path"]
        if args.get("truth_path"):
            experiment_context["truth_path"] = args["truth_path"]

        return orchestrator.run_cycle(
            research_goal=goal,
            max_hypotheses=int(args.get("max_hypotheses", 3)),
            skip_literature=bool(args.get("skip_literature", False)),
            experiment_context=experiment_context or None,
            generate_paper=bool(args.get("generate_paper", False)),
            paper_title=args.get("paper_title"),
            max_revisions=int(args.get("max_revisions", 2)),
        )

    def _tool_analyze_anomalies(self, args: dict[str, Any]) -> dict[str, Any]:
        pred_path = args.get("predictions_path", "")
        truth_path = args.get("truth_path", "")
        if not pred_path or not truth_path:
            return {"ok": False, "error": "predictions_path and truth_path are required"}

        from .anomaly_analyzer import AnomalyAnalyzer

        analyzer = AnomalyAnalyzer(
            config=self.config,
            db=self.db,
            inference=self.inference,
        )

        # Load per-sample data via evaluate_jsonl
        eval_result = self.execute("run_cmd", {
            "cmd": (
                f"uv run --no-project python src/evaluate_jsonl.py "
                f"--predictions {pred_path} --truth {truth_path} "
                f"--per-sample-json"
            ),
        })

        per_sample = []
        metrics = {}
        if isinstance(eval_result, dict) and eval_result.get("result"):
            try:
                parsed = json.loads(eval_result["result"])
                metrics = parsed.get("summary", parsed)
                per_sample = parsed.get("per_sample", [])
            except (json.JSONDecodeError, TypeError):
                pass

        anomalies = analyzer.analyze({
            "metrics": metrics,
            "per_sample": per_sample,
        })

        return {
            "ok": True,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:10],  # cap output size
        }
