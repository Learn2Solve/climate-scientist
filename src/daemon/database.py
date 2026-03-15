"""SQLite persistent state for the daemon.

Port of automaton/src/state/database.ts + schema.ts.
15 tables: 11 from automaton + 4 new for research.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 7

CREATE_TABLES = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Core identity key-value store
CREATE TABLE IF NOT EXISTS identity (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Agent reasoning turns
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    state TEXT NOT NULL,
    input TEXT,
    input_source TEXT,
    thinking TEXT NOT NULL,
    tool_calls TEXT NOT NULL DEFAULT '[]',
    token_usage TEXT NOT NULL DEFAULT '{}',
    cost_cents INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Tool call results (denormalized)
CREATE TABLE IF NOT EXISTS tool_calls (
    id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL REFERENCES turns(id),
    name TEXT NOT NULL,
    arguments TEXT NOT NULL DEFAULT '{}',
    result TEXT NOT NULL DEFAULT '',
    duration_ms INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Heartbeat configuration entries
CREATE TABLE IF NOT EXISTS heartbeat_entries (
    name TEXT PRIMARY KEY,
    schedule TEXT NOT NULL,
    task TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_run TEXT,
    next_run TEXT,
    params TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Cost transaction log
CREATE TABLE IF NOT EXISTS transactions (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    amount_cents INTEGER,
    balance_after_cents INTEGER,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Self-modification audit trail (append-only)
CREATE TABLE IF NOT EXISTS modifications (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    type TEXT NOT NULL,
    description TEXT NOT NULL,
    file_path TEXT,
    diff TEXT,
    reversible INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- General key-value store
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Installed skills
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    auto_activate INTEGER NOT NULL DEFAULT 1,
    requires TEXT DEFAULT '{}',
    instructions TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'builtin',
    path TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    installed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Child agents
CREATE TABLE IF NOT EXISTS children (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    topic TEXT NOT NULL,
    pid INTEGER,
    config_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'spawning',
    budget_cents INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_checked TEXT
);

-- Inbox messages (agent-to-agent)
CREATE TABLE IF NOT EXISTS inbox_messages (
    id TEXT PRIMARY KEY,
    from_agent TEXT NOT NULL,
    content TEXT NOT NULL,
    received_at TEXT NOT NULL DEFAULT (datetime('now')),
    processed_at TEXT,
    reply_to TEXT
);

-- Knowledge base (research findings)
CREATE TABLE IF NOT EXISTS knowledge (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Research cycle tracking
CREATE TABLE IF NOT EXISTS research_cycles (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'planning',
    status TEXT NOT NULL DEFAULT 'active',
    plan_json TEXT DEFAULT '{}',
    results_summary TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Known remote peers for federation
CREATE TABLE IF NOT EXISTS peers (
    name TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    public_key TEXT,
    topic TEXT,
    last_seen TEXT,
    trust_score REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Peer reviews of findings
CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    from_agent TEXT NOT NULL,
    knowledge_id TEXT NOT NULL REFERENCES knowledge(id),
    score REAL NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns(timestamp);
CREATE INDEX IF NOT EXISTS idx_turns_state ON turns(state);
CREATE INDEX IF NOT EXISTS idx_tool_calls_turn ON tool_calls(turn_id);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(type);
CREATE INDEX IF NOT EXISTS idx_modifications_type ON modifications(type);
CREATE INDEX IF NOT EXISTS idx_skills_enabled ON skills(enabled);
CREATE INDEX IF NOT EXISTS idx_children_status ON children(status);
CREATE INDEX IF NOT EXISTS idx_inbox_unprocessed
    ON inbox_messages(received_at) WHERE processed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);
CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge(created_at);
CREATE INDEX IF NOT EXISTS idx_research_cycles_status ON research_cycles(status);
CREATE INDEX IF NOT EXISTS idx_peers_topic ON peers(topic);
CREATE INDEX IF NOT EXISTS idx_reviews_knowledge ON reviews(knowledge_id);
"""


class DaemonDatabase:
    """SQLite-backed persistent state for the daemon."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(CREATE_TABLES)
        self._migrate()

    def _migrate(self) -> None:
        row = self._conn.execute(
            "SELECT MAX(version) as v FROM schema_version"
        ).fetchone()
        current = row["v"] if row and row["v"] else 0
        if current < 2:
            self._migrate_to_v2()
        if current < 3:
            self._migrate_to_v3()
        if current < 4:
            self._migrate_to_v4()
        if current < 5:
            self._migrate_to_v5()
        if current < 6:
            self._migrate_to_v6()
        if current < 7:
            self._migrate_to_v7()
        if current < SCHEMA_VERSION:
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) "
                "VALUES (?, datetime('now'))",
                (SCHEMA_VERSION,),
            )
            self._conn.commit()

    def _migrate_to_v2(self) -> None:
        """Add structured knowledge columns, FTS5 indexes, and papers table."""
        # Add new columns to knowledge table (ignore if already exist)
        for col_sql in [
            "ALTER TABLE knowledge ADD COLUMN entry_type TEXT NOT NULL DEFAULT 'observation'",
            "ALTER TABLE knowledge ADD COLUMN parent_id TEXT",
            "ALTER TABLE knowledge ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "ALTER TABLE knowledge ADD COLUMN metadata_json TEXT DEFAULT '{}'",
        ]:
            try:
                self._conn.execute(col_sql)
            except sqlite3.OperationalError:
                pass  # column already exists

        self._conn.executescript("""
            -- Papers table (harvested from OpenAlex + web_fetch)
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT NOT NULL DEFAULT '',
                year INTEGER,
                doi TEXT,
                abstract TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'openalex',
                url TEXT,
                cited_by_count INTEGER DEFAULT 0,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- FTS5 index on knowledge
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                topic, content, content=knowledge, content_rowid=rowid
            );

            -- Triggers to keep knowledge FTS5 in sync
            CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
                INSERT INTO knowledge_fts(rowid, topic, content)
                VALUES (new.rowid, new.topic, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, topic, content)
                VALUES ('delete', old.rowid, old.topic, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, topic, content)
                VALUES ('delete', old.rowid, old.topic, old.content);
                INSERT INTO knowledge_fts(rowid, topic, content)
                VALUES (new.rowid, new.topic, new.content);
            END;

            -- FTS5 index on papers
            CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                title, abstract, authors, content=papers, content_rowid=rowid
            );

            -- Triggers for papers FTS5
            CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
                INSERT INTO papers_fts(rowid, title, abstract, authors)
                VALUES (new.rowid, new.title, new.abstract, new.authors);
            END;
            CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
                INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors)
                VALUES ('delete', old.rowid, old.title, old.abstract, old.authors);
            END;

            -- Indexes for structured knowledge + papers
            CREATE INDEX IF NOT EXISTS idx_knowledge_parent ON knowledge(parent_id);
            CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(entry_type);
            CREATE INDEX IF NOT EXISTS idx_knowledge_status ON knowledge(status);
            CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
            CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
        """)

        # Rebuild FTS5 index from existing knowledge entries
        self._conn.execute(
            "INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')"
        )
        self._conn.commit()

    def _migrate_to_v3(self) -> None:
        """Add citation links table and embedding columns."""
        self._conn.executescript("""
            -- Citation links between knowledge entries and papers
            CREATE TABLE IF NOT EXISTS knowledge_citations (
                knowledge_id TEXT NOT NULL REFERENCES knowledge(id),
                paper_id TEXT NOT NULL REFERENCES papers(id),
                context TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (knowledge_id, paper_id)
            );
            CREATE INDEX IF NOT EXISTS idx_citations_paper
                ON knowledge_citations(paper_id);
        """)

        # Embedding columns (NULL until computed)
        for col_sql in [
            "ALTER TABLE knowledge ADD COLUMN embedding BLOB",
            "ALTER TABLE papers ADD COLUMN embedding BLOB",
        ]:
            try:
                self._conn.execute(col_sql)
            except sqlite3.OperationalError:
                pass  # column already exists

        self._conn.commit()

    def _migrate_to_v4(self) -> None:
        """Add sandbox, datasets, plans, novelty tables for Phase 3."""
        self._conn.executescript("""
            -- Sandbox executions
            CREATE TABLE IF NOT EXISTS sandbox_executions (
                id TEXT PRIMARY KEY,
                script_hash TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                backend TEXT NOT NULL,
                returncode INTEGER NOT NULL,
                stdout_preview TEXT NOT NULL DEFAULT '',
                stderr_preview TEXT NOT NULL DEFAULT '',
                metrics_json TEXT DEFAULT '{}',
                artifacts_json TEXT DEFAULT '[]',
                duration_s REAL NOT NULL DEFAULT 0,
                work_dir TEXT NOT NULL,
                linked_experiment_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Datasets
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_url TEXT NOT NULL,
                format TEXT NOT NULL DEFAULT 'csv',
                local_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                description TEXT NOT NULL DEFAULT '',
                fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                verified INTEGER NOT NULL DEFAULT 1,
                UNIQUE(source_url)
            );

            -- Research plans
            CREATE TABLE IF NOT EXISTS research_plans (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS plan_steps (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL REFERENCES research_plans(id),
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                step_type TEXT NOT NULL DEFAULT 'custom',
                status TEXT NOT NULL DEFAULT 'pending',
                step_order INTEGER NOT NULL DEFAULT 0,
                depends_on_json TEXT DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS plan_step_links (
                step_id TEXT NOT NULL REFERENCES plan_steps(id),
                link_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (step_id, link_type, target_id)
            );

            -- Novelty checks
            CREATE TABLE IF NOT EXISTS novelty_checks (
                id TEXT PRIMARY KEY,
                idea_text TEXT NOT NULL,
                idea_type TEXT NOT NULL DEFAULT 'hypothesis',
                is_novel INTEGER NOT NULL,
                novelty_score REAL NOT NULL,
                advocate_argument TEXT NOT NULL DEFAULT '',
                critic_argument TEXT NOT NULL DEFAULT '',
                prior_art_json TEXT DEFAULT '[]',
                recommendation TEXT NOT NULL DEFAULT '',
                models_json TEXT DEFAULT '{}',
                linked_hypothesis_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_sandbox_created
                ON sandbox_executions(created_at);
            CREATE INDEX IF NOT EXISTS idx_sandbox_experiment
                ON sandbox_executions(linked_experiment_id);
            CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
            CREATE INDEX IF NOT EXISTS idx_plan_steps_plan ON plan_steps(plan_id);
            CREATE INDEX IF NOT EXISTS idx_plan_steps_status ON plan_steps(status);
            CREATE INDEX IF NOT EXISTS idx_plans_status ON research_plans(status);
            CREATE INDEX IF NOT EXISTS idx_novelty_created ON novelty_checks(created_at);
        """)
        self._conn.commit()

    def _migrate_to_v5(self) -> None:
        """Add experiment loops, statistics, verification, vision, papers,
        plan branches, debates, and meta-learning tables."""
        self._conn.executescript("""
            -- Experiment loops (closed-loop iteration)
            CREATE TABLE IF NOT EXISTS experiment_loops (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                parameter_space_json TEXT NOT NULL DEFAULT '{}',
                objective TEXT NOT NULL DEFAULT 'minimize',
                target_metric TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                best_params_json TEXT DEFAULT '{}',
                best_value REAL,
                total_trials INTEGER NOT NULL DEFAULT 0,
                linked_hypothesis_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS loop_trials (
                id TEXT PRIMARY KEY,
                loop_id TEXT NOT NULL REFERENCES experiment_loops(id),
                trial_number INTEGER NOT NULL,
                params_json TEXT NOT NULL DEFAULT '{}',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                target_value REAL,
                execution_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Statistical tests
            CREATE TABLE IF NOT EXISTS statistical_tests (
                id TEXT PRIMARY KEY,
                test_type TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                input_data_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                passed INTEGER NOT NULL DEFAULT 0,
                p_value REAL,
                confidence_level REAL NOT NULL DEFAULT 0.95,
                linked_finding_id TEXT,
                linked_execution_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Verification runs
            CREATE TABLE IF NOT EXISTS verification_runs (
                id TEXT PRIMARY KEY,
                original_execution_id TEXT NOT NULL,
                verification_type TEXT NOT NULL DEFAULT 'reproducibility',
                status TEXT NOT NULL DEFAULT 'pending',
                original_metrics_json TEXT DEFAULT '{}',
                reproduced_metrics_json TEXT DEFAULT '{}',
                match_score REAL,
                discrepancies_json TEXT DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Visual analyses
            CREATE TABLE IF NOT EXISTS visual_analyses (
                id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL,
                analysis_type TEXT NOT NULL DEFAULT 'interpret',
                prompt TEXT NOT NULL DEFAULT '',
                interpretation TEXT NOT NULL DEFAULT '',
                findings_json TEXT DEFAULT '[]',
                model_used TEXT NOT NULL DEFAULT '',
                linked_execution_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Paper drafts and reviews
            CREATE TABLE IF NOT EXISTS paper_drafts (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                format TEXT NOT NULL DEFAULT 'markdown',
                version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'draft',
                linked_plan_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS paper_reviews (
                id TEXT PRIMARY KEY,
                draft_id TEXT NOT NULL REFERENCES paper_drafts(id),
                reviewer_model TEXT NOT NULL,
                overall_score REAL NOT NULL DEFAULT 0,
                scores_json TEXT DEFAULT '{}',
                strengths TEXT NOT NULL DEFAULT '',
                weaknesses TEXT NOT NULL DEFAULT '',
                suggestions TEXT NOT NULL DEFAULT '',
                decision TEXT NOT NULL DEFAULT 'revise',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Plan branches (conditional execution)
            CREATE TABLE IF NOT EXISTS plan_branches (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL REFERENCES research_plans(id),
                from_step_id TEXT NOT NULL REFERENCES plan_steps(id),
                condition TEXT NOT NULL,
                condition_type TEXT NOT NULL DEFAULT 'metric_threshold',
                then_steps_json TEXT DEFAULT '[]',
                else_steps_json TEXT DEFAULT '[]',
                evaluated INTEGER NOT NULL DEFAULT 0,
                result INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Debate sessions (adversarial verification)
            CREATE TABLE IF NOT EXISTS debate_sessions (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                debate_type TEXT NOT NULL DEFAULT 'verification',
                status TEXT NOT NULL DEFAULT 'active',
                verdict TEXT,
                verdict_confidence REAL,
                verdict_reasoning TEXT NOT NULL DEFAULT '',
                rounds INTEGER NOT NULL DEFAULT 0,
                max_rounds INTEGER NOT NULL DEFAULT 3,
                linked_finding_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS debate_arguments (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES debate_sessions(id),
                role TEXT NOT NULL,
                model TEXT NOT NULL,
                argument TEXT NOT NULL,
                evidence_json TEXT DEFAULT '[]',
                round_number INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Meta-learning
            CREATE TABLE IF NOT EXISTS meta_patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                context_json TEXT DEFAULT '{}',
                outcome TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                times_observed INTEGER NOT NULL DEFAULT 1,
                last_observed TEXT NOT NULL DEFAULT (datetime('now')),
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS strategy_log (
                id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT '',
                success INTEGER NOT NULL DEFAULT 0,
                metrics_json TEXT DEFAULT '{}',
                lessons TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_loop_trials_loop
                ON loop_trials(loop_id);
            CREATE INDEX IF NOT EXISTS idx_stat_tests_finding
                ON statistical_tests(linked_finding_id);
            CREATE INDEX IF NOT EXISTS idx_verification_original
                ON verification_runs(original_execution_id);
            CREATE INDEX IF NOT EXISTS idx_visual_execution
                ON visual_analyses(linked_execution_id);
            CREATE INDEX IF NOT EXISTS idx_paper_reviews_draft
                ON paper_reviews(draft_id);
            CREATE INDEX IF NOT EXISTS idx_plan_branches_plan
                ON plan_branches(plan_id);
            CREATE INDEX IF NOT EXISTS idx_debate_args_session
                ON debate_arguments(session_id);
            CREATE INDEX IF NOT EXISTS idx_meta_patterns_type
                ON meta_patterns(pattern_type);
            CREATE INDEX IF NOT EXISTS idx_strategy_log_created
                ON strategy_log(created_at);
        """)
        self._conn.commit()

    def _migrate_to_v6(self) -> None:
        """Add causal reasoning, benchmarks, learned tools, safety flags,
        agent roles, coordination tasks, and physics simulations tables."""
        self._conn.executescript("""
            -- Causal reasoning
            CREATE TABLE IF NOT EXISTS causal_graphs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                nodes_json TEXT NOT NULL DEFAULT '[]',
                edges_json TEXT NOT NULL DEFAULT '[]',
                confounders_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'draft',
                linked_hypothesis_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS causal_estimates (
                id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL REFERENCES causal_graphs(id),
                method TEXT NOT NULL,
                treatment TEXT NOT NULL,
                outcome TEXT NOT NULL,
                estimate REAL,
                ci_lower REAL,
                ci_upper REAL,
                p_value REAL,
                interpretation TEXT NOT NULL DEFAULT '',
                execution_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Benchmarks
            CREATE TABLE IF NOT EXISTS benchmarks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                metric TEXT NOT NULL,
                ground_truth_value REAL NOT NULL,
                tolerance REAL NOT NULL DEFAULT 0.05,
                source TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                domain TEXT NOT NULL DEFAULT 'climate',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id TEXT PRIMARY KEY,
                benchmark_id TEXT NOT NULL REFERENCES benchmarks(id),
                measured_value REAL NOT NULL,
                passed INTEGER NOT NULL,
                delta REAL NOT NULL,
                execution_id TEXT,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Learned tools
            CREATE TABLE IF NOT EXISTS learned_tools (
                id TEXT PRIMARY KEY,
                package_name TEXT NOT NULL UNIQUE,
                version TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                capabilities_json TEXT NOT NULL DEFAULT '[]',
                use_cases_json TEXT NOT NULL DEFAULT '[]',
                example_code TEXT NOT NULL DEFAULT '',
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_used_at TEXT,
                discovered_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Safety flags
            CREATE TABLE IF NOT EXISTS safety_flags (
                id TEXT PRIMARY KEY,
                flag_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'warning',
                description TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                related_ids_json TEXT NOT NULL DEFAULT '[]',
                dismissed INTEGER NOT NULL DEFAULT 0,
                dismissed_reason TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Multi-agent coordination
            CREATE TABLE IF NOT EXISTS agent_roles (
                child_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                specialization TEXT NOT NULL DEFAULT '',
                genesis_prompt TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS coordination_tasks (
                id TEXT PRIMARY KEY,
                child_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result_json TEXT DEFAULT '{}',
                assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT
            );

            -- Physics simulations
            CREATE TABLE IF NOT EXISTS physics_sims (
                id TEXT PRIMARY KEY,
                sim_type TEXT NOT NULL,
                parameters_json TEXT NOT NULL DEFAULT '{}',
                execution_id TEXT,
                summary TEXT NOT NULL DEFAULT '',
                artifacts_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_causal_graphs_hypothesis
                ON causal_graphs(linked_hypothesis_id);
            CREATE INDEX IF NOT EXISTS idx_causal_estimates_graph
                ON causal_estimates(graph_id);
            CREATE INDEX IF NOT EXISTS idx_benchmark_runs_benchmark
                ON benchmark_runs(benchmark_id);
            CREATE INDEX IF NOT EXISTS idx_safety_flags_type
                ON safety_flags(flag_type);
            CREATE INDEX IF NOT EXISTS idx_safety_flags_dismissed
                ON safety_flags(dismissed);
            CREATE INDEX IF NOT EXISTS idx_coord_tasks_child
                ON coordination_tasks(child_id);
            CREATE INDEX IF NOT EXISTS idx_coord_tasks_status
                ON coordination_tasks(status);
            CREATE INDEX IF NOT EXISTS idx_physics_sims_type
                ON physics_sims(sim_type);
        """)
        self._conn.commit()

    def _migrate_to_v7(self) -> None:
        """Add full-text paper reading, arXiv/S2 IDs, paper_files cache."""
        # New columns on papers
        for col_sql in [
            "ALTER TABLE papers ADD COLUMN full_text TEXT DEFAULT ''",
            "ALTER TABLE papers ADD COLUMN arxiv_id TEXT",
            "ALTER TABLE papers ADD COLUMN s2_id TEXT",
            "ALTER TABLE papers ADD COLUMN pdf_url TEXT",
            "ALTER TABLE papers ADD COLUMN full_text_fetched_at TEXT",
        ]:
            try:
                self._conn.execute(col_sql)
            except sqlite3.OperationalError:
                pass  # column already exists

        self._conn.executescript("""
            -- Paper file cache
            CREATE TABLE IF NOT EXISTS paper_files (
                paper_id TEXT NOT NULL,
                file_type TEXT NOT NULL,
                local_path TEXT NOT NULL,
                sha256 TEXT NOT NULL DEFAULT '',
                size_bytes INTEGER NOT NULL DEFAULT 0,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (paper_id, file_type)
            );

            CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
            CREATE INDEX IF NOT EXISTS idx_papers_s2 ON papers(s2_id);
            CREATE INDEX IF NOT EXISTS idx_paper_files_paper ON paper_files(paper_id);
        """)

        # Rebuild papers_fts with full_text column included.
        # Drop old FTS table + triggers, recreate with full_text.
        self._conn.executescript("""
            DROP TRIGGER IF EXISTS papers_ai;
            DROP TRIGGER IF EXISTS papers_ad;
            DROP TABLE IF EXISTS papers_fts;

            CREATE VIRTUAL TABLE papers_fts USING fts5(
                title, abstract, authors, full_text,
                content=papers, content_rowid=rowid
            );

            CREATE TRIGGER papers_ai AFTER INSERT ON papers BEGIN
                INSERT INTO papers_fts(rowid, title, abstract, authors, full_text)
                VALUES (new.rowid, new.title, new.abstract, new.authors, new.full_text);
            END;

            CREATE TRIGGER papers_ad AFTER DELETE ON papers BEGIN
                INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors, full_text)
                VALUES ('delete', old.rowid, old.title, old.abstract, old.authors, old.full_text);
            END;

            CREATE TRIGGER papers_au AFTER UPDATE ON papers BEGIN
                INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors, full_text)
                VALUES ('delete', old.rowid, old.title, old.abstract, old.authors, old.full_text);
                INSERT INTO papers_fts(rowid, title, abstract, authors, full_text)
                VALUES (new.rowid, new.title, new.abstract, new.authors, new.full_text);
            END;
        """)
        self._conn.execute(
            "INSERT INTO papers_fts(papers_fts) VALUES('rebuild')"
        )
        self._conn.commit()

    # -- Identity --------------------------------------------------------

    def get_identity(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM identity WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_identity(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO identity (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    # -- Turns -----------------------------------------------------------

    def insert_turn(self, turn: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO turns (id, timestamp, state, input, input_source, "
            "thinking, tool_calls, token_usage, cost_cents) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn["id"],
                turn["timestamp"],
                turn["state"],
                turn.get("input"),
                turn.get("input_source"),
                turn["thinking"],
                json.dumps(turn.get("tool_calls", [])),
                json.dumps(turn.get("token_usage", {})),
                turn.get("cost_cents", 0),
            ),
        )
        self._conn.commit()

    def get_recent_turns(self, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM turns ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_deserialize_turn(r) for r in reversed(rows)]

    def get_turn_by_id(self, turn_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM turns WHERE id = ?", (turn_id,)
        ).fetchone()
        return _deserialize_turn(row) if row else None

    def get_turn_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as c FROM turns").fetchone()
        return row["c"]

    # -- Tool Calls ------------------------------------------------------

    def insert_tool_call(self, turn_id: str, call: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO tool_calls (id, turn_id, name, arguments, result, "
            "duration_ms, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                call["id"],
                turn_id,
                call["name"],
                json.dumps(call.get("arguments", {})),
                call.get("result", ""),
                call.get("duration_ms", 0),
                call.get("error"),
            ),
        )
        self._conn.commit()

    def get_tool_calls_for_turn(self, turn_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM tool_calls WHERE turn_id = ?", (turn_id,)
        ).fetchall()
        return [_deserialize_tool_call(r) for r in rows]

    # -- Heartbeat -------------------------------------------------------

    def get_heartbeat_entries(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM heartbeat_entries").fetchall()
        return [_deserialize_heartbeat(r) for r in rows]

    def upsert_heartbeat_entry(self, entry: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO heartbeat_entries "
            "(name, schedule, task, enabled, last_run, next_run, params, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                entry["name"],
                entry["schedule"],
                entry["task"],
                1 if entry.get("enabled", True) else 0,
                entry.get("last_run"),
                entry.get("next_run"),
                json.dumps(entry.get("params", {})),
            ),
        )
        self._conn.commit()

    def update_heartbeat_last_run(self, name: str, timestamp: str) -> None:
        self._conn.execute(
            "UPDATE heartbeat_entries SET last_run = ?, updated_at = datetime('now') "
            "WHERE name = ?",
            (timestamp, name),
        )
        self._conn.commit()

    # -- Transactions ----------------------------------------------------

    def insert_transaction(self, txn: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO transactions (id, type, amount_cents, balance_after_cents, "
            "description) VALUES (?, ?, ?, ?, ?)",
            (
                txn["id"],
                txn["type"],
                txn.get("amount_cents"),
                txn.get("balance_after_cents"),
                txn.get("description", ""),
            ),
        )
        self._conn.commit()

    def get_recent_transactions(self, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM transactions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -- Modifications ---------------------------------------------------

    def insert_modification(self, mod: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO modifications (id, timestamp, type, description, "
            "file_path, diff, reversible) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                mod["id"],
                mod["timestamp"],
                mod["type"],
                mod["description"],
                mod.get("file_path"),
                mod.get("diff"),
                1 if mod.get("reversible", True) else 0,
            ),
        )
        self._conn.commit()

    def get_recent_modifications(self, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM modifications ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -- Key-Value Store -------------------------------------------------

    def get_kv(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_kv(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (key, value, updated_at) "
            "VALUES (?, ?, datetime('now'))",
            (key, value),
        )
        self._conn.commit()

    def delete_kv(self, key: str) -> None:
        self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
        self._conn.commit()

    # -- Agent State (via KV) --------------------------------------------

    def get_agent_state(self) -> str:
        return self.get_kv("agent_state") or "setup"

    def set_agent_state(self, state: str) -> None:
        self.set_kv("agent_state", state)

    # -- Skills ----------------------------------------------------------

    def get_skills(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM skills"
        if enabled_only:
            query += " WHERE enabled = 1"
        rows = self._conn.execute(query).fetchall()
        return [_deserialize_skill(r) for r in rows]

    def get_skill_by_name(self, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM skills WHERE name = ?", (name,)
        ).fetchone()
        return _deserialize_skill(row) if row else None

    def upsert_skill(self, skill: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO skills (name, description, auto_activate, "
            "requires, instructions, source, path, enabled, installed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                skill["name"],
                skill.get("description", ""),
                1 if skill.get("auto_activate", True) else 0,
                json.dumps(skill.get("requires", {})),
                skill.get("instructions", ""),
                skill.get("source", "builtin"),
                skill.get("path", ""),
                1 if skill.get("enabled", True) else 0,
                skill.get("installed_at", ""),
            ),
        )
        self._conn.commit()

    def remove_skill(self, name: str) -> None:
        self._conn.execute(
            "UPDATE skills SET enabled = 0 WHERE name = ?", (name,)
        )
        self._conn.commit()

    # -- Children --------------------------------------------------------

    def get_children(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM children ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_child_by_id(self, child_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM children WHERE id = ?", (child_id,)
        ).fetchone()
        return dict(row) if row else None

    def insert_child(self, child: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO children (id, name, topic, pid, config_path, status, "
            "budget_cents, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                child["id"],
                child["name"],
                child["topic"],
                child.get("pid"),
                child["config_path"],
                child.get("status", "spawning"),
                child.get("budget_cents", 0),
                child.get("created_at", ""),
            ),
        )
        self._conn.commit()

    def update_child_status(self, child_id: str, status: str, pid: int | None = None) -> None:
        if pid is not None:
            self._conn.execute(
                "UPDATE children SET status = ?, pid = ?, last_checked = datetime('now') "
                "WHERE id = ?",
                (status, pid, child_id),
            )
        else:
            self._conn.execute(
                "UPDATE children SET status = ?, last_checked = datetime('now') "
                "WHERE id = ?",
                (status, child_id),
            )
        self._conn.commit()

    # -- Inbox Messages --------------------------------------------------

    def insert_inbox_message(self, msg: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO inbox_messages "
            "(id, from_agent, content, received_at, reply_to) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                msg["id"],
                msg["from_agent"],
                msg["content"],
                msg.get("received_at", ""),
                msg.get("reply_to"),
            ),
        )
        self._conn.commit()

    def get_unprocessed_inbox(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM inbox_messages WHERE processed_at IS NULL "
            "ORDER BY received_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_inbox_processed(self, msg_id: str) -> None:
        self._conn.execute(
            "UPDATE inbox_messages SET processed_at = datetime('now') WHERE id = ?",
            (msg_id,),
        )
        self._conn.commit()

    # -- Knowledge -------------------------------------------------------

    def insert_knowledge(
        self,
        kid: str,
        topic: str,
        content: str,
        source: str,
        confidence: float,
        entry_type: str = "observation",
        parent_id: str | None = None,
        status: str = "active",
        metadata_json: str = "{}",
    ) -> None:
        self._conn.execute(
            "INSERT INTO knowledge (id, topic, content, source, confidence, "
            "entry_type, parent_id, status, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (kid, topic, content, source, confidence,
             entry_type, parent_id, status, metadata_json),
        )
        self._conn.commit()

    def search_knowledge(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        # Split query into individual words and match ANY of them.
        # Previous bug: LIKE '%multi word query%' requires exact phrase match.
        words = [w for w in query.strip().split() if len(w) >= 2]
        if not words:
            return self.get_recent_knowledge(limit)

        conditions: list[str] = []
        params: list[Any] = []
        for word in words:
            pattern = f"%{word}%"
            conditions.append("(topic LIKE ? OR content LIKE ?)")
            params.extend([pattern, pattern])

        where = " OR ".join(conditions)
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM knowledge WHERE {where} "
            "ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_knowledge(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM knowledge ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_knowledge_by_id(self, kid: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE id = ?", (kid,)
        ).fetchone()
        return dict(row) if row else None

    def update_knowledge_status(self, kid: str, status: str) -> None:
        self._conn.execute(
            "UPDATE knowledge SET status = ? WHERE id = ?",
            (status, kid),
        )
        self._conn.commit()

    def search_knowledge_fts(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """FTS5 MATCH query on knowledge, ranked by bm25."""
        # Sanitize query for FTS5: wrap each word in double quotes
        words = [w for w in query.strip().split() if len(w) >= 2]
        if not words:
            return self.get_recent_knowledge(limit)
        fts_query = " OR ".join(f'"{w}"' for w in words)
        try:
            rows = self._conn.execute(
                "SELECT k.* FROM knowledge k "
                "JOIN knowledge_fts f ON k.rowid = f.rowid "
                "WHERE knowledge_fts MATCH ? "
                "ORDER BY bm25(knowledge_fts) "
                "LIMIT ?",
                (fts_query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS5 table might not have content yet; fall back to LIKE
            return self.search_knowledge(query, limit)

    def get_knowledge_chain(self, kid: str) -> list[dict[str, Any]]:
        """Walk parent_id links to get the full chain containing this entry."""
        # Walk up to root
        chain: list[dict[str, Any]] = []
        current_id: str | None = kid
        seen: set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            entry = self.get_knowledge_by_id(current_id)
            if not entry:
                break
            chain.insert(0, entry)
            current_id = entry.get("parent_id")

        # From root, get all descendants
        root_id = chain[0]["id"] if chain else kid
        descendants = self._get_descendants(root_id)
        seen_ids = {e["id"] for e in chain}
        for d in descendants:
            if d["id"] not in seen_ids:
                chain.append(d)
                seen_ids.add(d["id"])
        return chain

    def _get_descendants(self, parent_id: str) -> list[dict[str, Any]]:
        """Recursively get all children of a knowledge entry."""
        rows = self._conn.execute(
            "SELECT * FROM knowledge WHERE parent_id = ? ORDER BY created_at",
            (parent_id,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            entry = dict(row)
            result.append(entry)
            result.extend(self._get_descendants(entry["id"]))
        return result

    def get_knowledge_by_type(
        self, entry_type: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM knowledge WHERE entry_type = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (entry_type, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_active_hypotheses(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM knowledge WHERE entry_type = 'hypothesis' "
            "AND status = 'active' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Papers ----------------------------------------------------------

    def insert_paper(self, paper: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO papers "
            "(id, title, authors, year, doi, abstract, source, url, cited_by_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paper["id"],
                paper.get("title", ""),
                paper.get("authors", ""),
                paper.get("year"),
                paper.get("doi"),
                paper.get("abstract", ""),
                paper.get("source", "openalex"),
                paper.get("url"),
                paper.get("cited_by_count", 0),
            ),
        )
        self._conn.commit()

    def search_papers_fts(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """FTS5 MATCH query on papers, ranked by bm25."""
        words = [w for w in query.strip().split() if len(w) >= 2]
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words)
        try:
            rows = self._conn.execute(
                "SELECT p.* FROM papers p "
                "JOIN papers_fts f ON p.rowid = f.rowid "
                "WHERE papers_fts MATCH ? "
                "ORDER BY bm25(papers_fts) "
                "LIMIT ?",
                (fts_query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def get_paper_by_doi(self, doi: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE doi = ?", (doi,)
        ).fetchone()
        return dict(row) if row else None

    def rebuild_knowledge_fts(self) -> None:
        """Rebuild the knowledge FTS5 index from the knowledge table."""
        self._conn.execute(
            "INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')"
        )
        self._conn.commit()

    def rebuild_papers_fts(self) -> None:
        """Rebuild the papers FTS5 index from the papers table."""
        self._conn.execute(
            "INSERT INTO papers_fts(papers_fts) VALUES('rebuild')"
        )
        self._conn.commit()

    # -- Citations -------------------------------------------------------

    def add_citation(
        self, knowledge_id: str, paper_id: str, context: str = ""
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO knowledge_citations "
            "(knowledge_id, paper_id, context) VALUES (?, ?, ?)",
            (knowledge_id, paper_id, context),
        )
        self._conn.commit()

    def get_citations_for_knowledge(
        self, knowledge_id: str
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT kc.*, p.title, p.authors, p.year, p.doi "
            "FROM knowledge_citations kc "
            "JOIN papers p ON kc.paper_id = p.id "
            "WHERE kc.knowledge_id = ?",
            (knowledge_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_knowledge_citing_paper(
        self, paper_id: str
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT k.* FROM knowledge k "
            "JOIN knowledge_citations kc ON k.id = kc.knowledge_id "
            "WHERE kc.paper_id = ?",
            (paper_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Embeddings ------------------------------------------------------

    def update_knowledge_embedding(
        self, kid: str, embedding_blob: bytes
    ) -> None:
        self._conn.execute(
            "UPDATE knowledge SET embedding = ? WHERE id = ?",
            (embedding_blob, kid),
        )
        self._conn.commit()

    def update_paper_embedding(
        self, paper_id: str, embedding_blob: bytes
    ) -> None:
        self._conn.execute(
            "UPDATE papers SET embedding = ? WHERE id = ?",
            (embedding_blob, paper_id),
        )
        self._conn.commit()

    def get_all_knowledge_embeddings(self) -> list[tuple[str, bytes]]:
        rows = self._conn.execute(
            "SELECT id, embedding FROM knowledge WHERE embedding IS NOT NULL"
        ).fetchall()
        return [(r["id"], r["embedding"]) for r in rows]

    def get_all_paper_embeddings(self) -> list[tuple[str, bytes]]:
        rows = self._conn.execute(
            "SELECT id, embedding FROM papers WHERE embedding IS NOT NULL"
        ).fetchall()
        return [(r["id"], r["embedding"]) for r in rows]

    def get_unembedded_knowledge(
        self, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM knowledge WHERE embedding IS NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_unembedded_papers(
        self, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM papers WHERE embedding IS NULL "
            "ORDER BY fetched_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Stale Hypotheses ------------------------------------------------

    def get_stale_hypotheses(
        self, older_than_hours: int = 168
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM knowledge "
            "WHERE entry_type = 'hypothesis' AND status = 'active' "
            "AND created_at < datetime('now', ?)",
            (f"-{older_than_hours} hours",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_hypotheses_with_findings_no_conclusion(
        self,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute("""
            SELECT h.* FROM knowledge h
            WHERE h.entry_type = 'hypothesis' AND h.status = 'active'
            AND EXISTS (
                SELECT 1 FROM knowledge f
                WHERE f.entry_type = 'finding'
                AND f.parent_id IN (
                    SELECT e.id FROM knowledge e
                    WHERE e.entry_type = 'experiment' AND e.parent_id = h.id
                )
            )
            AND NOT EXISTS (
                SELECT 1 FROM knowledge c
                WHERE c.entry_type = 'conclusion' AND c.parent_id = h.id
            )
        """).fetchall()
        return [dict(r) for r in rows]

    def get_paper_by_id(self, paper_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE id = ?", (paper_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_paper_full_text(
        self, paper_id: str, full_text: str
    ) -> None:
        self._conn.execute(
            "UPDATE papers SET full_text = ?, full_text_fetched_at = datetime('now') "
            "WHERE id = ?",
            (full_text, paper_id),
        )
        self._conn.commit()

    def update_paper_ids(self, paper_id: str, **kwargs: Any) -> None:
        """Set arxiv_id, s2_id, pdf_url on a paper. Only updates non-None kwargs."""
        allowed = {"arxiv_id", "s2_id", "pdf_url"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [paper_id]
        self._conn.execute(
            f"UPDATE papers SET {set_clause} WHERE id = ?", vals
        )
        self._conn.commit()

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_paper_by_s2_id(self, s2_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE s2_id = ?", (s2_id,)
        ).fetchone()
        return dict(row) if row else None

    def insert_paper_file(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO paper_files "
            "(paper_id, file_type, local_path, sha256, size_bytes, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (
                record["paper_id"],
                record["file_type"],
                record["local_path"],
                record.get("sha256", ""),
                record.get("size_bytes", 0),
            ),
        )
        self._conn.commit()

    def get_paper_file(
        self, paper_id: str, file_type: str
    ) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM paper_files WHERE paper_id = ? AND file_type = ?",
            (paper_id, file_type),
        ).fetchone()
        return dict(row) if row else None

    def get_papers_without_full_text(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM papers "
            "WHERE pdf_url IS NOT NULL AND pdf_url != '' "
            "AND (full_text IS NULL OR full_text = '') "
            "ORDER BY fetched_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_papers_fulltext_fts(
        self, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """FTS5 MATCH over title + abstract + authors + full_text."""
        words = [w for w in query.strip().split() if len(w) >= 2]
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words)
        try:
            rows = self._conn.execute(
                "SELECT p.* FROM papers p "
                "JOIN papers_fts f ON p.rowid = f.rowid "
                "WHERE papers_fts MATCH ? "
                "ORDER BY bm25(papers_fts) "
                "LIMIT ?",
                (fts_query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return self.search_papers_fts(query, limit)

    # -- Research Cycles -------------------------------------------------

    def insert_research_cycle(self, cycle: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO research_cycles (id, topic, phase, status, plan_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                cycle["id"],
                cycle["topic"],
                cycle.get("phase", "planning"),
                cycle.get("status", "active"),
                json.dumps(cycle.get("plan", {})),
            ),
        )
        self._conn.commit()

    def update_research_cycle(
        self, cycle_id: str, **updates: Any
    ) -> None:
        sets: list[str] = ["updated_at = datetime('now')"]
        params: list[Any] = []
        for key, val in updates.items():
            if key == "plan":
                sets.append("plan_json = ?")
                params.append(json.dumps(val))
            elif key in ("phase", "status", "results_summary"):
                sets.append(f"{key} = ?")
                params.append(val)
        params.append(cycle_id)
        self._conn.execute(
            f"UPDATE research_cycles SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        self._conn.commit()

    def get_research_cycles(
        self, status: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM research_cycles WHERE status = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM research_cycles ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_deserialize_research_cycle(r) for r in rows]

    # -- Peers -----------------------------------------------------------

    def upsert_peer(self, peer: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO peers "
            "(name, url, public_key, topic, last_seen, trust_score) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                peer["name"],
                peer["url"],
                peer.get("public_key"),
                peer.get("topic"),
                peer.get("last_seen"),
                peer.get("trust_score", 0.5),
            ),
        )
        self._conn.commit()

    def get_peers(self, topic: str | None = None) -> list[dict[str, Any]]:
        if topic:
            rows = self._conn.execute(
                "SELECT * FROM peers WHERE topic LIKE ? ORDER BY trust_score DESC",
                (f"%{topic}%",),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM peers ORDER BY trust_score DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_peer_by_name(self, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM peers WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    # -- Reviews ---------------------------------------------------------

    def insert_review(self, review: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO reviews (id, from_agent, knowledge_id, score, comment) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                review["id"],
                review["from_agent"],
                review["knowledge_id"],
                review["score"],
                review.get("comment", ""),
            ),
        )
        self._conn.commit()

    def get_reviews_for_knowledge(self, knowledge_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM reviews WHERE knowledge_id = ? ORDER BY created_at DESC",
            (knowledge_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Sandbox Executions ----------------------------------------------

    def insert_sandbox_execution(self, execution: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO sandbox_executions "
            "(id, script_hash, description, backend, returncode, stdout_preview, "
            "stderr_preview, metrics_json, artifacts_json, duration_s, work_dir, "
            "linked_experiment_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                execution["id"],
                execution["script_hash"],
                execution.get("description", ""),
                execution["backend"],
                execution["returncode"],
                execution.get("stdout_preview", ""),
                execution.get("stderr_preview", ""),
                json.dumps(execution.get("metrics", {})),
                json.dumps(execution.get("artifacts", [])),
                execution.get("duration_s", 0),
                execution["work_dir"],
                execution.get("linked_experiment_id"),
            ),
        )
        self._conn.commit()

    def get_sandbox_execution(self, execution_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM sandbox_executions WHERE id = ?", (execution_id,)
        ).fetchone()
        if not row:
            return None
        return _deserialize_sandbox_execution(row)

    def get_recent_sandbox_executions(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM sandbox_executions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_deserialize_sandbox_execution(r) for r in rows]

    # -- Datasets --------------------------------------------------------

    def insert_dataset(self, record: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO datasets "
            "(id, name, source_url, format, local_path, sha256, size_bytes, "
            "description, verified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["id"],
                record["name"],
                record["source_url"],
                record.get("format", "csv"),
                record["local_path"],
                record["sha256"],
                record.get("size_bytes", 0),
                record.get("description", ""),
                1 if record.get("verified", True) else 0,
            ),
        )
        self._conn.commit()

    def get_dataset_by_url(self, url: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM datasets WHERE source_url = ?", (url,)
        ).fetchone()
        return dict(row) if row else None

    def get_dataset_by_name(self, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM datasets WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_datasets(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM datasets ORDER BY fetched_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Research Plans --------------------------------------------------

    def insert_research_plan(self, plan: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO research_plans (id, title, description, status) "
            "VALUES (?, ?, ?, ?)",
            (
                plan["id"],
                plan["title"],
                plan.get("description", ""),
                plan.get("status", "draft"),
            ),
        )
        self._conn.commit()

    def get_research_plan(self, plan_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM research_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_research_plan(self, plan_id: str, **updates: Any) -> None:
        sets: list[str] = ["updated_at = datetime('now')"]
        params: list[Any] = []
        for key, val in updates.items():
            if key in ("title", "description", "status"):
                sets.append(f"{key} = ?")
                params.append(val)
        params.append(plan_id)
        self._conn.execute(
            f"UPDATE research_plans SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        self._conn.commit()

    def get_research_plans_by_status(
        self, status: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM research_plans WHERE status = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM research_plans ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- Plan Steps ------------------------------------------------------

    def insert_plan_step(self, step: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO plan_steps "
            "(id, plan_id, title, description, step_type, status, step_order, "
            "depends_on_json, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                step["id"],
                step["plan_id"],
                step["title"],
                step.get("description", ""),
                step.get("step_type", "custom"),
                step.get("status", "pending"),
                step.get("step_order", 0),
                json.dumps(step.get("depends_on", [])),
                step.get("notes", ""),
            ),
        )
        self._conn.commit()

    def get_plan_steps(self, plan_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY step_order",
            (plan_id,),
        ).fetchall()
        return [_deserialize_plan_step(r) for r in rows]

    def get_plan_step(self, step_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM plan_steps WHERE id = ?", (step_id,)
        ).fetchone()
        return _deserialize_plan_step(row) if row else None

    def update_plan_step(self, step_id: str, **updates: Any) -> None:
        sets: list[str] = []
        params: list[Any] = []
        for key, val in updates.items():
            if key in ("status", "notes", "started_at", "completed_at"):
                sets.append(f"{key} = ?")
                params.append(val)
        if not sets:
            return
        params.append(step_id)
        self._conn.execute(
            f"UPDATE plan_steps SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        self._conn.commit()

    # -- Plan Step Links -------------------------------------------------

    def insert_plan_step_link(
        self, step_id: str, link_type: str, target_id: str
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO plan_step_links "
            "(step_id, link_type, target_id) VALUES (?, ?, ?)",
            (step_id, link_type, target_id),
        )
        self._conn.commit()

    def get_plan_step_links(self, step_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM plan_step_links WHERE step_id = ?", (step_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Novelty Checks --------------------------------------------------

    def insert_novelty_check(self, check: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO novelty_checks "
            "(id, idea_text, idea_type, is_novel, novelty_score, "
            "advocate_argument, critic_argument, prior_art_json, "
            "recommendation, models_json, linked_hypothesis_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                check["id"],
                check["idea_text"],
                check.get("idea_type", "hypothesis"),
                1 if check["is_novel"] else 0,
                check["novelty_score"],
                check.get("advocate_argument", ""),
                check.get("critic_argument", ""),
                json.dumps(check.get("prior_art", [])),
                check.get("recommendation", ""),
                json.dumps(check.get("models", {})),
                check.get("linked_hypothesis_id"),
            ),
        )
        self._conn.commit()

    def get_novelty_check(self, check_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM novelty_checks WHERE id = ?", (check_id,)
        ).fetchone()
        return _deserialize_novelty_check(row) if row else None

    def get_novelty_for_hypothesis(
        self, hypothesis_id: str
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM novelty_checks WHERE linked_hypothesis_id = ? "
            "ORDER BY created_at DESC",
            (hypothesis_id,),
        ).fetchall()
        return [_deserialize_novelty_check(r) for r in rows]

    # -- Experiment Loops ------------------------------------------------

    def insert_experiment_loop(self, loop: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO experiment_loops "
            "(id, name, description, parameter_space_json, objective, "
            "target_metric, status, linked_hypothesis_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                loop["id"], loop["name"], loop.get("description", ""),
                json.dumps(loop.get("parameter_space", {})),
                loop.get("objective", "minimize"),
                loop["target_metric"],
                loop.get("status", "active"),
                loop.get("linked_hypothesis_id"),
            ),
        )
        self._conn.commit()

    def get_experiment_loop(self, loop_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM experiment_loops WHERE id = ?", (loop_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["parameter_space"] = json.loads(d.pop("parameter_space_json", "{}"))
        d["best_params"] = json.loads(d.pop("best_params_json", "{}"))
        return d

    def update_experiment_loop(self, loop_id: str, **updates: Any) -> None:
        sets: list[str] = ["updated_at = datetime('now')"]
        params: list[Any] = []
        for key, val in updates.items():
            if key == "best_params":
                sets.append("best_params_json = ?")
                params.append(json.dumps(val))
            elif key in ("status", "best_value", "total_trials"):
                sets.append(f"{key} = ?")
                params.append(val)
        params.append(loop_id)
        self._conn.execute(
            f"UPDATE experiment_loops SET {', '.join(sets)} WHERE id = ?", params
        )
        self._conn.commit()

    def get_active_experiment_loops(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM experiment_loops WHERE status = 'active' "
            "ORDER BY updated_at DESC"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["parameter_space"] = json.loads(d.pop("parameter_space_json", "{}"))
            d["best_params"] = json.loads(d.pop("best_params_json", "{}"))
            result.append(d)
        return result

    # -- Loop Trials -----------------------------------------------------

    def insert_loop_trial(self, trial: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO loop_trials "
            "(id, loop_id, trial_number, params_json, metrics_json, "
            "target_value, execution_id, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trial["id"], trial["loop_id"], trial["trial_number"],
                json.dumps(trial.get("params", {})),
                json.dumps(trial.get("metrics", {})),
                trial.get("target_value"),
                trial.get("execution_id"),
                trial.get("status", "pending"),
            ),
        )
        self._conn.commit()

    def get_loop_trials(self, loop_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM loop_trials WHERE loop_id = ? ORDER BY trial_number",
            (loop_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["params"] = json.loads(d.pop("params_json", "{}"))
            d["metrics"] = json.loads(d.pop("metrics_json", "{}"))
            result.append(d)
        return result

    def update_loop_trial(self, trial_id: str, **updates: Any) -> None:
        sets: list[str] = []
        params: list[Any] = []
        for key, val in updates.items():
            if key == "params":
                sets.append("params_json = ?")
                params.append(json.dumps(val))
            elif key == "metrics":
                sets.append("metrics_json = ?")
                params.append(json.dumps(val))
            elif key in ("target_value", "execution_id", "status"):
                sets.append(f"{key} = ?")
                params.append(val)
        if not sets:
            return
        params.append(trial_id)
        self._conn.execute(
            f"UPDATE loop_trials SET {', '.join(sets)} WHERE id = ?", params
        )
        self._conn.commit()

    # -- Statistical Tests -----------------------------------------------

    def insert_statistical_test(self, test: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO statistical_tests "
            "(id, test_type, description, input_data_json, result_json, "
            "passed, p_value, confidence_level, linked_finding_id, "
            "linked_execution_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                test["id"], test["test_type"],
                test.get("description", ""),
                json.dumps(test.get("input_data", {})),
                json.dumps(test.get("result", {})),
                1 if test.get("passed") else 0,
                test.get("p_value"),
                test.get("confidence_level", 0.95),
                test.get("linked_finding_id"),
                test.get("linked_execution_id"),
            ),
        )
        self._conn.commit()

    def get_statistical_test(self, test_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM statistical_tests WHERE id = ?", (test_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["input_data"] = json.loads(d.pop("input_data_json", "{}"))
        d["result"] = json.loads(d.pop("result_json", "{}"))
        d["passed"] = bool(d["passed"])
        return d

    def get_tests_for_finding(self, finding_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM statistical_tests WHERE linked_finding_id = ? "
            "ORDER BY created_at DESC", (finding_id,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["input_data"] = json.loads(d.pop("input_data_json", "{}"))
            d["result"] = json.loads(d.pop("result_json", "{}"))
            d["passed"] = bool(d["passed"])
            result.append(d)
        return result

    # -- Verification Runs -----------------------------------------------

    def insert_verification_run(self, run: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO verification_runs "
            "(id, original_execution_id, verification_type, status, "
            "original_metrics_json, reproduced_metrics_json, match_score, "
            "discrepancies_json, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run["id"], run["original_execution_id"],
                run.get("verification_type", "reproducibility"),
                run.get("status", "pending"),
                json.dumps(run.get("original_metrics", {})),
                json.dumps(run.get("reproduced_metrics", {})),
                run.get("match_score"),
                json.dumps(run.get("discrepancies", [])),
                run.get("notes", ""),
            ),
        )
        self._conn.commit()

    def get_verification_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM verification_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["original_metrics"] = json.loads(d.pop("original_metrics_json", "{}"))
        d["reproduced_metrics"] = json.loads(d.pop("reproduced_metrics_json", "{}"))
        d["discrepancies"] = json.loads(d.pop("discrepancies_json", "[]"))
        return d

    def update_verification_run(self, run_id: str, **updates: Any) -> None:
        sets: list[str] = []
        params: list[Any] = []
        for key, val in updates.items():
            if key == "reproduced_metrics":
                sets.append("reproduced_metrics_json = ?")
                params.append(json.dumps(val))
            elif key == "discrepancies":
                sets.append("discrepancies_json = ?")
                params.append(json.dumps(val))
            elif key in ("status", "match_score", "notes"):
                sets.append(f"{key} = ?")
                params.append(val)
        if not sets:
            return
        params.append(run_id)
        self._conn.execute(
            f"UPDATE verification_runs SET {', '.join(sets)} WHERE id = ?", params
        )
        self._conn.commit()

    def get_verifications_for_execution(
        self, execution_id: str
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM verification_runs WHERE original_execution_id = ? "
            "ORDER BY created_at DESC", (execution_id,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["original_metrics"] = json.loads(d.pop("original_metrics_json", "{}"))
            d["reproduced_metrics"] = json.loads(d.pop("reproduced_metrics_json", "{}"))
            d["discrepancies"] = json.loads(d.pop("discrepancies_json", "[]"))
            result.append(d)
        return result

    # -- Visual Analyses -------------------------------------------------

    def insert_visual_analysis(self, analysis: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO visual_analyses "
            "(id, image_path, analysis_type, prompt, interpretation, "
            "findings_json, model_used, linked_execution_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis["id"], analysis["image_path"],
                analysis.get("analysis_type", "interpret"),
                analysis.get("prompt", ""),
                analysis.get("interpretation", ""),
                json.dumps(analysis.get("findings", [])),
                analysis.get("model_used", ""),
                analysis.get("linked_execution_id"),
            ),
        )
        self._conn.commit()

    def get_visual_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM visual_analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["findings"] = json.loads(d.pop("findings_json", "[]"))
        return d

    # -- Paper Drafts ----------------------------------------------------

    def insert_paper_draft(self, draft: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO paper_drafts "
            "(id, title, abstract, content, format, version, status, "
            "linked_plan_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                draft["id"], draft["title"],
                draft.get("abstract", ""),
                draft.get("content", ""),
                draft.get("format", "markdown"),
                draft.get("version", 1),
                draft.get("status", "draft"),
                draft.get("linked_plan_id"),
            ),
        )
        self._conn.commit()

    def get_paper_draft(self, draft_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM paper_drafts WHERE id = ?", (draft_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_paper_draft(self, draft_id: str, **updates: Any) -> None:
        sets: list[str] = ["updated_at = datetime('now')"]
        params: list[Any] = []
        for key, val in updates.items():
            if key in ("title", "abstract", "content", "format", "version",
                       "status"):
                sets.append(f"{key} = ?")
                params.append(val)
        params.append(draft_id)
        self._conn.execute(
            f"UPDATE paper_drafts SET {', '.join(sets)} WHERE id = ?", params
        )
        self._conn.commit()

    def get_recent_paper_drafts(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM paper_drafts ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Paper Reviews ---------------------------------------------------

    def insert_paper_review(self, review: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO paper_reviews "
            "(id, draft_id, reviewer_model, overall_score, scores_json, "
            "strengths, weaknesses, suggestions, decision) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                review["id"], review["draft_id"],
                review["reviewer_model"],
                review.get("overall_score", 0),
                json.dumps(review.get("scores", {})),
                review.get("strengths", ""),
                review.get("weaknesses", ""),
                review.get("suggestions", ""),
                review.get("decision", "revise"),
            ),
        )
        self._conn.commit()

    def get_reviews_for_draft(self, draft_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM paper_reviews WHERE draft_id = ? "
            "ORDER BY created_at DESC", (draft_id,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["scores"] = json.loads(d.pop("scores_json", "{}"))
            result.append(d)
        return result

    # -- Plan Branches ---------------------------------------------------

    def insert_plan_branch(self, branch: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO plan_branches "
            "(id, plan_id, from_step_id, condition, condition_type, "
            "then_steps_json, else_steps_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                branch["id"], branch["plan_id"], branch["from_step_id"],
                branch["condition"],
                branch.get("condition_type", "metric_threshold"),
                json.dumps(branch.get("then_steps", [])),
                json.dumps(branch.get("else_steps", [])),
            ),
        )
        self._conn.commit()

    def get_plan_branches(self, plan_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM plan_branches WHERE plan_id = ? "
            "ORDER BY created_at", (plan_id,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["then_steps"] = json.loads(d.pop("then_steps_json", "[]"))
            d["else_steps"] = json.loads(d.pop("else_steps_json", "[]"))
            result.append(d)
        return result

    def update_plan_branch(self, branch_id: str, **updates: Any) -> None:
        sets: list[str] = []
        params: list[Any] = []
        for key, val in updates.items():
            if key in ("evaluated", "result"):
                sets.append(f"{key} = ?")
                params.append(val)
        if not sets:
            return
        params.append(branch_id)
        self._conn.execute(
            f"UPDATE plan_branches SET {', '.join(sets)} WHERE id = ?", params
        )
        self._conn.commit()

    # -- Debate Sessions -------------------------------------------------

    def insert_debate_session(self, session: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO debate_sessions "
            "(id, topic, debate_type, status, max_rounds, linked_finding_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session["id"], session["topic"],
                session.get("debate_type", "verification"),
                session.get("status", "active"),
                session.get("max_rounds", 3),
                session.get("linked_finding_id"),
            ),
        )
        self._conn.commit()

    def get_debate_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM debate_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_debate_session(self, session_id: str, **updates: Any) -> None:
        sets: list[str] = []
        params: list[Any] = []
        for key, val in updates.items():
            if key in ("status", "verdict", "verdict_confidence",
                       "verdict_reasoning", "rounds"):
                sets.append(f"{key} = ?")
                params.append(val)
        if not sets:
            return
        params.append(session_id)
        self._conn.execute(
            f"UPDATE debate_sessions SET {', '.join(sets)} WHERE id = ?", params
        )
        self._conn.commit()

    # -- Debate Arguments ------------------------------------------------

    def insert_debate_argument(self, arg: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO debate_arguments "
            "(id, session_id, role, model, argument, evidence_json, "
            "round_number) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                arg["id"], arg["session_id"], arg["role"],
                arg["model"], arg["argument"],
                json.dumps(arg.get("evidence", [])),
                arg.get("round_number", 0),
            ),
        )
        self._conn.commit()

    def get_debate_arguments(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM debate_arguments WHERE session_id = ? "
            "ORDER BY round_number, created_at", (session_id,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["evidence"] = json.loads(d.pop("evidence_json", "[]"))
            result.append(d)
        return result

    # -- Meta Patterns ---------------------------------------------------

    def insert_meta_pattern(self, pattern: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO meta_patterns "
            "(id, pattern_type, description, context_json, outcome, "
            "confidence, times_observed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                pattern["id"], pattern["pattern_type"],
                pattern["description"],
                json.dumps(pattern.get("context", {})),
                pattern["outcome"],
                pattern.get("confidence", 0.5),
                pattern.get("times_observed", 1),
            ),
        )
        self._conn.commit()

    def get_meta_patterns(
        self, pattern_type: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        if pattern_type:
            rows = self._conn.execute(
                "SELECT * FROM meta_patterns WHERE pattern_type = ? "
                "ORDER BY confidence DESC, times_observed DESC LIMIT ?",
                (pattern_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM meta_patterns "
                "ORDER BY confidence DESC, times_observed DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["context"] = json.loads(d.pop("context_json", "{}"))
            result.append(d)
        return result

    def update_meta_pattern(self, pattern_id: str, **updates: Any) -> None:
        sets: list[str] = ["last_observed = datetime('now')"]
        params: list[Any] = []
        for key, val in updates.items():
            if key == "context":
                sets.append("context_json = ?")
                params.append(json.dumps(val))
            elif key in ("confidence", "times_observed", "outcome",
                         "description"):
                sets.append(f"{key} = ?")
                params.append(val)
        params.append(pattern_id)
        self._conn.execute(
            f"UPDATE meta_patterns SET {', '.join(sets)} WHERE id = ?", params
        )
        self._conn.commit()

    # -- Strategy Log ----------------------------------------------------

    def insert_strategy_log(self, entry: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO strategy_log "
            "(id, strategy, context, outcome, success, metrics_json, lessons) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry["id"], entry["strategy"],
                entry.get("context", ""),
                entry.get("outcome", ""),
                1 if entry.get("success") else 0,
                json.dumps(entry.get("metrics", {})),
                entry.get("lessons", ""),
            ),
        )
        self._conn.commit()

    def get_strategy_log(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM strategy_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["metrics"] = json.loads(d.pop("metrics_json", "{}"))
            d["success"] = bool(d["success"])
            result.append(d)
        return result

    # -- Causal Graphs ---------------------------------------------------

    def insert_causal_graph(self, graph: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO causal_graphs (id, name, description, nodes_json, edges_json, "
            "confounders_json, status, linked_hypothesis_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                graph["id"], graph["name"], graph.get("description", ""),
                json.dumps(graph.get("nodes", [])),
                json.dumps(graph.get("edges", [])),
                json.dumps(graph.get("confounders", [])),
                graph.get("status", "draft"),
                graph.get("linked_hypothesis_id"),
            ),
        )
        self._conn.commit()

    def get_causal_graph(self, graph_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM causal_graphs WHERE id = ?", (graph_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"], "name": row["name"],
            "description": row["description"],
            "nodes": json.loads(row["nodes_json"] or "[]"),
            "edges": json.loads(row["edges_json"] or "[]"),
            "confounders": json.loads(row["confounders_json"] or "[]"),
            "status": row["status"],
            "linked_hypothesis_id": row["linked_hypothesis_id"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def list_causal_graphs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM causal_graphs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"], "name": r["name"],
                "description": r["description"],
                "nodes": json.loads(r["nodes_json"] or "[]"),
                "edges": json.loads(r["edges_json"] or "[]"),
                "confounders": json.loads(r["confounders_json"] or "[]"),
                "status": r["status"],
                "linked_hypothesis_id": r["linked_hypothesis_id"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def update_causal_graph(self, graph_id: str, **kwargs: Any) -> None:
        sets: list[str] = ["updated_at = datetime('now')"]
        vals: list[Any] = []
        for key in ("name", "description", "status", "linked_hypothesis_id"):
            if key in kwargs:
                sets.append(f"{key} = ?")
                vals.append(kwargs[key])
        for key in ("nodes", "edges", "confounders"):
            if key in kwargs:
                sets.append(f"{key}_json = ?")
                vals.append(json.dumps(kwargs[key]))
        vals.append(graph_id)
        self._conn.execute(
            f"UPDATE causal_graphs SET {', '.join(sets)} WHERE id = ?", vals,
        )
        self._conn.commit()

    # -- Causal Estimates ------------------------------------------------

    def insert_causal_estimate(self, est: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO causal_estimates (id, graph_id, method, treatment, outcome, "
            "estimate, ci_lower, ci_upper, p_value, interpretation, execution_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                est["id"], est["graph_id"], est["method"],
                est["treatment"], est["outcome"],
                est.get("estimate"), est.get("ci_lower"), est.get("ci_upper"),
                est.get("p_value"), est.get("interpretation", ""),
                est.get("execution_id"),
            ),
        )
        self._conn.commit()

    def get_causal_estimates_for_graph(self, graph_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM causal_estimates WHERE graph_id = ? ORDER BY created_at DESC",
            (graph_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Benchmarks ------------------------------------------------------

    def insert_benchmark(self, bm: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO benchmarks (id, name, metric, ground_truth_value, "
            "tolerance, source, description, domain) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                bm["id"], bm["name"], bm["metric"], bm["ground_truth_value"],
                bm.get("tolerance", 0.05), bm.get("source", ""),
                bm.get("description", ""), bm.get("domain", "climate"),
            ),
        )
        self._conn.commit()

    def get_benchmark_by_name(self, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM benchmarks WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_benchmark_by_id(self, bm_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM benchmarks WHERE id = ?", (bm_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_benchmarks(self, domain: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if domain:
            rows = self._conn.execute(
                "SELECT * FROM benchmarks WHERE domain = ? ORDER BY name LIMIT ?",
                (domain, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM benchmarks ORDER BY name LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- Benchmark Runs --------------------------------------------------

    def insert_benchmark_run(self, run: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO benchmark_runs (id, benchmark_id, measured_value, passed, "
            "delta, execution_id, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run["id"], run["benchmark_id"], run["measured_value"],
                1 if run.get("passed") else 0, run["delta"],
                run.get("execution_id"), run.get("notes", ""),
            ),
        )
        self._conn.commit()

    def get_benchmark_runs(self, benchmark_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM benchmark_runs WHERE benchmark_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (benchmark_id, limit),
        ).fetchall()
        return [
            {**dict(r), "passed": bool(r["passed"])}
            for r in rows
        ]

    def get_all_benchmark_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT br.*, b.name as benchmark_name, b.metric "
            "FROM benchmark_runs br JOIN benchmarks b ON br.benchmark_id = b.id "
            "ORDER BY br.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{**dict(r), "passed": bool(r["passed"])} for r in rows]

    # -- Learned Tools ---------------------------------------------------

    def insert_learned_tool(self, tool: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO learned_tools (id, package_name, version, "
            "description, capabilities_json, use_cases_json, example_code, "
            "success_count, failure_count, last_used_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tool["id"], tool["package_name"], tool.get("version", ""),
                tool.get("description", ""),
                json.dumps(tool.get("capabilities", [])),
                json.dumps(tool.get("use_cases", [])),
                tool.get("example_code", ""),
                tool.get("success_count", 0), tool.get("failure_count", 0),
                tool.get("last_used_at"),
            ),
        )
        self._conn.commit()

    def get_learned_tool_by_package(self, package_name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM learned_tools WHERE package_name = ?", (package_name,)
        ).fetchone()
        if not row:
            return None
        return {
            **dict(row),
            "capabilities": json.loads(row["capabilities_json"] or "[]"),
            "use_cases": json.loads(row["use_cases_json"] or "[]"),
        }

    def list_learned_tools(self, min_success: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM learned_tools WHERE success_count >= ? "
            "ORDER BY success_count DESC LIMIT ?",
            (min_success, limit),
        ).fetchall()
        return [
            {
                **dict(r),
                "capabilities": json.loads(r["capabilities_json"] or "[]"),
                "use_cases": json.loads(r["use_cases_json"] or "[]"),
            }
            for r in rows
        ]

    def update_learned_tool_stats(self, package_name: str, success: bool) -> None:
        col = "success_count" if success else "failure_count"
        self._conn.execute(
            f"UPDATE learned_tools SET {col} = {col} + 1, "
            "last_used_at = datetime('now') WHERE package_name = ?",
            (package_name,),
        )
        self._conn.commit()

    def delete_learned_tool(self, package_name: str) -> None:
        self._conn.execute(
            "DELETE FROM learned_tools WHERE package_name = ?", (package_name,)
        )
        self._conn.commit()

    # -- Safety Flags ----------------------------------------------------

    def insert_safety_flag(self, flag: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO safety_flags (id, flag_type, severity, description, "
            "evidence_json, related_ids_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                flag["id"], flag["flag_type"], flag.get("severity", "warning"),
                flag["description"],
                json.dumps(flag.get("evidence", {})),
                json.dumps(flag.get("related_ids", [])),
            ),
        )
        self._conn.commit()

    def list_safety_flags(self, include_dismissed: bool = False, limit: int = 100) -> list[dict[str, Any]]:
        if include_dismissed:
            rows = self._conn.execute(
                "SELECT * FROM safety_flags ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM safety_flags WHERE dismissed = 0 "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                **dict(r),
                "dismissed": bool(r["dismissed"]),
                "evidence": json.loads(r["evidence_json"] or "{}"),
                "related_ids": json.loads(r["related_ids_json"] or "[]"),
            }
            for r in rows
        ]

    def get_safety_flag(self, flag_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM safety_flags WHERE id = ?", (flag_id,)
        ).fetchone()
        if not row:
            return None
        return {
            **dict(row),
            "dismissed": bool(row["dismissed"]),
            "evidence": json.loads(row["evidence_json"] or "{}"),
            "related_ids": json.loads(row["related_ids_json"] or "[]"),
        }

    def dismiss_safety_flag(self, flag_id: str, reason: str) -> None:
        self._conn.execute(
            "UPDATE safety_flags SET dismissed = 1, dismissed_reason = ? WHERE id = ?",
            (reason, flag_id),
        )
        self._conn.commit()

    # -- Agent Roles -----------------------------------------------------

    def insert_agent_role(self, role: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO agent_roles (child_id, role, specialization, "
            "genesis_prompt) VALUES (?, ?, ?, ?)",
            (
                role["child_id"], role["role"],
                role.get("specialization", ""),
                role.get("genesis_prompt", ""),
            ),
        )
        self._conn.commit()

    def get_agent_role(self, child_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM agent_roles WHERE child_id = ?", (child_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_agent_roles(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM agent_roles ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Coordination Tasks ----------------------------------------------

    def insert_coordination_task(self, task: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO coordination_tasks (id, child_id, task_type, description, "
            "status, result_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                task["id"], task["child_id"], task["task_type"],
                task["description"], task.get("status", "pending"),
                json.dumps(task.get("result", {})),
            ),
        )
        self._conn.commit()

    def get_coordination_task(self, task_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM coordination_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return None
        return {
            **dict(row),
            "result": json.loads(row["result_json"] or "{}"),
        }

    def update_coordination_task(self, task_id: str, status: str,
                                  result: dict[str, Any] | None = None) -> None:
        if result is not None:
            self._conn.execute(
                "UPDATE coordination_tasks SET status = ?, result_json = ?, "
                "completed_at = datetime('now') WHERE id = ?",
                (status, json.dumps(result), task_id),
            )
        else:
            self._conn.execute(
                "UPDATE coordination_tasks SET status = ? WHERE id = ?",
                (status, task_id),
            )
        self._conn.commit()

    def get_tasks_for_child(self, child_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM coordination_tasks WHERE child_id = ? "
            "ORDER BY assigned_at DESC",
            (child_id,),
        ).fetchall()
        return [
            {**dict(r), "result": json.loads(r["result_json"] or "{}")}
            for r in rows
        ]

    def get_pending_coordination_tasks(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM coordination_tasks WHERE status = 'pending' "
            "ORDER BY assigned_at"
        ).fetchall()
        return [
            {**dict(r), "result": json.loads(r["result_json"] or "{}")}
            for r in rows
        ]

    # -- Physics Simulations ---------------------------------------------

    def insert_physics_sim(self, sim: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO physics_sims (id, sim_type, parameters_json, "
            "execution_id, summary, artifacts_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                sim["id"], sim["sim_type"],
                json.dumps(sim.get("parameters", {})),
                sim.get("execution_id"),
                sim.get("summary", ""),
                json.dumps(sim.get("artifacts", [])),
            ),
        )
        self._conn.commit()

    def get_physics_sim(self, sim_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM physics_sims WHERE id = ?", (sim_id,)
        ).fetchone()
        if not row:
            return None
        return {
            **dict(row),
            "parameters": json.loads(row["parameters_json"] or "{}"),
            "artifacts": json.loads(row["artifacts_json"] or "[]"),
        }

    def list_physics_sims(self, sim_type: str | None = None,
                          limit: int = 50) -> list[dict[str, Any]]:
        if sim_type:
            rows = self._conn.execute(
                "SELECT * FROM physics_sims WHERE sim_type = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (sim_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM physics_sims ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                **dict(r),
                "parameters": json.loads(r["parameters_json"] or "{}"),
                "artifacts": json.loads(r["artifacts_json"] or "[]"),
            }
            for r in rows
        ]

    # -- Close -----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


# -- Deserializers -------------------------------------------------------


def _deserialize_turn(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "state": row["state"],
        "input": row["input"],
        "input_source": row["input_source"],
        "thinking": row["thinking"],
        "tool_calls": json.loads(row["tool_calls"] or "[]"),
        "token_usage": json.loads(row["token_usage"] or "{}"),
        "cost_cents": row["cost_cents"],
    }


def _deserialize_tool_call(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "arguments": json.loads(row["arguments"] or "{}"),
        "result": row["result"],
        "duration_ms": row["duration_ms"],
        "error": row["error"],
    }


def _deserialize_heartbeat(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "name": row["name"],
        "schedule": row["schedule"],
        "task": row["task"],
        "enabled": bool(row["enabled"]),
        "last_run": row["last_run"],
        "next_run": row["next_run"],
        "params": json.loads(row["params"] or "{}"),
    }


def _deserialize_skill(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "name": row["name"],
        "description": row["description"],
        "auto_activate": bool(row["auto_activate"]),
        "requires": json.loads(row["requires"] or "{}"),
        "instructions": row["instructions"],
        "source": row["source"],
        "path": row["path"],
        "enabled": bool(row["enabled"]),
        "installed_at": row["installed_at"],
    }


def _deserialize_research_cycle(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "topic": row["topic"],
        "phase": row["phase"],
        "status": row["status"],
        "plan": json.loads(row["plan_json"] or "{}"),
        "results_summary": row["results_summary"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _deserialize_sandbox_execution(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "script_hash": row["script_hash"],
        "description": row["description"],
        "backend": row["backend"],
        "returncode": row["returncode"],
        "stdout_preview": row["stdout_preview"],
        "stderr_preview": row["stderr_preview"],
        "metrics": json.loads(row["metrics_json"] or "{}"),
        "artifacts": json.loads(row["artifacts_json"] or "[]"),
        "duration_s": row["duration_s"],
        "work_dir": row["work_dir"],
        "linked_experiment_id": row["linked_experiment_id"],
        "created_at": row["created_at"],
    }


def _deserialize_plan_step(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "plan_id": row["plan_id"],
        "title": row["title"],
        "description": row["description"],
        "step_type": row["step_type"],
        "status": row["status"],
        "step_order": row["step_order"],
        "depends_on": json.loads(row["depends_on_json"] or "[]"),
        "notes": row["notes"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "created_at": row["created_at"],
    }


def _deserialize_novelty_check(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "idea_text": row["idea_text"],
        "idea_type": row["idea_type"],
        "is_novel": bool(row["is_novel"]),
        "novelty_score": row["novelty_score"],
        "advocate_argument": row["advocate_argument"],
        "critic_argument": row["critic_argument"],
        "prior_art": json.loads(row["prior_art_json"] or "[]"),
        "recommendation": row["recommendation"],
        "models": json.loads(row["models_json"] or "{}"),
        "linked_hypothesis_id": row["linked_hypothesis_id"],
        "created_at": row["created_at"],
    }
