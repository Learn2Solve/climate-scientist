"""Meta-learning: track patterns across research activity and suggest strategies.

Analyzes the strategy log to identify what works and what does not,
discovers patterns, and recommends strategies for new research contexts.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.meta_learning")

# Pattern types tracked by the meta-learner
PATTERN_TYPES = {
    "experiment_type_success",
    "feature_importance",
    "methodology_outcome",
    "failure_mode",
    "data_quality",
}


class MetaLearner:
    """Track patterns across research activity and suggest strategies."""

    def __init__(self, db: Any, config: Any) -> None:
        self._db = db
        self._config = config

    def log_strategy_outcome(
        self,
        strategy: str,
        context: str,
        outcome: str,
        success: bool,
        metrics: dict[str, Any] | None = None,
        lessons: str = "",
    ) -> str:
        """Record a strategy attempt and its outcome.

        strategy: free text description of the approach taken
        context: what situation prompted this strategy
        outcome: what happened
        success: whether the strategy achieved its goal
        metrics: optional quantitative metrics
        lessons: optional free-text lessons learned

        Returns the log entry ID.
        """
        log_id = str(ULID())
        self._db.insert_strategy_log({
            "id": log_id,
            "strategy": strategy,
            "context": context,
            "outcome": outcome,
            "success": success,
            "metrics": metrics or {},
            "lessons": lessons,
        })
        log.info(
            "Logged strategy outcome %s: success=%s strategy='%s'",
            log_id, success, strategy[:60],
        )
        return log_id

    def extract_patterns(self) -> list[dict[str, Any]]:
        """Analyze strategy_log to discover patterns.

        Groups strategies by keyword similarity, calculates success rates,
        and stores/updates meta_patterns in the DB.

        Returns list of discovered/updated patterns.
        """
        entries = self._db.get_strategy_log(limit=200)
        if not entries:
            return []

        # Group strategies by keywords
        groups = self._group_by_keywords(entries)

        discovered = []
        for group_key, group_entries in groups.items():
            total = len(group_entries)
            if total < 2:
                continue

            successes = sum(1 for e in group_entries if e["success"])
            success_rate = successes / total

            # Classify the pattern type
            pattern_type = self._classify_pattern_type(group_key, success_rate)

            # Build context from the group
            context = {
                "keywords": group_key,
                "total_attempts": total,
                "success_count": successes,
                "sample_strategies": [e["strategy"][:100] for e in group_entries[:3]],
            }

            # Determine outcome description
            if success_rate >= 0.7:
                outcome = "high_success"
                description = f"Strategy group '{group_key}' has high success rate ({success_rate:.0%})"
            elif success_rate <= 0.3:
                outcome = "low_success"
                description = f"Strategy group '{group_key}' has low success rate ({success_rate:.0%})"
            else:
                outcome = "mixed"
                description = f"Strategy group '{group_key}' has mixed results ({success_rate:.0%})"

            # Check if pattern already exists, update or create
            existing = self._find_existing_pattern(group_key)
            if existing:
                self._db.update_meta_pattern(
                    existing["id"],
                    confidence=success_rate,
                    times_observed=total,
                    outcome=outcome,
                )
                pattern = {
                    "id": existing["id"],
                    "pattern_type": pattern_type,
                    "description": description,
                    "context": context,
                    "outcome": outcome,
                    "confidence": success_rate,
                    "times_observed": total,
                    "updated": True,
                }
            else:
                pattern_id = str(ULID())
                self._db.insert_meta_pattern({
                    "id": pattern_id,
                    "pattern_type": pattern_type,
                    "description": description,
                    "context": context,
                    "outcome": outcome,
                    "confidence": success_rate,
                    "times_observed": total,
                })
                pattern = {
                    "id": pattern_id,
                    "pattern_type": pattern_type,
                    "description": description,
                    "context": context,
                    "outcome": outcome,
                    "confidence": success_rate,
                    "times_observed": total,
                    "updated": False,
                }

            discovered.append(pattern)

        log.info("Extracted %d patterns from %d strategy log entries", len(discovered), len(entries))
        return discovered

    def suggest_strategy(self, context: str) -> dict[str, Any]:
        """Suggest a strategy based on learned patterns.

        Matches context keywords against stored meta_patterns,
        weighted by confidence and times_observed.

        Returns {"suggestion": str, "confidence": float,
                 "based_on": str, "past_success_rate": float}.
        """
        patterns = self._db.get_meta_patterns(limit=50)
        if not patterns:
            return {
                "suggestion": "No patterns available yet. Try different approaches and log outcomes.",
                "confidence": 0.0,
                "based_on": "none",
                "past_success_rate": 0.0,
            }

        # Score each pattern against the query context
        scored: list[tuple[float, dict[str, Any]]] = []
        for p in patterns:
            match_score = self._match_context(context, p.get("context", {}))
            # Weight by confidence and observation count
            weight = match_score * p.get("confidence", 0.5) * min(p.get("times_observed", 1), 10) / 10
            scored.append((weight, p))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored or scored[0][0] == 0.0:
            return {
                "suggestion": "No matching patterns found for this context.",
                "confidence": 0.0,
                "based_on": "none",
                "past_success_rate": 0.0,
            }

        best_score, best_pattern = scored[0]
        description = best_pattern.get("description", "")
        pattern_context = best_pattern.get("context", {})
        sample_strategies = pattern_context.get("sample_strategies", [])

        suggestion = description
        if sample_strategies:
            suggestion += f" Example approach: {sample_strategies[0]}"

        return {
            "suggestion": suggestion,
            "confidence": best_pattern.get("confidence", 0.5),
            "based_on": best_pattern.get("id", "unknown"),
            "past_success_rate": best_pattern.get("confidence", 0.5),
        }

    def get_research_insights(self) -> dict[str, Any]:
        """Aggregate meta-learning insights across all research activity.

        Returns a dict with:
        - total_strategies: count of logged strategies
        - overall_success_rate: fraction of successful strategies
        - top_successful_patterns: top 5 high-confidence patterns
        - top_failure_modes: top 5 failure patterns
        - recommendations: suggested next steps
        """
        entries = self._db.get_strategy_log(limit=500)
        total = len(entries)
        successes = sum(1 for e in entries if e["success"])
        overall_rate = successes / total if total > 0 else 0.0

        # Get patterns
        all_patterns = self._db.get_meta_patterns(limit=100)

        # Top successful patterns (high confidence, outcome != low_success)
        successful = [
            p for p in all_patterns
            if p.get("outcome") == "high_success"
        ]
        successful.sort(key=lambda p: p.get("confidence", 0) * p.get("times_observed", 1), reverse=True)

        # Top failure modes
        failures = [
            p for p in all_patterns
            if p.get("outcome") == "low_success" or p.get("pattern_type") == "failure_mode"
        ]
        failures.sort(key=lambda p: p.get("times_observed", 0), reverse=True)

        # Build recommendations
        recommendations = []
        if successful:
            top = successful[0]
            recommendations.append(
                f"Continue using approaches related to: {top.get('description', '')[:100]}"
            )
        if failures:
            top_fail = failures[0]
            recommendations.append(
                f"Avoid or revise: {top_fail.get('description', '')[:100]}"
            )
        if overall_rate < 0.5 and total >= 5:
            recommendations.append(
                "Overall success rate is below 50%. Consider changing experimental methodology."
            )
        if total < 5:
            recommendations.append(
                "Insufficient data for strong recommendations. Continue logging strategy outcomes."
            )

        return {
            "total_strategies": total,
            "overall_success_rate": round(overall_rate, 3),
            "successful_count": successes,
            "failed_count": total - successes,
            "top_successful_patterns": [
                _pattern_summary(p) for p in successful[:5]
            ],
            "top_failure_modes": [
                _pattern_summary(p) for p in failures[:5]
            ],
            "total_patterns": len(all_patterns),
            "recommendations": recommendations,
        }

    # -- Private helpers --------------------------------------------------

    def _match_context(self, query_context: str, pattern_context: dict[str, Any]) -> float:
        """Simple keyword-based context matching. Returns similarity score 0-1."""
        query_words = set(_tokenize(query_context))
        if not query_words:
            return 0.0

        # Build target word set from pattern context
        target_words: set[str] = set()

        # Extract words from the keywords field
        keywords = pattern_context.get("keywords", "")
        if keywords:
            target_words.update(_tokenize(keywords))

        # Extract words from sample strategies
        for strategy in pattern_context.get("sample_strategies", []):
            target_words.update(_tokenize(strategy))

        if not target_words:
            return 0.0

        # Jaccard-like overlap
        overlap = len(query_words & target_words)
        union = len(query_words | target_words)
        return overlap / union if union > 0 else 0.0

    def _group_by_keywords(self, entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Group strategy log entries by shared keywords.

        Uses a simple approach: extract the most common non-trivial words
        from each strategy, then group entries that share dominant keywords.
        """
        # Count word frequencies across all entries
        word_counts: dict[str, int] = defaultdict(int)
        entry_words: list[tuple[dict[str, Any], set[str]]] = []

        for entry in entries:
            words = _tokenize(entry.get("strategy", "") + " " + entry.get("context", ""))
            entry_words.append((entry, words))
            for w in words:
                word_counts[w] += 1

        # Use words that appear in at least 2 entries as grouping keys
        significant_words = {w for w, c in word_counts.items() if c >= 2}

        # Assign each entry to a group based on its most significant word
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry, words in entry_words:
            sig = words & significant_words
            if sig:
                # Use the most common significant word as the group key
                group_key = max(sig, key=lambda w: word_counts[w])
                groups[group_key].append(entry)
            else:
                groups["_other"].append(entry)

        # Remove the catch-all group if it exists
        groups.pop("_other", None)
        return dict(groups)

    def _classify_pattern_type(self, group_key: str, success_rate: float) -> str:
        """Classify a pattern into one of the known pattern types."""
        key_lower = group_key.lower()

        # Experiment type keywords
        experiment_words = {"experiment", "trial", "test", "run", "baseline", "model", "training"}
        if key_lower in experiment_words:
            return "experiment_type_success"

        # Feature keywords
        feature_words = {"feature", "variable", "predictor", "input", "column", "sst", "shear", "humidity"}
        if key_lower in feature_words:
            return "feature_importance"

        # Methodology keywords
        method_words = {"method", "approach", "technique", "algorithm", "regression", "xgboost", "neural"}
        if key_lower in method_words:
            return "methodology_outcome"

        # Data keywords
        data_words = {"data", "dataset", "source", "era5", "hurdat", "ibtracs", "quality"}
        if key_lower in data_words:
            return "data_quality"

        # Low success rate -> failure mode
        if success_rate <= 0.3:
            return "failure_mode"

        return "methodology_outcome"

    def _find_existing_pattern(self, group_key: str) -> dict[str, Any] | None:
        """Check if a pattern with matching keywords already exists."""
        patterns = self._db.get_meta_patterns(limit=100)
        for p in patterns:
            ctx = p.get("context", {})
            if ctx.get("keywords") == group_key:
                return p
        return None


def _tokenize(text: str) -> set[str]:
    """Extract lowercase tokens, filtering out short and stop words."""
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "was", "were",
        "are", "has", "had", "have", "been", "not", "but", "its", "also",
        "can", "will", "may", "more", "than", "each", "all", "into",
        "using", "used", "use", "based", "when", "which", "their", "does",
    }
    words = set()
    for word in text.lower().split():
        # Strip non-alphanumeric characters from edges
        cleaned = word.strip(".,;:!?()[]{}\"'")
        if len(cleaned) >= 3 and cleaned not in stop_words:
            words.add(cleaned)
    return words


def _pattern_summary(pattern: dict[str, Any]) -> dict[str, Any]:
    """Create a concise summary of a pattern for reporting."""
    return {
        "id": pattern.get("id", ""),
        "type": pattern.get("pattern_type", ""),
        "description": pattern.get("description", "")[:200],
        "confidence": pattern.get("confidence", 0.0),
        "times_observed": pattern.get("times_observed", 0),
        "outcome": pattern.get("outcome", ""),
    }
