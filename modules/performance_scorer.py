"""
Performance Scorer — Multi-Dimensional Task Reinforcement Engine.

v2.0: Consciousness-inspired scoring system.

Scores each completed task on a multi-dimensional rubric:
  - Token cost (how much compute was spent)
  - Quality (test pass rate, DoD pass rate, first-attempt success)
  - Complexity (estimated difficulty of the task)
  - Effort-to-outcome ratio (the key metric: was the investment justified?)

The scoring produces a reinforcement signal that accumulates over time
in the performance journal, biasing future strategy selection.

This is NOT a single-metric optimizer. A task that took a million tokens
but produced exceptional quality on a hard problem scores POSITIVELY.
A task that took a million tokens on a simple problem and produced
garbage scores NEGATIVELY. Same cost, opposite signal — because the
ratio between effort and justified complexity is what matters.

Philosophy: This is artificial emotion implemented as attention modulation.
The system develops preferences for approaches that historically produce
good effort-to-outcome ratios, and aversion to approaches that waste
resources or underinvest on complex tasks.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ── Scoring Constants ────────────────────────────────────────────

# Complexity estimation weights
COMPLEXITY_SIGNALS = {
    "source_files": 1.5,       # More files = more complexity
    "test_files": 0.5,         # Tests add moderate complexity
    "has_database": 2.0,       # DB integration is hard
    "has_api": 2.0,            # REST API is hard
    "has_auth": 1.5,           # Auth logic is tricky
    "has_file_io": 1.0,        # File I/O adds complexity
    "inter_module_deps": 1.5,  # Cross-file dependencies
}

# Reinforcement signal thresholds
EXCELLENT_RATIO = 0.8    # Top 20% — strong positive reinforcement
GOOD_RATIO = 0.5         # Above median — mild positive
NEUTRAL_RATIO = 0.3      # Median — no reinforcement
POOR_RATIO = 0.15        # Below median — mild negative
TERRIBLE_RATIO = 0.0     # Bottom — strong negative


@dataclass
class TaskScore:
    """Complete scoring profile for a single task execution."""

    # Identity
    task_id: str
    goal: str
    timestamp: str = ""

    # Raw metrics
    total_tokens: int = 0          # Total eval_count across all agent calls
    duration_seconds: float = 0.0  # Wall clock time
    iterations_used: int = 1       # How many EXPLORE→BUILD→TEST loops
    max_iterations: int = 3        # Budget allocated

    # Quality metrics
    tests_passed: int = 0
    tests_total: int = 0
    dod_passed: int = 0
    dod_total: int = 0
    first_iteration_success: bool = False  # Did it pass on iteration 1?

    # Complexity estimate
    complexity_score: float = 1.0   # 0-10 scale
    source_file_count: int = 0
    task_level: int = 0             # From benchmark (0 if unknown)

    # Computed scores (filled by score() method)
    quality_score: float = 0.0      # 0.0 - 1.0
    efficiency_score: float = 0.0   # 0.0 - 1.0
    effort_ratio: float = 0.0       # quality / normalized_effort
    reinforcement_signal: float = 0.0  # -1.0 to 1.0

    # Strategy metadata
    approach_used: str = ""         # "micro-build", "monolithic", etc.
    model_used: str = ""
    playbook_bullets_active: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskScore":
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class PerformanceScorer:
    """
    Scores task executions and maintains the performance journal.

    The journal accumulates over time, giving the system a history
    of what works, what doesn't, and how to calibrate effort to
    complexity. This is the persistence layer for system identity.
    """

    def __init__(self, db_path: str = "performance_journal.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the performance journal database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                goal TEXT,
                timestamp TEXT,
                total_tokens INTEGER DEFAULT 0,
                duration_seconds REAL DEFAULT 0,
                iterations_used INTEGER DEFAULT 1,
                max_iterations INTEGER DEFAULT 3,
                tests_passed INTEGER DEFAULT 0,
                tests_total INTEGER DEFAULT 0,
                dod_passed INTEGER DEFAULT 0,
                dod_total INTEGER DEFAULT 0,
                first_iteration_success INTEGER DEFAULT 0,
                complexity_score REAL DEFAULT 1.0,
                source_file_count INTEGER DEFAULT 0,
                task_level INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 0,
                efficiency_score REAL DEFAULT 0,
                effort_ratio REAL DEFAULT 0,
                reinforcement_signal REAL DEFAULT 0,
                approach_used TEXT DEFAULT '',
                model_used TEXT DEFAULT '',
                playbook_bullets_active INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS approach_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complexity_bucket TEXT,
                approach TEXT,
                avg_reinforcement REAL,
                sample_count INTEGER,
                last_updated TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS effort_curves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complexity_bucket TEXT,
                avg_tokens INTEGER,
                avg_quality REAL,
                avg_ratio REAL,
                sample_count INTEGER,
                last_updated TEXT
            )
        """)
        conn.commit()
        conn.close()

    def score_task(self, score: TaskScore) -> TaskScore:
        """
        Compute the full scoring profile for a completed task.

        This is where the multi-dimensional rubric produces
        the reinforcement signal.
        """
        score.timestamp = score.timestamp or datetime.now().isoformat()

        # 1. Quality score (0.0 - 1.0)
        score.quality_score = self._compute_quality(score)

        # 2. Efficiency score (0.0 - 1.0)
        score.efficiency_score = self._compute_efficiency(score)

        # 3. Effort-to-outcome ratio
        score.effort_ratio = self._compute_effort_ratio(score)

        # 4. Reinforcement signal (-1.0 to 1.0)
        score.reinforcement_signal = self._compute_reinforcement(score)

        # Store in journal
        self._store_score(score)

        # Update aggregate tables
        self._update_preferences(score)
        self._update_effort_curves(score)

        logger.info(
            f"📊 Task scored: quality={score.quality_score:.2f} "
            f"efficiency={score.efficiency_score:.2f} "
            f"ratio={score.effort_ratio:.2f} "
            f"reinforcement={score.reinforcement_signal:+.2f}"
        )

        return score

    def _compute_quality(self, score: TaskScore) -> float:
        """
        Quality is a weighted combination of test pass rate,
        DoD pass rate, and iteration efficiency.
        """
        # Test pass rate (0-1)
        test_rate = score.tests_passed / max(score.tests_total, 1)

        # DoD pass rate (0-1)
        dod_rate = score.dod_passed / max(score.dod_total, 1)

        # Iteration efficiency: did it finish early?
        # 1.0 if first iteration, decays as iterations increase
        iter_efficiency = 1.0 / score.iterations_used

        # First-attempt bonus
        first_bonus = 0.1 if score.first_iteration_success else 0.0

        # Weighted combination
        quality = (
            test_rate * 0.40 +
            dod_rate * 0.35 +
            iter_efficiency * 0.15 +
            first_bonus * 0.10
        )

        return min(1.0, max(0.0, quality))

    def _compute_efficiency(self, score: TaskScore) -> float:
        """
        Efficiency measures how well resources were used.
        Normalized against complexity — spending tokens on hard
        tasks is expected, spending them on easy tasks is waste.
        """
        if score.total_tokens == 0:
            return 0.5  # No data, assume neutral

        # Token cost per unit of complexity
        # Higher complexity = more tokens expected = less penalty
        complexity_factor = max(score.complexity_score, 0.5)
        normalized_tokens = score.total_tokens / (complexity_factor * 10000)

        # Efficiency decays as normalized tokens increase
        # exp(-x) curve: 1.0 at 0 tokens, approaches 0 at infinity
        import math
        efficiency = math.exp(-normalized_tokens * 0.3)

        return min(1.0, max(0.0, efficiency))

    def _compute_effort_ratio(self, score: TaskScore) -> float:
        """
        The key metric: quality / normalized_effort.

        This is what determines whether the investment was justified.
        High quality + high effort on hard task = good ratio.
        Low quality + high effort on any task = bad ratio.
        High quality + low effort on easy task = great ratio.
        """
        if score.quality_score == 0:
            return 0.0

        # Normalize effort by complexity
        complexity_factor = max(score.complexity_score, 0.5)

        # Effort combines tokens and iterations
        token_effort = score.total_tokens / max(complexity_factor * 5000, 1)
        iter_effort = score.iterations_used / max(score.max_iterations, 1)
        combined_effort = (token_effort * 0.7 + iter_effort * 0.3)

        # Ratio: quality relative to effort
        if combined_effort == 0:
            return score.quality_score  # No effort recorded, quality is the signal

        ratio = score.quality_score / max(combined_effort, 0.01)

        # Clamp to 0-1 range
        return min(1.0, max(0.0, ratio))

    def _compute_reinforcement(self, score: TaskScore) -> float:
        """
        Convert the effort ratio into a reinforcement signal.

        Positive signal: approach worked well for this complexity level.
        Negative signal: approach was wasteful or underperformed.
        Neutral: about what we'd expect.

        Uses historical data to set expectations — the signal is
        relative to past performance, not absolute.
        """
        # Get historical baseline for this complexity level
        baseline = self._get_historical_baseline(score.complexity_score)

        if baseline is None:
            # No history — use absolute thresholds
            ratio = score.effort_ratio
            if ratio >= EXCELLENT_RATIO:
                return 0.8
            elif ratio >= GOOD_RATIO:
                return 0.4
            elif ratio >= NEUTRAL_RATIO:
                return 0.0
            elif ratio >= POOR_RATIO:
                return -0.4
            else:
                return -0.8

        # Relative to historical performance
        deviation = score.effort_ratio - baseline
        # Scale deviation to -1..1 range
        signal = max(-1.0, min(1.0, deviation * 3.0))

        return signal

    def _get_historical_baseline(self, complexity: float) -> Optional[float]:
        """Get the average effort ratio for similar complexity tasks."""
        bucket = self._complexity_bucket(complexity)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT avg_ratio, sample_count FROM effort_curves WHERE complexity_bucket = ?",
            (bucket,)
        ).fetchone()
        conn.close()

        if row and row[1] >= 3:  # Need at least 3 samples
            return row[0]
        return None

    def _store_score(self, score: TaskScore):
        """Persist a task score to the journal."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO task_scores (
                task_id, goal, timestamp,
                total_tokens, duration_seconds, iterations_used, max_iterations,
                tests_passed, tests_total, dod_passed, dod_total,
                first_iteration_success,
                complexity_score, source_file_count, task_level,
                quality_score, efficiency_score, effort_ratio, reinforcement_signal,
                approach_used, model_used, playbook_bullets_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            score.task_id, score.goal, score.timestamp,
            score.total_tokens, score.duration_seconds,
            score.iterations_used, score.max_iterations,
            score.tests_passed, score.tests_total,
            score.dod_passed, score.dod_total,
            1 if score.first_iteration_success else 0,
            score.complexity_score, score.source_file_count, score.task_level,
            score.quality_score, score.efficiency_score,
            score.effort_ratio, score.reinforcement_signal,
            score.approach_used, score.model_used,
            score.playbook_bullets_active,
        ))
        conn.commit()
        conn.close()

    def _update_preferences(self, score: TaskScore):
        """Update approach preference weights based on reinforcement signal."""
        if not score.approach_used:
            return

        bucket = self._complexity_bucket(score.complexity_score)
        conn = sqlite3.connect(self.db_path)

        existing = conn.execute(
            "SELECT avg_reinforcement, sample_count FROM approach_preferences "
            "WHERE complexity_bucket = ? AND approach = ?",
            (bucket, score.approach_used)
        ).fetchone()

        if existing:
            # Running average
            old_avg, count = existing
            new_count = count + 1
            new_avg = (old_avg * count + score.reinforcement_signal) / new_count
            conn.execute(
                "UPDATE approach_preferences SET avg_reinforcement = ?, "
                "sample_count = ?, last_updated = ? "
                "WHERE complexity_bucket = ? AND approach = ?",
                (new_avg, new_count, datetime.now().isoformat(),
                 bucket, score.approach_used)
            )
        else:
            conn.execute(
                "INSERT INTO approach_preferences "
                "(complexity_bucket, approach, avg_reinforcement, sample_count, last_updated) "
                "VALUES (?, ?, ?, 1, ?)",
                (bucket, score.approach_used, score.reinforcement_signal,
                 datetime.now().isoformat())
            )

        conn.commit()
        conn.close()

    def _update_effort_curves(self, score: TaskScore):
        """Update the effort-to-quality curves for complexity estimation."""
        bucket = self._complexity_bucket(score.complexity_score)
        conn = sqlite3.connect(self.db_path)

        existing = conn.execute(
            "SELECT avg_tokens, avg_quality, avg_ratio, sample_count "
            "FROM effort_curves WHERE complexity_bucket = ?",
            (bucket,)
        ).fetchone()

        if existing:
            old_tokens, old_quality, old_ratio, count = existing
            new_count = count + 1
            new_tokens = int((old_tokens * count + score.total_tokens) / new_count)
            new_quality = (old_quality * count + score.quality_score) / new_count
            new_ratio = (old_ratio * count + score.effort_ratio) / new_count
            conn.execute(
                "UPDATE effort_curves SET avg_tokens = ?, avg_quality = ?, "
                "avg_ratio = ?, sample_count = ?, last_updated = ? "
                "WHERE complexity_bucket = ?",
                (new_tokens, new_quality, new_ratio, new_count,
                 datetime.now().isoformat(), bucket)
            )
        else:
            conn.execute(
                "INSERT INTO effort_curves "
                "(complexity_bucket, avg_tokens, avg_quality, avg_ratio, "
                "sample_count, last_updated) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (bucket, score.total_tokens, score.quality_score,
                 score.effort_ratio, datetime.now().isoformat())
            )

        conn.commit()
        conn.close()

    @staticmethod
    def _complexity_bucket(complexity: float) -> str:
        """Map complexity score to a bucket for aggregation."""
        if complexity <= 2.0:
            return "simple"
        elif complexity <= 4.0:
            return "moderate"
        elif complexity <= 6.0:
            return "complex"
        elif complexity <= 8.0:
            return "hard"
        else:
            return "extreme"

    # ── Query Methods ────────────────────────────────────────────

    def get_preferred_approach(self, complexity: float) -> Optional[Dict[str, Any]]:
        """
        Get the historically best-performing approach for a given
        complexity level. Returns approach name and confidence.
        """
        bucket = self._complexity_bucket(complexity)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT approach, avg_reinforcement, sample_count "
            "FROM approach_preferences "
            "WHERE complexity_bucket = ? AND sample_count >= 2 "
            "ORDER BY avg_reinforcement DESC LIMIT 3",
            (bucket,)
        ).fetchall()
        conn.close()

        if not rows:
            return None

        return {
            "recommended": rows[0][0],
            "reinforcement": rows[0][1],
            "confidence": min(1.0, rows[0][2] / 10.0),  # 10+ samples = full confidence
            "alternatives": [
                {"approach": r[0], "reinforcement": r[1], "samples": r[2]}
                for r in rows[1:]
            ]
        }

    def get_effort_estimate(self, complexity: float) -> Optional[Dict[str, Any]]:
        """
        Get expected effort metrics for a given complexity level
        based on historical performance.
        """
        bucket = self._complexity_bucket(complexity)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT avg_tokens, avg_quality, avg_ratio, sample_count "
            "FROM effort_curves WHERE complexity_bucket = ?",
            (bucket,)
        ).fetchone()
        conn.close()

        if not row or row[3] < 2:
            return None

        return {
            "expected_tokens": row[0],
            "expected_quality": row[1],
            "expected_ratio": row[2],
            "confidence": min(1.0, row[3] / 10.0),
            "bucket": bucket,
        }

    def get_performance_trend(self, last_n: int = 20) -> Dict[str, Any]:
        """
        Get the system's performance trend over the last N tasks.
        Is the system improving over time?
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT quality_score, efficiency_score, effort_ratio, "
            "reinforcement_signal, timestamp "
            "FROM task_scores ORDER BY id DESC LIMIT ?",
            (last_n,)
        ).fetchall()
        conn.close()

        if len(rows) < 4:
            return {"trend": "insufficient_data", "sample_count": len(rows)}

        # Split into first half and second half
        mid = len(rows) // 2
        recent = rows[:mid]
        older = rows[mid:]

        recent_avg = sum(r[2] for r in recent) / len(recent)
        older_avg = sum(r[2] for r in older) / len(older)

        if recent_avg > older_avg * 1.1:
            trend = "improving"
        elif recent_avg < older_avg * 0.9:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "recent_avg_ratio": recent_avg,
            "older_avg_ratio": older_avg,
            "improvement_pct": ((recent_avg - older_avg) / max(older_avg, 0.01)) * 100,
            "sample_count": len(rows),
        }

    def get_journal_summary(self) -> Dict[str, Any]:
        """Get overall journal statistics."""
        conn = sqlite3.connect(self.db_path)

        total = conn.execute("SELECT COUNT(*) FROM task_scores").fetchone()[0]
        if total == 0:
            conn.close()
            return {"total_tasks": 0}

        stats = conn.execute("""
            SELECT
                COUNT(*),
                AVG(quality_score),
                AVG(efficiency_score),
                AVG(effort_ratio),
                AVG(reinforcement_signal),
                SUM(total_tokens),
                AVG(iterations_used)
            FROM task_scores
        """).fetchone()

        conn.close()

        return {
            "total_tasks": stats[0],
            "avg_quality": round(stats[1], 3),
            "avg_efficiency": round(stats[2], 3),
            "avg_effort_ratio": round(stats[3], 3),
            "avg_reinforcement": round(stats[4], 3),
            "total_tokens_lifetime": stats[5],
            "avg_iterations": round(stats[6], 2),
        }


def estimate_complexity(
    goal: str,
    source_file_count: int = 0,
    task_level: int = 0,
) -> float:
    """
    Estimate task complexity from available signals.

    Returns a 0-10 score. Used when we don't have a benchmark
    level to go by.
    """
    score = 1.0  # Base complexity

    # File count signal
    score += min(source_file_count * 0.8, 4.0)

    # Keyword signals from the goal
    goal_lower = goal.lower()
    if any(w in goal_lower for w in ["database", "sqlite", "postgres", "sql"]):
        score += COMPLEXITY_SIGNALS["has_database"]
    if any(w in goal_lower for w in ["api", "rest", "flask", "endpoint", "route"]):
        score += COMPLEXITY_SIGNALS["has_api"]
    if any(w in goal_lower for w in ["auth", "login", "jwt", "token", "session"]):
        score += COMPLEXITY_SIGNALS["has_auth"]
    if any(w in goal_lower for w in ["file", "read", "write", "json", "csv", "persist"]):
        score += COMPLEXITY_SIGNALS["has_file_io"]

    # Use benchmark level if available (overrides estimation)
    if task_level > 0:
        # Map levels 1-10 to complexity 1-10
        score = max(score, task_level * 1.2)

    return min(10.0, score)
