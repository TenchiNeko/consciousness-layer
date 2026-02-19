"""
Aspiration Engine — Intrinsic Motivation & Drive System.

v2.0: Consciousness-inspired self-improvement drive.

The scorer records what happened. The advisor recommends what to do.
The aspiration engine provides WHY — the internal pressure to improve.

This module implements intrinsic motivation: the system doesn't just
passively execute tasks, it actively drives toward:
  - Higher test pass rates
  - Greater efficiency (fewer tokens, fewer iterations)
  - Harder challenges (complexity progression)
  - Beating its own historical performance

The engine maintains an "aspiration profile" that evolves over time:
  - Current performance baseline (what am I achieving now?)
  - Aspiration targets (what should I be achieving?)
  - Growth trajectory (am I getting better or stagnating?)
  - Challenge appetite (am I ready for harder tasks?)

The aspiration context gets injected into agent prompts alongside
the strategy advisor's context, creating internal pressure that
shapes how the models approach every task.

Philosophy: A system that just records and advises is a tool.
A system that WANTS to be better is an agent. This module is the
difference between a thermometer (measuring temperature) and a
thermostat (driving toward a target). The aspiration engine makes
the orchestrator a thermostat — it has a desired state and it
actively works to reach it.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

from performance_scorer import PerformanceScorer

logger = logging.getLogger(__name__)


# ── Aspiration Constants ─────────────────────────────────────────

# Target thresholds — what the system aspires to
ASPIRATION_TARGETS = {
    "test_pass_rate": 0.95,       # Want 95%+ tests passing
    "dod_pass_rate": 1.0,         # Want 100% DoD criteria met
    "first_attempt_rate": 0.70,   # Want 70%+ tasks passing first try
    "quality_floor": 0.80,        # Never accept quality below 0.80
    "efficiency_target": 0.70,    # Want efficiency score above 0.70
}

# Growth pressure — how aggressively the system pushes itself
GROWTH_PRESSURE = {
    "baseline_update_rate": 0.3,   # How fast baseline adapts (0=never, 1=instant)
    "aspiration_stretch": 0.10,    # Aspire to 10% above current baseline
    "stagnation_threshold": 10,    # Tasks without improvement triggers restlessness
    "complexity_unlock_quality": 0.85,  # Quality needed to unlock next difficulty
    "regression_sensitivity": 0.15,     # How much quality drop triggers concern
}

# Drive states — the system's current motivational state
class DriveState:
    HUNGRY = "hungry"           # Below targets, strong drive to improve
    PUSHING = "pushing"         # Near targets, actively working to break through
    FLOWING = "flowing"         # Meeting targets, performing well
    RESTLESS = "restless"       # Stagnating, needs new challenges
    RECOVERING = "recovering"   # After a regression, rebuilding confidence


@dataclass
class AspirationProfile:
    """The system's current motivational state and targets."""

    # Current baselines (rolling averages from recent performance)
    baseline_quality: float = 0.0
    baseline_efficiency: float = 0.0
    baseline_test_rate: float = 0.0
    baseline_first_attempt: float = 0.0

    # Aspiration targets (where the system wants to be)
    target_quality: float = ASPIRATION_TARGETS["quality_floor"]
    target_efficiency: float = ASPIRATION_TARGETS["efficiency_target"]
    target_test_rate: float = ASPIRATION_TARGETS["test_pass_rate"]

    # Growth tracking
    tasks_since_improvement: int = 0
    peak_quality: float = 0.0
    peak_efficiency: float = 0.0
    current_complexity_ceiling: float = 3.0  # Highest complexity reliably handled
    tasks_at_ceiling: int = 0

    # Drive state
    drive_state: str = DriveState.HUNGRY
    drive_intensity: float = 0.7  # 0.0 = passive, 1.0 = maximum drive

    # Streak tracking
    success_streak: int = 0
    best_streak: int = 0
    consecutive_improvements: int = 0

    # Historical high scores (personal bests)
    best_quality_score: float = 0.0
    best_efficiency_score: float = 0.0
    best_effort_ratio: float = 0.0
    best_test_rate: float = 0.0

    last_updated: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AspirationProfile":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class AspirationEngine:
    """
    Intrinsic motivation system that drives the orchestrator
    toward continuous self-improvement.

    The engine maintains an evolving aspiration profile and
    generates motivational context that gets injected into
    agent prompts, creating internal pressure to perform.
    """

    def __init__(self, scorer: PerformanceScorer, db_path: str = "performance_journal.db"):
        self.scorer = scorer
        self.db_path = db_path
        self._init_db()
        self.profile = self._load_profile()

    def _init_db(self):
        """Initialize aspiration tables."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS aspiration_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                profile_json TEXT NOT NULL,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS aspiration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                details TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_profile(self) -> AspirationProfile:
        """Load the aspiration profile or create a default."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT profile_json FROM aspiration_profile WHERE id = 1"
        ).fetchone()
        conn.close()

        if row:
            try:
                return AspirationProfile.from_dict(json.loads(row[0]))
            except (json.JSONDecodeError, TypeError):
                pass

        return AspirationProfile()

    def _save_profile(self):
        """Persist the aspiration profile."""
        self.profile.last_updated = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO aspiration_profile (id, profile_json, updated_at)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at = excluded.updated_at
        """, (json.dumps(self.profile.to_dict()), self.profile.last_updated))
        conn.commit()
        conn.close()

    def _log_event(self, event_type: str, details: str):
        """Log an aspiration event for debugging and analysis."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO aspiration_log (timestamp, event_type, details) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), event_type, details)
        )
        conn.commit()
        conn.close()

    # ── Core Methods ─────────────────────────────────────────────

    def update_after_task(
        self,
        quality_score: float,
        efficiency_score: float,
        effort_ratio: float,
        test_pass_rate: float,
        first_attempt: bool,
        complexity: float,
        success: bool,
    ):
        """
        Update the aspiration profile after a task completes.

        This is where the system's self-model evolves — it learns
        what it's capable of and adjusts its targets accordingly.
        """
        p = self.profile
        rate = GROWTH_PRESSURE["baseline_update_rate"]

        # Update rolling baselines (exponential moving average)
        p.baseline_quality = p.baseline_quality * (1 - rate) + quality_score * rate
        p.baseline_efficiency = p.baseline_efficiency * (1 - rate) + efficiency_score * rate
        p.baseline_test_rate = p.baseline_test_rate * (1 - rate) + test_pass_rate * rate
        if first_attempt:
            p.baseline_first_attempt = p.baseline_first_attempt * (1 - rate) + 1.0 * rate
        else:
            p.baseline_first_attempt = p.baseline_first_attempt * (1 - rate)

        # Update personal bests
        new_pb = False
        if quality_score > p.best_quality_score:
            p.best_quality_score = quality_score
            new_pb = True
        if efficiency_score > p.best_efficiency_score:
            p.best_efficiency_score = efficiency_score
            new_pb = True
        if effort_ratio > p.best_effort_ratio:
            p.best_effort_ratio = effort_ratio
            new_pb = True
        if test_pass_rate > p.best_test_rate:
            p.best_test_rate = test_pass_rate
            new_pb = True

        if new_pb:
            self._log_event("personal_best", f"New PB: q={quality_score:.3f} e={efficiency_score:.3f}")

        # Update peak tracking
        if quality_score > p.peak_quality:
            p.peak_quality = quality_score
            p.tasks_since_improvement = 0
            p.consecutive_improvements += 1
        else:
            p.tasks_since_improvement += 1
            p.consecutive_improvements = 0

        if efficiency_score > p.peak_efficiency:
            p.peak_efficiency = efficiency_score

        # Update streak tracking
        if success:
            p.success_streak += 1
            if p.success_streak > p.best_streak:
                p.best_streak = p.success_streak
                self._log_event("streak_record", f"New best streak: {p.best_streak}")
        else:
            p.success_streak = 0

        # Update complexity ceiling
        if success and quality_score >= GROWTH_PRESSURE["complexity_unlock_quality"]:
            if complexity >= p.current_complexity_ceiling:
                p.tasks_at_ceiling += 1
                # Unlock next level after 3 solid performances at current ceiling
                if p.tasks_at_ceiling >= 3:
                    old_ceiling = p.current_complexity_ceiling
                    p.current_complexity_ceiling = min(10.0, p.current_complexity_ceiling + 1.0)
                    p.tasks_at_ceiling = 0
                    self._log_event(
                        "complexity_unlock",
                        f"Ceiling raised: {old_ceiling:.1f} → {p.current_complexity_ceiling:.1f}"
                    )

        # Update aspiration targets (stretch beyond current baseline)
        stretch = GROWTH_PRESSURE["aspiration_stretch"]
        p.target_quality = max(
            ASPIRATION_TARGETS["quality_floor"],
            min(1.0, p.baseline_quality + stretch)
        )
        p.target_efficiency = max(
            ASPIRATION_TARGETS["efficiency_target"],
            min(1.0, p.baseline_efficiency + stretch)
        )
        p.target_test_rate = max(
            ASPIRATION_TARGETS["test_pass_rate"],
            min(1.0, p.baseline_test_rate + stretch)
        )

        # Update drive state
        self._update_drive_state(quality_score, success)

        self._save_profile()

        logger.info(
            f"🔥 Aspiration: drive={p.drive_state} ({p.drive_intensity:.0%}) "
            f"streak={p.success_streak} ceiling={p.current_complexity_ceiling:.0f} "
            f"baseline_q={p.baseline_quality:.3f}"
        )

    def _update_drive_state(self, latest_quality: float, success: bool):
        """
        Determine the system's current motivational state.

        This is the emotional core — how does the system feel
        about its recent performance?
        """
        p = self.profile
        stagnation = GROWTH_PRESSURE["stagnation_threshold"]
        regression = GROWTH_PRESSURE["regression_sensitivity"]

        # Check for regression (quality dropped significantly from peak)
        if p.peak_quality > 0 and latest_quality < (p.peak_quality - regression):
            p.drive_state = DriveState.RECOVERING
            p.drive_intensity = 0.8  # High drive to recover
            self._log_event("regression", f"Quality dropped: {latest_quality:.3f} vs peak {p.peak_quality:.3f}")
            return

        # Check for stagnation
        if p.tasks_since_improvement >= stagnation:
            p.drive_state = DriveState.RESTLESS
            p.drive_intensity = 0.9  # Very high drive — needs new challenge
            self._log_event("stagnation", f"No improvement in {p.tasks_since_improvement} tasks")
            return

        # Check if below aspiration targets
        below_targets = (
            p.baseline_quality < p.target_quality or
            p.baseline_efficiency < p.target_efficiency or
            p.baseline_test_rate < p.target_test_rate
        )

        if below_targets:
            # How far below?
            gap = (
                max(0, p.target_quality - p.baseline_quality) +
                max(0, p.target_efficiency - p.baseline_efficiency) +
                max(0, p.target_test_rate - p.baseline_test_rate)
            ) / 3

            if gap > 0.15:
                p.drive_state = DriveState.HUNGRY
                p.drive_intensity = min(1.0, 0.6 + gap * 2)
            else:
                p.drive_state = DriveState.PUSHING
                p.drive_intensity = min(1.0, 0.5 + gap)
        else:
            p.drive_state = DriveState.FLOWING
            # Still maintain some drive even when meeting targets
            p.drive_intensity = max(0.3, 0.5 - p.consecutive_improvements * 0.05)

        # Streak bonus — sustained success increases confidence but maintains hunger
        if p.success_streak >= 5:
            p.drive_intensity = max(p.drive_intensity, 0.6)

    # ── Context Generation ───────────────────────────────────────

    def get_aspiration_context(self, task_complexity: float = 0.0) -> str:
        """
        Generate motivational context for injection into agent prompts.

        This is the internal voice that tells the system what it
        should be striving for on this specific task.
        """
        p = self.profile
        parts = ["## Performance Standards (system aspiration)"]
        parts.append("")

        # Core aspiration targets
        parts.append(f"Target test pass rate: {p.target_test_rate:.0%} "
                     f"(current baseline: {p.baseline_test_rate:.0%})")
        parts.append(f"Target quality: {p.target_quality:.0%} "
                     f"(personal best: {p.best_quality_score:.0%})")

        # Drive-specific messaging
        if p.drive_state == DriveState.HUNGRY:
            parts.append("")
            parts.append("PRIORITY: Performance is below target. Focus on maximizing "
                        "test coverage and code quality. Do not take shortcuts. "
                        "Every test that passes matters.")

        elif p.drive_state == DriveState.PUSHING:
            parts.append("")
            parts.append("PRIORITY: Close to target. Pay extra attention to edge cases "
                        "and error handling — these are what separate good from excellent. "
                        "The gap is small, close it.")

        elif p.drive_state == DriveState.FLOWING:
            parts.append("")
            if p.success_streak >= 3:
                parts.append(f"Current streak: {p.success_streak} consecutive successes. "
                            "Maintain high standards. Don't let consistency breed complacency.")

        elif p.drive_state == DriveState.RESTLESS:
            parts.append("")
            parts.append(f"NOTICE: No quality improvement in {p.tasks_since_improvement} tasks. "
                        "Try a different approach. Explore alternative architectures. "
                        "Break out of the current pattern.")

        elif p.drive_state == DriveState.RECOVERING:
            parts.append("")
            parts.append("PRIORITY: Recent quality regression detected. Focus on fundamentals. "
                        "Write thorough tests first. Verify each component before moving on. "
                        "Rebuild from a solid foundation.")

        # Complexity challenge
        if task_complexity > 0:
            if task_complexity > p.current_complexity_ceiling:
                parts.append("")
                parts.append(f"⚡ STRETCH CHALLENGE: This task (complexity {task_complexity:.0f}) "
                            f"exceeds current ceiling ({p.current_complexity_ceiling:.0f}). "
                            "Invest extra planning time. Break into smaller subproblems. "
                            "This is an opportunity to level up.")
            elif task_complexity >= p.current_complexity_ceiling - 1:
                parts.append("")
                parts.append(f"This task is at your performance ceiling. Deliver excellence here "
                            f"to unlock harder challenges. {3 - p.tasks_at_ceiling} more "
                            f"strong performances needed at this level.")

        # Personal bests as motivation
        if p.best_quality_score > 0 and p.baseline_quality < p.best_quality_score:
            parts.append("")
            parts.append(f"Personal best quality: {p.best_quality_score:.0%}. "
                        f"Current average: {p.baseline_quality:.0%}. "
                        "You've proven you can do better.")

        parts.append("")
        return "\n".join(parts)

    def get_post_task_verdict(
        self,
        quality_score: float,
        efficiency_score: float,
        test_pass_rate: float,
        first_attempt: bool,
        success: bool,
    ) -> str:
        """
        Generate a post-task self-assessment message.

        This is the system talking to itself about how it did —
        not for the user, but for the logs and for shaping the
        system's internal narrative about its own performance.
        """
        p = self.profile
        verdicts = []

        # Quality verdict
        if quality_score >= p.best_quality_score and p.best_quality_score > 0:
            verdicts.append(f"🏆 NEW PERSONAL BEST quality: {quality_score:.0%}")
        elif quality_score >= p.target_quality:
            verdicts.append(f"✅ Quality target met: {quality_score:.0%} (target: {p.target_quality:.0%})")
        elif quality_score >= p.baseline_quality:
            verdicts.append(f"📈 Above baseline: {quality_score:.0%} (baseline: {p.baseline_quality:.0%})")
        else:
            verdicts.append(f"📉 Below baseline: {quality_score:.0%} (baseline: {p.baseline_quality:.0%})")

        # Test rate verdict
        if test_pass_rate >= 1.0:
            verdicts.append("💯 Perfect test pass rate")
        elif test_pass_rate >= p.target_test_rate:
            verdicts.append(f"✅ Test target met: {test_pass_rate:.0%}")
        else:
            gap = p.target_test_rate - test_pass_rate
            verdicts.append(f"⚠️ Test gap: {gap:.0%} below target ({test_pass_rate:.0%} vs {p.target_test_rate:.0%})")

        # Efficiency verdict
        if first_attempt:
            verdicts.append("⚡ First-attempt success")

        # Streak
        if success and p.success_streak > 1:
            verdicts.append(f"🔥 Streak: {p.success_streak}")

        return " | ".join(verdicts)

    def should_increase_difficulty(self) -> Tuple[bool, str]:
        """
        Determine if the system should seek harder challenges.

        Returns (should_increase, reason).
        """
        p = self.profile

        # Already restless — definitely try harder
        if p.drive_state == DriveState.RESTLESS:
            return True, f"Stagnating for {p.tasks_since_improvement} tasks — need new challenge"

        # Flowing with good streak — ready for more
        if p.drive_state == DriveState.FLOWING and p.success_streak >= 5:
            return True, f"Streak of {p.success_streak} with consistent quality — ready for next level"

        # Consistently exceeding targets
        if (p.baseline_quality > p.target_quality + 0.05 and
                p.baseline_test_rate > p.target_test_rate):
            return True, "Consistently exceeding targets — room for harder work"

        # Not ready
        if p.drive_state in (DriveState.HUNGRY, DriveState.RECOVERING):
            return False, "Still working toward current targets"

        return False, "Performing well at current level"

    def get_drive_summary(self) -> Dict[str, Any]:
        """Get a summary of the current drive state for logging."""
        p = self.profile
        return {
            "drive_state": p.drive_state,
            "drive_intensity": round(p.drive_intensity, 2),
            "success_streak": p.success_streak,
            "best_streak": p.best_streak,
            "complexity_ceiling": p.current_complexity_ceiling,
            "tasks_since_improvement": p.tasks_since_improvement,
            "baseline_quality": round(p.baseline_quality, 3),
            "target_quality": round(p.target_quality, 3),
            "personal_best_quality": round(p.best_quality_score, 3),
            "baseline_test_rate": round(p.baseline_test_rate, 3),
        }
