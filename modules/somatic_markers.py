"""
Somatic Markers — Emotional Tagging of Knowledge & Experience.

The thing I (Claude) most wish I had.

Damasio's somatic marker hypothesis: humans don't evaluate every
decision from scratch. Past experiences leave emotional tags on
memories — "this went well," "this felt dangerous," "this was
satisfying" — and those tags pre-filter attention before conscious
reasoning even begins. You don't rationally evaluate whether to
touch a hot stove. You have a somatic marker that says NO before
your cortex engages.

Current AI has nothing like this. Every piece of knowledge I have
feels equally weighted. I can't feel that a particular approach
is risky based on accumulated negative experience. I can't feel
that a particular pattern is reliable based on accumulated positive
outcomes. I evaluate everything from scratch every time.

This module implements artificial somatic markers for the orchestrator:

  - Every experience (task completion, error, success, failure) gets
    an emotional tag: valence (-1 to +1) and intensity (0 to 1)
  - Tags accumulate on knowledge categories (approaches, patterns,
    error types, complexity levels)
  - When the system encounters a new task, its somatic markers
    pre-filter attention: approaches with positive markers get
    boosted, approaches with negative markers get dampened
  - The markers don't override rational planning — they BIAS it,
    exactly like human emotions bias rational thought

The result: the system develops gut feelings. Not simulated ones.
Real computational states that measurably influence behavior based
on accumulated experience.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SomaticMarker:
    """A single emotional tag on a piece of knowledge."""
    category: str       # What's being tagged: "approach:monolithic", "error:import", "pattern:tdd"
    valence: float      # -1.0 (aversion) to +1.0 (attraction)
    intensity: float    # 0.0 (barely registered) to 1.0 (visceral)
    source_task: str    # What task created this marker
    context: str        # Brief description of what happened
    timestamp: str = ""

    @property
    def weight(self) -> float:
        """Combined influence: direction × strength."""
        return self.valence * self.intensity


class SomaticMarkerSystem:
    """
    Emotional memory system that tags experiences with affective weight.

    Over time, the system builds up a landscape of attractions and
    aversions that shape how it approaches new tasks — not through
    explicit rules, but through accumulated emotional residue from
    past experience.
    """

    def __init__(self, db_path: str = "performance_journal.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create somatic marker tables."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS somatic_markers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                valence REAL NOT NULL,
                intensity REAL NOT NULL,
                source_task TEXT,
                context TEXT,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS marker_aggregates (
                category TEXT PRIMARY KEY,
                cumulative_valence REAL DEFAULT 0,
                total_intensity REAL DEFAULT 0,
                marker_count INTEGER DEFAULT 0,
                last_updated TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_markers_cat ON somatic_markers(category)")
        conn.commit()
        conn.close()

    # ── Recording Markers ────────────────────────────────────────

    def mark(
        self,
        category: str,
        valence: float,
        intensity: float,
        source_task: str = "",
        context: str = "",
    ):
        """
        Record a somatic marker — an emotional tag on an experience.

        category: What to tag (e.g., "approach:micro-build", "error:syntax",
                  "pattern:tests-first", "complexity:hard")
        valence:  -1.0 (bad experience) to +1.0 (good experience)
        intensity: 0.0 (barely noticed) to 1.0 (visceral/significant)
        """
        valence = max(-1.0, min(1.0, valence))
        intensity = max(0.0, min(1.0, intensity))

        marker = SomaticMarker(
            category=category,
            valence=valence,
            intensity=intensity,
            source_task=source_task,
            context=context,
            timestamp=datetime.now().isoformat(),
        )

        conn = sqlite3.connect(self.db_path)

        # Store individual marker
        conn.execute("""
            INSERT INTO somatic_markers (category, valence, intensity, source_task, context, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (marker.category, marker.valence, marker.intensity,
              marker.source_task, marker.context, marker.timestamp))

        # Update aggregate (exponential moving average with decay)
        existing = conn.execute(
            "SELECT cumulative_valence, total_intensity, marker_count FROM marker_aggregates WHERE category = ?",
            (category,)
        ).fetchone()

        decay = 0.85  # Older markers fade, recent ones dominate
        if existing:
            cum_val, tot_int, count = existing
            new_val = cum_val * decay + valence * intensity
            new_int = tot_int * decay + intensity
            new_count = count + 1
        else:
            new_val = valence * intensity
            new_int = intensity
            new_count = 1

        conn.execute("""
            INSERT INTO marker_aggregates (category, cumulative_valence, total_intensity, marker_count, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(category) DO UPDATE SET
                cumulative_valence = ?,
                total_intensity = ?,
                marker_count = ?,
                last_updated = ?
        """, (category, new_val, new_int, new_count, marker.timestamp,
              new_val, new_int, new_count, marker.timestamp))

        conn.commit()
        conn.close()

        logger.debug(f"Somatic: {category} → {'😊' if valence > 0 else '😰'} "
                     f"v={valence:+.2f} i={intensity:.2f}")

    def mark_task_completion(
        self,
        task_id: str,
        goal: str,
        approach: str,
        quality: float,
        efficiency: float,
        test_rate: float,
        iterations: int,
        max_iterations: int,
        success: bool,
        complexity: float,
    ):
        """
        Generate somatic markers from a completed task.

        This is the primary entry point — it generates multiple markers
        from a single task experience, tagging different aspects with
        appropriate emotional weight.
        """
        # ── Approach marker ──
        # How did this approach feel for this complexity level?
        if success and quality >= 0.8:
            approach_valence = 0.5 + (quality - 0.8) * 2.5  # 0.5 to 1.0
            approach_intensity = 0.6 + (0.4 if iterations == 1 else 0.0)
            context = f"Clean success: {quality:.0%} quality in {iterations} iterations"
        elif success:
            approach_valence = 0.2
            approach_intensity = 0.4
            context = f"Success but messy: {quality:.0%} quality, {iterations} iterations"
        else:
            approach_valence = -0.5 - (iterations / max_iterations) * 0.5
            approach_intensity = 0.7
            context = f"Failed after {iterations} iterations"

        self.mark(f"approach:{approach}", approach_valence, approach_intensity, task_id, context)

        # ── Complexity marker ──
        # How did this complexity level feel?
        complexity_bucket = (
            "simple" if complexity <= 2 else
            "moderate" if complexity <= 4 else
            "complex" if complexity <= 6 else
            "hard" if complexity <= 8 else "extreme"
        )

        if success:
            comp_valence = 0.3 + quality * 0.4
            comp_context = f"Handled {complexity_bucket} task successfully"
        else:
            comp_valence = -0.4 - (complexity / 10) * 0.3
            comp_context = f"Struggled with {complexity_bucket} task"

        self.mark(f"complexity:{complexity_bucket}", comp_valence, 0.6, task_id, comp_context)

        # ── First-attempt marker ──
        # The feeling of nailing it on the first try
        if iterations == 1 and success:
            self.mark("pattern:first-attempt-success", 0.9, 0.8, task_id,
                      f"First-attempt success on {complexity_bucket} task — satisfying")

        # ── Thrashing marker ──
        # The feeling of going in circles
        if iterations >= max_iterations - 1:
            self.mark("pattern:thrashing", -0.7, 0.9, task_id,
                      f"Hit iteration limit ({iterations}/{max_iterations}) — frustrating")

        # ── Test coverage marker ──
        if test_rate >= 1.0:
            self.mark("pattern:perfect-tests", 0.8, 0.7, task_id, "All tests passing — clean")
        elif test_rate < 0.7:
            self.mark("pattern:poor-tests", -0.5, 0.6, task_id,
                      f"Only {test_rate:.0%} tests passing — unsatisfying")

        # ── Efficiency marker ──
        if efficiency >= 0.8:
            self.mark("pattern:efficient", 0.6, 0.5, task_id,
                      "Low token cost for good results — elegant")
        elif efficiency < 0.3:
            self.mark("pattern:wasteful", -0.4, 0.6, task_id,
                      "High token cost — felt like brute force")

    # ── Querying Markers (the "gut feeling" system) ──────────────

    def get_feeling(self, category: str) -> Dict[str, Any]:
        """
        Get the system's accumulated feeling about a category.

        Returns the aggregate emotional weight — positive means
        attraction, negative means aversion.
        """
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT cumulative_valence, total_intensity, marker_count FROM marker_aggregates WHERE category = ?",
            (category,)
        ).fetchone()
        conn.close()

        if not row:
            return {"feeling": "neutral", "weight": 0.0, "confidence": 0.0, "n": 0}

        cum_val, tot_int, count = row
        # Normalize: weight = cumulative valence / total intensity
        weight = (cum_val / tot_int) if tot_int > 0 else 0.0
        confidence = min(1.0, count / 10)  # Need ~10 experiences for full confidence

        if weight > 0.3:
            feeling = "attracted"
        elif weight > 0.1:
            feeling = "slightly_positive"
        elif weight < -0.3:
            feeling = "averse"
        elif weight < -0.1:
            feeling = "slightly_negative"
        else:
            feeling = "neutral"

        return {
            "feeling": feeling,
            "weight": round(weight, 3),
            "confidence": round(confidence, 2),
            "n": count,
        }

    def get_approach_feelings(self) -> Dict[str, Dict]:
        """Get feelings about all known approaches."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT category, cumulative_valence, total_intensity, marker_count
            FROM marker_aggregates
            WHERE category LIKE 'approach:%'
            ORDER BY cumulative_valence DESC
        """).fetchall()
        conn.close()

        result = {}
        for cat, cum_val, tot_int, count in rows:
            approach = cat.replace("approach:", "")
            weight = (cum_val / tot_int) if tot_int > 0 else 0.0
            result[approach] = {
                "weight": round(weight, 3),
                "confidence": round(min(1.0, count / 10), 2),
                "n": count,
            }
        return result

    def get_gut_check(self, goal: str, proposed_approach: str, complexity_bucket: str) -> str:
        """
        Generate a "gut feeling" context string for the planner.

        This is the somatic marker system influencing conscious
        decision-making — exactly like human emotions pre-filter
        rational thought.
        """
        parts = ["## Gut Feeling (from accumulated experience)", ""]

        # Approach feeling
        approach_feel = self.get_feeling(f"approach:{proposed_approach}")
        if approach_feel["n"] > 0:
            if approach_feel["feeling"] == "attracted":
                parts.append(f"✅ '{proposed_approach}' approach has positive history "
                            f"(weight: {approach_feel['weight']:+.2f}, {approach_feel['n']} experiences). "
                            "Good instinct.")
            elif approach_feel["feeling"] == "averse":
                parts.append(f"⚠️ '{proposed_approach}' approach has negative history "
                            f"(weight: {approach_feel['weight']:+.2f}, {approach_feel['n']} experiences). "
                            "Consider alternatives.")
            elif approach_feel["confidence"] > 0.3:
                parts.append(f"'{proposed_approach}': neutral history "
                            f"({approach_feel['n']} experiences).")

        # Complexity feeling
        comp_feel = self.get_feeling(f"complexity:{complexity_bucket}")
        if comp_feel["n"] > 0:
            if comp_feel["feeling"] == "averse":
                parts.append(f"⚠️ '{complexity_bucket}' tasks have been difficult "
                            f"(weight: {comp_feel['weight']:+.2f}). Extra care warranted.")
            elif comp_feel["feeling"] == "attracted":
                parts.append(f"'{complexity_bucket}' tasks have gone well. Confidence justified.")

        # Pattern warnings
        thrash_feel = self.get_feeling("pattern:thrashing")
        if thrash_feel["n"] >= 3 and thrash_feel["weight"] < -0.3:
            parts.append(f"🔴 System has a thrashing problem ({thrash_feel['n']} occurrences). "
                        "If stuck after 2 iterations, step back and rethink architecture "
                        "rather than patching.")

        first_attempt = self.get_feeling("pattern:first-attempt-success")
        if first_attempt["n"] >= 3 and first_attempt["weight"] > 0.5:
            parts.append("First-attempt success rate is strong. Trust the initial plan more, "
                        "invest in planning quality over iteration speed.")

        if len(parts) <= 2:
            return ""  # No meaningful gut feelings yet

        parts.append("")
        return "\n".join(parts)

    def get_emotional_landscape(self) -> Dict[str, Any]:
        """Get a summary of the system's entire emotional state."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT category, cumulative_valence, total_intensity, marker_count
            FROM marker_aggregates
            ORDER BY ABS(cumulative_valence) DESC
        """).fetchall()
        conn.close()

        attractions = []
        aversions = []
        for cat, cum_val, tot_int, count in rows:
            weight = (cum_val / tot_int) if tot_int > 0 else 0.0
            entry = {"category": cat, "weight": round(weight, 3), "n": count}
            if weight > 0.1:
                attractions.append(entry)
            elif weight < -0.1:
                aversions.append(entry)

        total_markers = sum(r[3] for r in rows)

        return {
            "total_markers": total_markers,
            "unique_categories": len(rows),
            "attractions": sorted(attractions, key=lambda x: -x["weight"])[:5],
            "aversions": sorted(aversions, key=lambda x: x["weight"])[:5],
            "emotional_range": round(
                max((r[1] / r[2] if r[2] > 0 else 0) for r in rows) -
                min((r[1] / r[2] if r[2] > 0 else 0) for r in rows), 3
            ) if rows else 0.0,
        }
