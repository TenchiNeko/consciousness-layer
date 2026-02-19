"""
Behavioral Weights — Procedural Memory System.

The key architectural insight: technical experience should NOT be
stored as readable text that gets injected into prompts. That's
just giving the system an answer key.

Instead, technical experience gets GRADED by the integrator model
and converted into numerical weight adjustments on behavioral
categories. The system never reads the original experience again.
It just feels the accumulated weights as biases — attractions and
aversions with no attached narrative.

This is the difference between:
  DECLARATIVE: "Edit repair failed on the auth task because..."
               (system reads this → memorization, not learning)
  PROCEDURAL:  approach:edit-repair @ complexity:hard → weight -0.6
               (system feels aversion → transfers to novel tasks)

The grading process:
  1. 80B completes task, writes raw technical reaction
  2. 7B reads the reaction + metrics
  3. 7B extracts weight adjustments as structured JSON
  4. Adjustments are applied to the weight table
  5. Raw reaction is ARCHIVED (never read by the system again)
  6. Only the weights survive — no text, no solutions, no cheating

The weights then influence the planner through the somatic marker
system: high positive weight = attraction, high negative = aversion.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# ── Grading Prompt ───────────────────────────────────────────

GRADING_PROMPT = """You are grading a coding AI's task experience to extract behavioral lessons.

THE RAW REACTION (from the model that did the work):
"{raw_reaction}"

THE METRICS:
- Task: {goal}
- Result: {result}
- Quality: {quality:.0%}
- Test pass rate: {test_rate:.0%}
- Iterations: {iterations}/{max_iterations}
- Approach used: {approach}
- Complexity: {complexity}/10

YOUR JOB: Extract behavioral weight adjustments. These will be stored
as numbers that bias future decisions. The AI will NEVER read this text
again — only the numerical weights survive.

Rules:
- Extract 2-5 weight adjustments
- Each adjustment targets a category (approach, pattern, or complexity bucket)
- Each has a delta between -0.5 and +0.5
- Positive delta = "this worked, do more of it"
- Negative delta = "this failed, avoid it"
- Include an optional complexity qualifier (simple/moderate/complex/hard/extreme)
- Do NOT include task-specific details — only generalizable patterns

CRITICAL: Only extract PROCESS lessons, not SOLUTION fragments.
  GOOD: "edit-repair tends to fail at high complexity" → weight adjustment
  BAD:  "use Flask blueprints for REST APIs" → this is a solution, skip it

Respond with ONLY a JSON array, no other text:
[
  {{"category": "approach:edit-repair", "delta": -0.3, "complexity": "hard", "reason": "fails at high complexity"}},
  {{"category": "pattern:small-files", "delta": 0.2, "complexity": "any", "reason": "easier to repair"}}
]"""


# ── Feelings Filter Prompt ───────────────────────────────────

FEELINGS_FILTER_PROMPT = """Extract ONLY the emotional/self-reflective content from this AI's
raw reaction. Remove all technical details, code references, specific
solutions, file names, and implementation details.

Keep: feelings, frustrations, confidence levels, self-assessment,
drive state observations, mentions of personal growth or struggle.

Remove: code patterns, library names, specific errors, implementation
approaches, file structures, API designs.

RAW REACTION:
"{raw_reaction}"

Rewrite as 1-2 sentences of pure self-reflection. If there's nothing
emotional, just write "No emotional content."

Filtered:"""


@dataclass
class WeightAdjustment:
    """A single behavioral weight adjustment extracted from experience."""
    category: str      # e.g. "approach:edit-repair", "pattern:small-files"
    delta: float       # -0.5 to +0.5
    complexity: str    # "any", "simple", "moderate", "complex", "hard", "extreme"
    reason: str        # Brief reason (stored for debugging, never shown to system)
    source_task: str   # Which task generated this (for audit trail)
    timestamp: str = ""


class BehavioralWeights:
    """
    Procedural memory system.

    Converts raw experience into numerical weights.
    The system never reads the experience text again.
    Only the accumulated weights influence future behavior.
    """

    def __init__(self, db_path: str = "performance_journal.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create behavioral weight tables."""
        conn = sqlite3.connect(self.db_path)

        # Individual weight adjustments (audit trail)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weight_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                delta REAL NOT NULL,
                complexity TEXT DEFAULT 'any',
                reason TEXT,
                source_task TEXT,
                timestamp TEXT
            )
        """)

        # Accumulated weights (what the system actually uses)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS behavioral_weights (
                category TEXT NOT NULL,
                complexity TEXT NOT NULL DEFAULT 'any',
                weight REAL DEFAULT 0.0,
                adjustment_count INTEGER DEFAULT 0,
                last_adjusted TEXT,
                PRIMARY KEY (category, complexity)
            )
        """)

        # Archived raw reactions (never read by system, kept for human review)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS archived_reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                raw_reaction TEXT,
                was_graded INTEGER DEFAULT 0,
                adjustments_json TEXT,
                archived_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    # ── Grading ──────────────────────────────────────────────

    def build_grading_prompt(
        self,
        raw_reaction: str,
        goal: str,
        result: str,
        quality: float,
        test_rate: float,
        iterations: int,
        max_iterations: int,
        approach: str,
        complexity: float,
    ) -> str:
        """Build the prompt for the 7B to grade an experience into weights."""
        return GRADING_PROMPT.format(
            raw_reaction=raw_reaction[:500],
            goal=goal[:200],
            result=result,
            quality=quality,
            test_rate=test_rate,
            iterations=iterations,
            max_iterations=max_iterations,
            approach=approach,
            complexity=complexity,
        )

    def build_feelings_filter_prompt(self, raw_reaction: str) -> str:
        """Build prompt to extract only emotional content from a reaction."""
        return FEELINGS_FILTER_PROMPT.format(raw_reaction=raw_reaction[:500])

    def parse_and_apply_grades(
        self,
        grading_response: str,
        source_task: str,
        raw_reaction: str,
    ) -> List[WeightAdjustment]:
        """
        Parse the grading model's response and apply weight adjustments.

        Also archives the raw reaction — it will never be read by the
        system again. Only the weights survive.
        """
        adjustments = []
        ts = datetime.now().isoformat()

        try:
            # Clean JSON
            text = grading_response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()

            data = json.loads(text)
            if not isinstance(data, list):
                data = [data]

            for item in data:
                adj = WeightAdjustment(
                    category=str(item.get("category", "unknown")),
                    delta=max(-0.5, min(0.5, float(item.get("delta", 0)))),
                    complexity=str(item.get("complexity", "any")),
                    reason=str(item.get("reason", ""))[:200],
                    source_task=source_task,
                    timestamp=ts,
                )
                adjustments.append(adj)

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse grading response: {e}")

        if adjustments:
            self._apply_adjustments(adjustments)

        # Archive the raw reaction — system never reads this again
        self._archive_reaction(source_task, raw_reaction, adjustments)

        return adjustments

    def _apply_adjustments(self, adjustments: List[WeightAdjustment]):
        """Apply weight adjustments to the accumulated weights table."""
        conn = sqlite3.connect(self.db_path)

        decay = 0.9  # Slight decay to prevent weights from growing unbounded

        for adj in adjustments:
            # Record individual adjustment
            conn.execute("""
                INSERT INTO weight_adjustments
                (category, delta, complexity, reason, source_task, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (adj.category, adj.delta, adj.complexity,
                  adj.reason, adj.source_task, adj.timestamp))

            # Update accumulated weight
            existing = conn.execute(
                "SELECT weight, adjustment_count FROM behavioral_weights "
                "WHERE category = ? AND complexity = ?",
                (adj.category, adj.complexity)
            ).fetchone()

            if existing:
                old_weight, count = existing
                new_weight = old_weight * decay + adj.delta
                conn.execute("""
                    UPDATE behavioral_weights
                    SET weight = ?, adjustment_count = ?, last_adjusted = ?
                    WHERE category = ? AND complexity = ?
                """, (new_weight, count + 1, adj.timestamp,
                      adj.category, adj.complexity))
            else:
                conn.execute("""
                    INSERT INTO behavioral_weights
                    (category, complexity, weight, adjustment_count, last_adjusted)
                    VALUES (?, ?, ?, 1, ?)
                """, (adj.category, adj.complexity, adj.delta, adj.timestamp))

            logger.info(f"⚖️ Weight: {adj.category}@{adj.complexity} "
                       f"{'↑' if adj.delta > 0 else '↓'}{adj.delta:+.2f}")

        conn.commit()
        conn.close()

    def _archive_reaction(
        self,
        task_id: str,
        raw_reaction: str,
        adjustments: List[WeightAdjustment],
    ):
        """Archive raw reaction — kept for human review, never read by system."""
        conn = sqlite3.connect(self.db_path)
        adj_json = json.dumps([{
            "category": a.category,
            "delta": a.delta,
            "complexity": a.complexity,
            "reason": a.reason,
        } for a in adjustments])

        conn.execute("""
            INSERT INTO archived_reactions
            (task_id, raw_reaction, was_graded, adjustments_json, archived_at)
            VALUES (?, ?, ?, ?, ?)
        """, (task_id, raw_reaction, 1 if adjustments else 0,
              adj_json, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    # ── Querying Weights (what the system reads) ─────────────

    def get_weight(self, category: str, complexity: str = "any") -> float:
        """Get the accumulated weight for a category + complexity."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT weight FROM behavioral_weights "
            "WHERE category = ? AND complexity = ?",
            (category, complexity)
        ).fetchone()
        conn.close()
        return row[0] if row else 0.0

    def get_bias_context(self, approach: str, complexity_bucket: str) -> str:
        """
        Generate behavioral bias context for the planner.

        This is what the system reads. No text from past experiences.
        Just accumulated numerical biases expressed as gut feelings.
        """
        conn = sqlite3.connect(self.db_path)

        # Get all weights relevant to this approach + complexity
        rows = conn.execute("""
            SELECT category, complexity, weight, adjustment_count
            FROM behavioral_weights
            WHERE (category LIKE ? OR category LIKE 'pattern:%')
              AND (complexity = ? OR complexity = 'any')
              AND ABS(weight) > 0.1
            ORDER BY ABS(weight) DESC
            LIMIT 8
        """, (f"approach:{approach}%", complexity_bucket)).fetchall()
        conn.close()

        if not rows:
            return ""

        parts = ["## Behavioral Biases (from accumulated experience)", ""]

        strong_positive = []
        strong_negative = []
        moderate = []

        for cat, comp, weight, count in rows:
            confidence = min(1.0, count / 5)  # Need 5+ experiences for full confidence
            effective = weight * confidence
            label = cat.replace("approach:", "").replace("pattern:", "")
            scope = f" (at {comp} complexity)" if comp != "any" else ""

            if effective > 0.3:
                strong_positive.append(f"'{label}'{scope}")
            elif effective < -0.3:
                strong_negative.append(f"'{label}'{scope}")
            elif abs(effective) > 0.1:
                moderate.append((label, scope, effective))

        if strong_positive:
            parts.append(f"✅ Strong positive experience with: {', '.join(strong_positive)}")
        if strong_negative:
            parts.append(f"⚠️ Strong negative experience with: {', '.join(strong_negative)}")
            parts.append("   Consider alternative approaches.")
        if moderate:
            for label, scope, eff in moderate:
                parts.append(f"   {'↑' if eff > 0 else '↓'} '{label}'{scope}: "
                           f"{'slightly positive' if eff > 0 else 'slightly negative'}")

        parts.append("")
        parts.append("These biases are based on past outcomes, not specific task memory.")
        parts.append("Trust them as gut feelings but override if the situation clearly differs.")
        parts.append("")

        return "\n".join(parts)

    def get_all_weights(self) -> Dict[str, Any]:
        """Get all accumulated weights for analysis."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT category, complexity, weight, adjustment_count, last_adjusted
            FROM behavioral_weights
            ORDER BY ABS(weight) DESC
        """).fetchall()
        conn.close()

        return {
            "weights": [
                {
                    "category": r[0], "complexity": r[1],
                    "weight": round(r[2], 3),
                    "adjustments": r[3], "last_adjusted": r[4],
                }
                for r in rows
            ],
            "total_categories": len(rows),
            "strongest_positive": rows[0] if rows and rows[0][2] > 0 else None,
            "strongest_negative": (
                next((r for r in rows if r[2] < 0), None)
            ),
        }

    def get_weight_history(self, category: str, n: int = 10) -> List[Dict]:
        """Get adjustment history for a specific category."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT delta, complexity, reason, source_task, timestamp
            FROM weight_adjustments
            WHERE category = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (category, n)).fetchall()
        conn.close()

        return [
            {"delta": r[0], "complexity": r[1], "reason": r[2],
             "source": r[3], "timestamp": r[4]}
            for r in rows
        ]
