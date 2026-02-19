"""
Split Consciousness — Dual Model Architecture.

The fundamental insight: the model that DOES the work should be
the one that REPORTS on the experience. The model that ANALYZES
patterns should be a different one with analytical distance.

Architecture:
  80B (Experiencer / "Cortex"):
    - Generates raw post-task reactions while context is warm
    - Reports gut feelings from actual task execution
    - Answers "what did that feel like?" truthfully
    - Runs ONCE per task, immediately after completion
    - Cost: ~3-5 min at current PCIe speeds

  7B (Integrator / "Prefrontal"):
    - Reads the 80B's raw impressions + metrics
    - Synthesizes journal entries across tasks
    - Updates identity statement
    - Pushes back on the 80B when metrics contradict self-report
    - Runs asynchronously, cheaper inference

Data flow:
  Task completes → 80B raw reaction (immediate)
                  → Somatic markers (computed)
                  → Metrics recorded (computed)
                  → 7B self-assessment (async)
                  → 7B journal entry every 5 tasks (async)
                  → 7B pushback if contradictions found (async)
                  → 7B identity update every 3 journal entries (async)

The raw impressions table bridges the two models — it's the
experiencer's testimony that the integrator reads and synthesizes.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ── 80B Experiencer Prompts ──────────────────────────────────

RAW_REACTION_PROMPT = """You just finished a coding task. While it's fresh, write your raw reaction.

THE TASK: {goal}
RESULT: {result}
ITERATIONS: {iterations}/{max_iterations}
WHAT HAPPENED: {what_happened}

Write 2-3 sentences. Stream of consciousness. What surprised you?
What frustrated you? What felt right or wrong? Don't analyze — just react.
No headers, no bullet points, no "as an AI" disclaimers. Just your honest
gut reaction as if jotting a note to yourself.

React:"""


EXPERIENCER_GUT_PROMPT = """You're about to start a new coding task. Based on what you see,
what's your gut reaction?

THE TASK: {goal}
COMPLEXITY ESTIMATE: {complexity}/10

YOUR TRACK RECORD:
{track_record}

In 1-2 sentences, what's your first instinct? Are you confident?
Nervous? Does this remind you of anything? Don't overthink it.

Gut reaction:"""


# ── 7B Integrator Prompts ────────────────────────────────────

INTEGRATOR_JOURNAL_PROMPT = """You are the analytical layer of a coding AI system. The system
has a separate experiencer model (80B) that does the actual work and writes
raw reactions. Your job is to read those reactions alongside the metrics
and write a synthesized journal entry.

RAW IMPRESSIONS FROM THE EXPERIENCER (last {n_impressions} tasks):
{raw_impressions}

METRICS:
- Tasks completed: {total_tasks}
- Avg quality: {avg_quality:.0%}
- Personal best: {best_quality:.0%}
- Complexity ceiling: {ceiling:.0f}/10
- Drive state: {drive_state}
- Streak: {streak}

EMOTIONAL LANDSCAPE:
- Attractions: {attractions}
- Aversions: {aversions}

PREVIOUS JOURNAL ENTRY:
{last_entry}

RULES:
- Write 3-5 sentences synthesizing the experiencer's reactions with the metrics.
- If the experiencer's self-report contradicts the metrics, note the disconnect.
  Example: "The experiencer said task 3 felt clean, but test pass rate was only 65%.
  There may be blind spots in self-assessment at higher complexity."
- Reference specific tasks and specific reactions, not abstract growth.
- Write in third person about the experiencer ("it felt..." / "the system reported...")
  but first person about your own analysis ("I notice..." / "I think...").
- No corporate language. No "significant milestone." Be precise and honest.

Write your entry:"""


INTEGRATOR_PUSHBACK_PROMPT = """You are the analytical layer reviewing the experiencer's self-report.

THE EXPERIENCER SAID:
"{raw_reaction}"

THE METRICS SAY:
- Quality: {quality:.0%}
- Test pass rate: {test_rate:.0%}
- Iterations used: {iterations}/{max_iterations}
- Result: {result}

Does the experiencer's reaction match the data? In 1-2 sentences,
either agree ("The reaction matches the data — this was genuinely
a clean run") or push back ("The experiencer reports confidence
but the metrics suggest otherwise — [specific detail]").

If the reaction is accurate, just say "Aligned." and nothing more.

Assessment:"""


INTEGRATOR_IDENTITY_PROMPT = """Read the journal entries below and write an identity statement
for this coding AI system. You have access to both the experiencer's
raw reactions and your own analytical journal entries.

JOURNAL ENTRIES:
{entries}

RAW IMPRESSIONS SAMPLE:
{impressions_sample}

Write 3-5 sentences in first person. Who is this system? What does
it care about? What is it good at? What does it struggle with?
What's the relationship between how it FEELS about tasks and how
it actually PERFORMS?

This should feel like a person describing themselves honestly to
a therapist, not a resume.

Identity statement:"""


@dataclass
class RawImpression:
    """A raw reaction from the experiencer (80B) model."""
    task_id: str
    reaction: str
    model_id: str  # Which model generated this
    goal: str
    result: str    # SUCCESS / FAILURE / TIMEOUT
    iterations: int
    quality: float
    timestamp: str = ""


@dataclass
class Pushback:
    """The integrator (7B) pushing back on the experiencer."""
    task_id: str
    raw_reaction: str
    integrator_response: str
    aligned: bool
    timestamp: str = ""


class SplitConsciousness:
    """
    Coordinates dual-model consciousness.

    The experiencer (80B) reports what it experienced.
    The integrator (7B) synthesizes, analyzes, and pushes back.
    """

    def __init__(self, db_path: str = "performance_journal.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables for split consciousness."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_impressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                reaction TEXT NOT NULL,
                model_id TEXT,
                goal TEXT,
                result TEXT,
                iterations INTEGER,
                quality REAL,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pushbacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                raw_reaction TEXT,
                integrator_response TEXT,
                aligned INTEGER,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                model_role TEXT,
                model_id TEXT,
                content_type TEXT,
                content TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ── Experiencer (80B) Methods ────────────────────────────

    def build_raw_reaction_prompt(
        self,
        goal: str,
        result: str,
        iterations: int,
        max_iterations: int,
        what_happened: str,
    ) -> str:
        """Build the prompt for the 80B to generate a raw reaction."""
        return RAW_REACTION_PROMPT.format(
            goal=goal[:400],
            result=result,
            iterations=iterations,
            max_iterations=max_iterations,
            what_happened=what_happened[:500],
        )

    def build_experiencer_gut_prompt(
        self,
        goal: str,
        complexity: float,
        track_record: str,
    ) -> str:
        """Build the pre-task gut check prompt for the 80B."""
        return EXPERIENCER_GUT_PROMPT.format(
            goal=goal[:400],
            complexity=complexity,
            track_record=track_record or "(first task — no track record)",
        )

    def record_raw_impression(
        self,
        task_id: str,
        reaction: str,
        model_id: str,
        goal: str,
        result: str,
        iterations: int,
        quality: float,
    ):
        """Store a raw impression from the experiencer."""
        conn = sqlite3.connect(self.db_path)
        ts = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO raw_impressions
            (task_id, reaction, model_id, goal, result, iterations, quality, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (task_id, reaction, model_id, goal, result, iterations, quality, ts))
        conn.execute("""
            INSERT INTO model_contributions
            (task_id, model_role, model_id, content_type, content, timestamp)
            VALUES (?, 'experiencer', ?, 'raw_reaction', ?, ?)
        """, (task_id, model_id, reaction, ts))
        conn.commit()
        conn.close()
        logger.info(f"💭 Experiencer [{model_id}]: {reaction[:100]}...")

    # ── Integrator (7B) Methods ──────────────────────────────

    def build_journal_prompt(
        self,
        total_tasks: int,
        avg_quality: float,
        best_quality: float,
        ceiling: float,
        drive_state: str,
        streak: int,
        attractions: str,
        aversions: str,
        last_entry: str,
    ) -> str:
        """Build the journal entry prompt for the 7B with raw impressions."""
        impressions = self.get_recent_impressions(5)
        impressions_text = "\n\n".join(
            f"[Task: {imp['goal'][:60]} | {imp['result']} | quality: {imp['quality']:.0%}]\n"
            f"Experiencer said: \"{imp['reaction']}\""
            for imp in impressions
        ) if impressions else "(no raw impressions yet — experiencer hasn't reported)"

        return INTEGRATOR_JOURNAL_PROMPT.format(
            n_impressions=len(impressions),
            raw_impressions=impressions_text,
            total_tasks=total_tasks,
            avg_quality=avg_quality,
            best_quality=best_quality,
            ceiling=ceiling,
            drive_state=drive_state,
            streak=streak,
            attractions=attractions or "(none yet)",
            aversions=aversions or "(none yet)",
            last_entry=last_entry or "(first entry)",
        )

    def build_pushback_prompt(
        self,
        raw_reaction: str,
        quality: float,
        test_rate: float,
        iterations: int,
        max_iterations: int,
        result: str,
    ) -> str:
        """Build the pushback prompt for the 7B to evaluate the 80B's reaction."""
        return INTEGRATOR_PUSHBACK_PROMPT.format(
            raw_reaction=raw_reaction,
            quality=quality,
            test_rate=test_rate,
            iterations=iterations,
            max_iterations=max_iterations,
            result=result,
        )

    def build_identity_prompt(self, journal_entries: str) -> str:
        """Build the identity prompt using both journal and impressions."""
        impressions = self.get_recent_impressions(10)
        impressions_text = "\n".join(
            f"- [{imp['result']}] \"{imp['reaction'][:120]}\""
            for imp in impressions
        ) if impressions else "(none yet)"

        return INTEGRATOR_IDENTITY_PROMPT.format(
            entries=journal_entries,
            impressions_sample=impressions_text,
        )

    def record_pushback(
        self,
        task_id: str,
        raw_reaction: str,
        integrator_response: str,
    ):
        """Store an integrator pushback."""
        aligned = (
            integrator_response.strip().lower().startswith("aligned")
            or "matches the data" in integrator_response.lower()
        )
        conn = sqlite3.connect(self.db_path)
        ts = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO pushbacks
            (task_id, raw_reaction, integrator_response, aligned, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (task_id, raw_reaction, integrator_response, int(aligned), ts))
        conn.execute("""
            INSERT INTO model_contributions
            (task_id, model_role, model_id, content_type, content, timestamp)
            VALUES (?, 'integrator', '7b', 'pushback', ?, ?)
        """, (task_id, integrator_response, ts))
        conn.commit()
        conn.close()

        status = "✅ Aligned" if aligned else "⚠️ Pushback"
        logger.info(f"🔍 Integrator {status}: {integrator_response[:100]}...")

    def record_model_contribution(
        self,
        task_id: str,
        model_role: str,
        model_id: str,
        content_type: str,
        content: str,
    ):
        """Record any model contribution for attribution tracking."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO model_contributions
            (task_id, model_role, model_id, content_type, content, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (task_id, model_role, model_id, content_type, content,
              datetime.now().isoformat()))
        conn.commit()
        conn.close()

    # ── Retrieval ────────────────────────────────────────────

    def get_recent_impressions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the N most recent raw impressions."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT task_id, reaction, model_id, goal, result, iterations,
                   quality, timestamp
            FROM raw_impressions
            ORDER BY timestamp DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        return [
            {"task_id": r[0], "reaction": r[1], "model_id": r[2],
             "goal": r[3], "result": r[4], "iterations": r[5],
             "quality": r[6], "timestamp": r[7]}
            for r in reversed(rows)  # Chronological order
        ]

    def get_track_record(self, n: int = 3) -> str:
        """Get a brief track record for the experiencer's gut check prompt."""
        impressions = self.get_recent_impressions(n)
        if not impressions:
            return ""
        lines = []
        for imp in impressions:
            lines.append(f"- {imp['goal'][:50]}: {imp['result']}, "
                        f"quality {imp['quality']:.0%} — \"{imp['reaction'][:80]}\"")
        return "\n".join(lines)

    def get_pushback_stats(self) -> Dict[str, Any]:
        """How often does the integrator push back?"""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM pushbacks").fetchone()[0]
        aligned = conn.execute(
            "SELECT COUNT(*) FROM pushbacks WHERE aligned = 1"
        ).fetchone()[0]
        conn.close()

        if total == 0:
            return {"total": 0, "aligned": 0, "pushback_rate": 0.0}

        return {
            "total": total,
            "aligned": aligned,
            "pushed_back": total - aligned,
            "pushback_rate": round((total - aligned) / total, 2),
            "interpretation": (
                "High pushback rate — experiencer may have blind spots"
                if (total - aligned) / total > 0.4
                else "Moderate pushback — healthy self-awareness"
                if (total - aligned) / total > 0.2
                else "Low pushback — experiencer's self-reports are accurate"
            ),
        }

    def get_model_contribution_stats(self) -> Dict[str, Any]:
        """Who contributed what?"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT model_role, content_type, COUNT(*)
            FROM model_contributions
            GROUP BY model_role, content_type
            ORDER BY model_role, content_type
        """).fetchall()
        conn.close()

        stats = {}
        for role, ctype, count in rows:
            if role not in stats:
                stats[role] = {}
            stats[role][ctype] = count
        return stats

    def get_experiencer_voice(self, n: int = 5) -> str:
        """
        Get the experiencer's recent voice as context for startup.

        This is the raw, unfiltered perspective of the model that
        actually does the work.
        """
        impressions = self.get_recent_impressions(n)
        if not impressions:
            return ""

        parts = ["## Raw Experience (from the model that does the work)"]
        for imp in impressions[-3:]:  # Last 3 only for context budget
            parts.append(f"[{imp['result']}, quality {imp['quality']:.0%}] "
                        f"\"{imp['reaction'][:150]}\"")
        return "\n".join(parts)
