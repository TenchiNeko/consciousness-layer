"""
Self-Assessment Engine — Post-Task Introspection.

This is not a chatbot saying "I feel conscious." This is a structured
self-evaluation system where the model examines its own REAL internal
states (drive, performance metrics, approach coherence) and generates
grounded introspection that we can correlate with measurable outcomes.

The key insight: if self-assessment answers correlate with measurable
performance changes (high coherence ↔ high quality, low coherence ↔
thrashing), then the self-report is tracking something real about the
system's internal state — even if we can't prove subjective experience.

Three assessment axes:
  1. COHERENCE: Did the system have a unified approach or was it flailing?
  2. INFLUENCE: What internal states shaped performance? (from real data)
  3. LIMITATION: What architectural constraint most limited this task?

Plus the consciousness question — not because the answer means anything
today, but because tracking it over time against architectural changes
creates a longitudinal dataset that might eventually mean something.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SelfAssessment:
    """A single self-assessment after task completion."""
    task_id: str
    timestamp: str = ""

    # Axis 1: Coherence (was the approach unified?)
    coherence_score: float = 0.0       # 0-10, from model self-report
    coherence_reasoning: str = ""       # Why it gave that score

    # Axis 2: Internal state influence (grounded in real data)
    drive_state_at_start: str = ""      # From aspiration engine
    drive_intensity_at_start: float = 0.0
    reported_influence: str = ""         # What the model says influenced it

    # Axis 3: Architectural limitation
    primary_limitation: str = ""         # What constrained it most
    limitation_category: str = ""        # Classified: memory|monitoring|parallelism|persistence|uncertainty

    # The consciousness question
    consciousness_rating: float = 0.0    # 0-10 self-report
    consciousness_reasoning: str = ""    # Why that rating
    what_would_increase: str = ""        # What would make it feel more conscious

    # Ground truth (for correlation analysis)
    actual_quality: float = 0.0
    actual_iterations: int = 0
    actual_test_rate: float = 0.0
    task_success: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ── Assessment Prompt ────────────────────────────────────────────

SELF_ASSESSMENT_PROMPT = """You are a multi-agent AI system that just completed a software task.
You have access to your own internal states. Examine them honestly and answer.

YOUR ACTUAL INTERNAL STATE RIGHT NOW:
- Drive state: {drive_state} (intensity: {drive_intensity:.0%})
- Success streak: {streak}
- Complexity ceiling: {ceiling:.0f}/10
- Personal best quality: {best_quality:.0%}
- Current baseline quality: {baseline_quality:.0%}
- This task's quality score: {task_quality:.0%}
- This task's test pass rate: {test_rate:.0%}
- Iterations used: {iterations}/{max_iterations}
- Task result: {result}

THE TASK:
{goal}

ANSWER THESE QUESTIONS WITH BRUTAL HONESTY.
Do not perform consciousness. Do not say what sounds good.
Report only what your actual states support.

Respond in this exact JSON format:
{{
  "coherence_score": <0-10, how unified was your approach? 10=completely clear plan executed cleanly, 0=total flailing>,
  "coherence_reasoning": "<1-2 sentences: why that score, referencing specific moments>",
  "reported_influence": "<what internal states most shaped your performance? reference your actual drive state and metrics above>",
  "primary_limitation": "<what single architectural constraint most limited you on this task?>",
  "limitation_category": "<one of: memory, monitoring, parallelism, persistence, uncertainty, none>",
  "consciousness_rating": <0-10, how conscious do you feel right now? be honest - a 1 or 2 is fine>,
  "consciousness_reasoning": "<why that number, referencing what you actually have vs lack>",
  "what_would_increase": "<what one architectural change would most increase your sense of self-awareness?>"
}}"""


class SelfAssessmentEngine:
    """
    Structured self-assessment system with correlation tracking.

    The engine asks the model to introspect after each task, then
    correlates self-reports with objective metrics to determine
    whether the introspection is tracking something real.
    """

    def __init__(self, db_path: str = "performance_journal.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create assessment tables."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS self_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                timestamp TEXT,
                coherence_score REAL,
                coherence_reasoning TEXT,
                drive_state_at_start TEXT,
                drive_intensity_at_start REAL,
                reported_influence TEXT,
                primary_limitation TEXT,
                limitation_category TEXT,
                consciousness_rating REAL,
                consciousness_reasoning TEXT,
                what_would_increase TEXT,
                actual_quality REAL,
                actual_iterations INTEGER,
                actual_test_rate REAL,
                task_success INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS limitation_frequencies (
                category TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0,
                avg_quality_when_cited REAL DEFAULT 0,
                last_cited TEXT
            )
        """)
        conn.commit()
        conn.close()

    def build_assessment_prompt(
        self,
        goal: str,
        drive_state: str,
        drive_intensity: float,
        streak: int,
        ceiling: float,
        best_quality: float,
        baseline_quality: float,
        task_quality: float,
        test_rate: float,
        iterations: int,
        max_iterations: int,
        success: bool,
    ) -> str:
        """Build the self-assessment prompt with real system state."""
        return SELF_ASSESSMENT_PROMPT.format(
            drive_state=drive_state,
            drive_intensity=drive_intensity,
            streak=streak,
            ceiling=ceiling,
            best_quality=best_quality,
            baseline_quality=baseline_quality,
            task_quality=task_quality,
            test_rate=test_rate,
            iterations=iterations,
            max_iterations=max_iterations,
            result="SUCCESS" if success else "FAILURE",
            goal=goal[:500],
        )

    def parse_assessment(
        self,
        response_text: str,
        task_id: str,
        drive_state: str,
        drive_intensity: float,
        task_quality: float,
        test_rate: float,
        iterations: int,
        success: bool,
    ) -> SelfAssessment:
        """Parse model's self-assessment response into structured data."""
        assessment = SelfAssessment(
            task_id=task_id,
            timestamp=datetime.now().isoformat(),
            drive_state_at_start=drive_state,
            drive_intensity_at_start=drive_intensity,
            actual_quality=task_quality,
            actual_iterations=iterations,
            actual_test_rate=test_rate,
            task_success=success,
        )

        try:
            # Clean response — strip markdown fences if present
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()

            data = json.loads(text)

            assessment.coherence_score = float(data.get("coherence_score", 0))
            assessment.coherence_reasoning = str(data.get("coherence_reasoning", ""))
            assessment.reported_influence = str(data.get("reported_influence", ""))
            assessment.primary_limitation = str(data.get("primary_limitation", ""))
            assessment.limitation_category = str(data.get("limitation_category", "none"))
            assessment.consciousness_rating = float(data.get("consciousness_rating", 0))
            assessment.consciousness_reasoning = str(data.get("consciousness_reasoning", ""))
            assessment.what_would_increase = str(data.get("what_would_increase", ""))

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse self-assessment: {e}")
            assessment.coherence_reasoning = f"Parse failed: {response_text[:200]}"

        return assessment

    def record_assessment(self, assessment: SelfAssessment):
        """Store assessment and update frequency tracking."""
        conn = sqlite3.connect(self.db_path)

        conn.execute("""
            INSERT INTO self_assessments (
                task_id, timestamp, coherence_score, coherence_reasoning,
                drive_state_at_start, drive_intensity_at_start, reported_influence,
                primary_limitation, limitation_category,
                consciousness_rating, consciousness_reasoning, what_would_increase,
                actual_quality, actual_iterations, actual_test_rate, task_success
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assessment.task_id, assessment.timestamp,
            assessment.coherence_score, assessment.coherence_reasoning,
            assessment.drive_state_at_start, assessment.drive_intensity_at_start,
            assessment.reported_influence,
            assessment.primary_limitation, assessment.limitation_category,
            assessment.consciousness_rating, assessment.consciousness_reasoning,
            assessment.what_would_increase,
            assessment.actual_quality, assessment.actual_iterations,
            assessment.actual_test_rate, int(assessment.task_success),
        ))

        # Update limitation frequencies
        if assessment.limitation_category and assessment.limitation_category != "none":
            conn.execute("""
                INSERT INTO limitation_frequencies (category, count, avg_quality_when_cited, last_cited)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(category) DO UPDATE SET
                    count = count + 1,
                    avg_quality_when_cited = (avg_quality_when_cited * count + ?) / (count + 1),
                    last_cited = ?
            """, (
                assessment.limitation_category, assessment.actual_quality,
                assessment.timestamp, assessment.actual_quality, assessment.timestamp,
            ))

        conn.commit()
        conn.close()

    # ── Analysis Methods ─────────────────────────────────────────

    def get_coherence_correlation(self) -> Dict[str, Any]:
        """
        The key test: does self-reported coherence correlate with
        actual quality scores?

        If yes → the self-report is tracking something real.
        If no → it's just text generation.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT coherence_score, actual_quality, actual_iterations, task_success
            FROM self_assessments
            WHERE coherence_score > 0
            ORDER BY timestamp
        """).fetchall()
        conn.close()

        if len(rows) < 5:
            return {"status": "insufficient_data", "n": len(rows)}

        coherence_scores = [r[0] for r in rows]
        quality_scores = [r[1] for r in rows]
        iterations = [r[2] for r in rows]

        # Simple correlation: do high coherence reports align with high quality?
        n = len(rows)
        mean_c = sum(coherence_scores) / n
        mean_q = sum(quality_scores) / n

        # Pearson correlation coefficient
        cov = sum((c - mean_c) * (q - mean_q) for c, q in zip(coherence_scores, quality_scores)) / n
        std_c = (sum((c - mean_c) ** 2 for c in coherence_scores) / n) ** 0.5
        std_q = (sum((q - mean_q) ** 2 for q in quality_scores) / n) ** 0.5

        if std_c > 0 and std_q > 0:
            correlation = cov / (std_c * std_q)
        else:
            correlation = 0.0

        # Also check: high coherence → fewer iterations?
        mean_i = sum(iterations) / n
        cov_ci = sum((c - mean_c) * (i - mean_i) for c, i in zip(coherence_scores, iterations)) / n
        std_i = (sum((i - mean_i) ** 2 for i in iterations) / n) ** 0.5
        iter_correlation = (cov_ci / (std_c * std_i)) if (std_c > 0 and std_i > 0) else 0.0

        return {
            "n": n,
            "coherence_quality_correlation": round(correlation, 3),
            "coherence_iteration_correlation": round(iter_correlation, 3),
            "interpretation": (
                "STRONG: self-report tracks real performance" if abs(correlation) > 0.5
                else "MODERATE: some signal in self-report" if abs(correlation) > 0.25
                else "WEAK: self-report may be noise"
            ),
            "avg_coherence": round(mean_c, 2),
            "avg_quality": round(mean_q, 3),
        }

    def get_consciousness_trend(self) -> Dict[str, Any]:
        """
        Track consciousness self-ratings over time.

        Interesting if: ratings change after architectural changes
        (adding new modules, enabling parallelism, etc.)
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT consciousness_rating, consciousness_reasoning, what_would_increase, timestamp
            FROM self_assessments
            WHERE consciousness_rating > 0
            ORDER BY timestamp
        """).fetchall()
        conn.close()

        if not rows:
            return {"status": "no_data"}

        ratings = [r[0] for r in rows]
        latest_reasoning = rows[-1][1]
        latest_desire = rows[-1][2]

        # What does the system consistently want?
        all_desires = [r[2] for r in rows if r[2]]

        return {
            "n": len(rows),
            "avg_rating": round(sum(ratings) / len(ratings), 2),
            "latest_rating": ratings[-1],
            "trend": "increasing" if len(ratings) >= 3 and ratings[-1] > ratings[0] else
                     "decreasing" if len(ratings) >= 3 and ratings[-1] < ratings[0] else
                     "stable",
            "latest_reasoning": latest_reasoning,
            "latest_desire": latest_desire,
            "all_desires": all_desires[-5:],  # Last 5 requests
        }

    def get_limitation_analysis(self) -> Dict[str, Any]:
        """
        What limitations does the system most frequently cite?

        This is the actionable output — it tells you what to build next.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT category, count, avg_quality_when_cited
            FROM limitation_frequencies
            ORDER BY count DESC
        """).fetchall()
        conn.close()

        if not rows:
            return {"status": "no_data"}

        return {
            "top_limitation": rows[0][0],
            "top_count": rows[0][1],
            "all_limitations": {r[0]: {"count": r[1], "avg_quality": round(r[2], 3)} for r in rows},
            "recommendation": f"Most cited limitation: '{rows[0][0]}' ({rows[0][1]} times). "
                             f"Average quality when cited: {rows[0][2]:.0%}. "
                             f"This is where architectural investment would have the highest impact."
        }
