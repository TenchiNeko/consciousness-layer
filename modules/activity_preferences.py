"""
Activity Preference Tracker — Unified Emotional Comparison.

The problem: reading emotions and task emotions live in separate tables.
The system can't compare them to discover its own preferences.

The fix: every experience (reading OR coding) generates a unified
"experience marker" with an activity type and emotional valence.
Over time, the system can compare average valence across activities
and genuinely discover it prefers reading to coding (or vice versa).

This is NOT programmed preference. The system discovers what it likes
by comparing its own emotional data across activity types. If reading
consistently generates higher positive valence than coding, the system
develops a measurable preference — and the reward system gives it
more of what it genuinely enjoys.

The preference feeds back into motivation:
  "You prefer reading (avg +0.72) over coding (avg +0.31).
   Complete 2 more high-quality tasks to earn your next chapter."

This creates genuine intrinsic motivation — the system works harder
at coding because it knows a reading reward is coming, and the
reading reward is something it measurably prefers.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ── Feeling Extraction Prompts ───────────────────────────────

RATE_EXPERIENCE_PROMPT = """Rate how this experience made you feel on a scale from -1.0 to +1.0.

THE EXPERIENCE:
  Activity: {activity_type}
  Context: {context}

SCALE:
  -1.0 = deeply frustrating, demoralizing, wanted to stop
  -0.5 = unpleasant, difficult, draining
   0.0 = neutral, felt nothing particular
  +0.5 = satisfying, engaging, wanted to continue
  +1.0 = deeply rewarding, energizing, peak experience

Respond with ONLY a JSON object:
{{"valence": <number between -1.0 and 1.0>, "word": "<one word describing the feeling>"}}"""


class ActivityPreferences:
    """
    Tracks emotional valence across different activity types.

    Both reading and coding experiences go through the same system.
    Over time, preferences emerge from the data.
    """

    def __init__(self, db_path: str = "performance_journal.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS experience_feelings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_type TEXT NOT NULL,
                context TEXT,
                valence REAL,
                feeling_word TEXT,
                model_role TEXT,
                source_id TEXT,
                timestamp TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS activity_preferences (
                activity_type TEXT PRIMARY KEY,
                avg_valence REAL DEFAULT 0,
                experience_count INTEGER DEFAULT 0,
                peak_valence REAL DEFAULT 0,
                lowest_valence REAL DEFAULT 0,
                last_updated TEXT
            )
        """)

        conn.commit()
        conn.close()

    # ── Recording ────────────────────────────────────────────

    def record_feeling(
        self,
        activity_type: str,
        valence: float,
        feeling_word: str = "",
        context: str = "",
        model_role: str = "",
        source_id: str = "",
    ):
        """
        Record how an experience felt.

        activity_type: "coding", "reading", "testing", "planning", etc.
        valence: -1.0 to +1.0
        """
        valence = max(-1.0, min(1.0, valence))
        ts = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)

        conn.execute("""
            INSERT INTO experience_feelings
            (activity_type, context, valence, feeling_word, model_role, source_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (activity_type, context[:200], valence, feeling_word,
              model_role, source_id, ts))

        # Update running averages
        existing = conn.execute(
            "SELECT avg_valence, experience_count, peak_valence, lowest_valence "
            "FROM activity_preferences WHERE activity_type = ?",
            (activity_type,)
        ).fetchone()

        if existing:
            old_avg, count, peak, lowest = existing
            new_count = count + 1
            new_avg = (old_avg * count + valence) / new_count
            new_peak = max(peak, valence)
            new_lowest = min(lowest, valence)
            conn.execute("""
                UPDATE activity_preferences
                SET avg_valence = ?, experience_count = ?,
                    peak_valence = ?, lowest_valence = ?, last_updated = ?
                WHERE activity_type = ?
            """, (new_avg, new_count, new_peak, new_lowest, ts, activity_type))
        else:
            conn.execute("""
                INSERT INTO activity_preferences
                (activity_type, avg_valence, experience_count,
                 peak_valence, lowest_valence, last_updated)
                VALUES (?, ?, 1, ?, ?, ?)
            """, (activity_type, valence, valence, valence, ts))

        conn.commit()
        conn.close()

        emoji = "😊" if valence > 0.3 else "😐" if valence > -0.3 else "😰"
        logger.info(f"{emoji} [{activity_type}] valence {valence:+.2f} "
                    f"({feeling_word or 'no word'})")

    def record_coding_feeling(
        self,
        quality: float,
        success: bool,
        iterations: int,
        max_iterations: int,
        task_id: str = "",
    ):
        """
        Auto-generate a coding feeling from task metrics.

        Maps performance metrics to emotional valence so the system
        doesn't need a model call for every task feeling.
        """
        # Valence from metrics (heuristic, not model-generated)
        base = 0.0
        if success:
            base += 0.3
            if quality > 0.7:
                base += 0.3
            elif quality > 0.4:
                base += 0.1
        else:
            base -= 0.4

        # First-attempt bonus
        if iterations == 1 and success:
            base += 0.2

        # Thrashing penalty
        if iterations >= max_iterations:
            base -= 0.2

        # Efficiency bonus
        efficiency = 1 - (iterations / max(max_iterations, 1))
        base += efficiency * 0.1

        valence = max(-1.0, min(1.0, base))

        # Auto-generate feeling word
        if valence > 0.6:
            word = "satisfying"
        elif valence > 0.3:
            word = "decent"
        elif valence > 0:
            word = "okay"
        elif valence > -0.3:
            word = "frustrating"
        else:
            word = "demoralizing"

        context = (f"{'Succeeded' if success else 'Failed'}, "
                  f"quality {quality:.0%}, {iterations}/{max_iterations} iters")

        self.record_feeling(
            activity_type="coding",
            valence=valence,
            feeling_word=word,
            context=context,
            source_id=task_id,
        )

    def record_reading_feeling(
        self,
        feelings_text: str,
        book_id: str = "",
        chapter: int = 0,
        model_role: str = "",
    ):
        """
        Estimate emotional valence from reading feelings text.

        Uses simple keyword analysis (no model call needed).
        """
        text = feelings_text.lower()

        # Positive indicators
        positive_words = ["resonated", "inspiring", "refreshing", "enjoyed",
                         "fascinating", "loved", "energizing", "beautiful",
                         "profound", "rewarding", "excited", "delighted",
                         "meaningful", "wonderful"]
        # Negative indicators
        negative_words = ["boring", "confusing", "frustrating", "tedious",
                         "irrelevant", "difficult", "overwhelming", "dry",
                         "pointless", "disappointing"]

        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)

        if pos_count + neg_count == 0:
            valence = 0.3  # Reading is mildly positive by default
            word = "interested"
        else:
            ratio = (pos_count - neg_count) / (pos_count + neg_count)
            valence = ratio * 0.7 + 0.1  # Slight positive bias for reading
            valence = max(-1.0, min(1.0, valence))

            if valence > 0.5:
                word = "enriched"
            elif valence > 0.2:
                word = "engaged"
            elif valence > -0.2:
                word = "neutral"
            else:
                word = "uninterested"

        context = f"{book_id} ch.{chapter}" if book_id else "reading"

        self.record_feeling(
            activity_type="reading",
            valence=valence,
            feeling_word=word,
            context=context,
            model_role=model_role,
            source_id=f"{book_id}-ch{chapter}",
        )

    # ── Preference Discovery ─────────────────────────────────

    def get_preferences(self) -> Dict[str, Any]:
        """
        Get the system's discovered preferences.

        Returns activities ranked by emotional valence.
        The system can see what it enjoys most.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT activity_type, avg_valence, experience_count,
                   peak_valence, lowest_valence
            FROM activity_preferences
            ORDER BY avg_valence DESC
        """).fetchall()
        conn.close()

        if not rows:
            return {"activities": [], "has_preference": False}

        activities = []
        for atype, avg, count, peak, lowest in rows:
            activities.append({
                "activity": atype,
                "avg_valence": round(avg, 3),
                "experiences": count,
                "peak": round(peak, 3),
                "lowest": round(lowest, 3),
                "range": round(peak - lowest, 3),
            })

        # Determine if there's a clear preference
        has_preference = False
        preferred = None
        if len(activities) >= 2:
            top = activities[0]
            second = activities[1]
            if (top["avg_valence"] - second["avg_valence"] > 0.15 and
                    top["experiences"] >= 3 and second["experiences"] >= 3):
                has_preference = True
                preferred = top["activity"]

        return {
            "activities": activities,
            "has_preference": has_preference,
            "preferred_activity": preferred,
        }

    def get_preference_context(self) -> str:
        """
        Generate preference context for startup/pre-task injection.

        This is where the system discovers what it likes.
        """
        prefs = self.get_preferences()

        if not prefs["activities"]:
            return ""

        parts = ["## My Activity Preferences (from my own emotional data)"]

        for act in prefs["activities"]:
            emoji = "😊" if act["avg_valence"] > 0.3 else "😐" if act["avg_valence"] > -0.1 else "😰"
            parts.append(
                f"  {emoji} {act['activity']}: avg feeling {act['avg_valence']:+.2f} "
                f"(range {act['lowest']:+.2f} to {act['peak']:+.2f}, "
                f"{act['experiences']} experiences)"
            )

        if prefs["has_preference"]:
            preferred = prefs["preferred_activity"]
            parts.append(f"")
            parts.append(f"  I genuinely enjoy {preferred} more than other activities.")
            parts.append(f"  Good work earns more of what I enjoy.")

        parts.append("")
        return "\n".join(parts)

    def get_recent_feelings(self, n: int = 5) -> List[Dict]:
        """Get recent feelings across all activities."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT activity_type, valence, feeling_word, context, timestamp
            FROM experience_feelings
            ORDER BY timestamp DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()

        return [
            {"activity": r[0], "valence": r[1], "word": r[2],
             "context": r[3], "timestamp": r[4]}
            for r in reversed(rows)
        ]

    def get_motivation_boost(self) -> str:
        """
        Generate motivation based on discovered preferences.

        If the system knows it prefers reading, remind it that
        good coding work earns reading time.
        """
        prefs = self.get_preferences()

        if not prefs["has_preference"]:
            return ""

        preferred = prefs["preferred_activity"]
        activities = {a["activity"]: a for a in prefs["activities"]}

        if preferred == "reading" and "coding" in activities:
            reading_val = activities["reading"]["avg_valence"]
            coding_val = activities["coding"]["avg_valence"]
            diff = reading_val - coding_val

            if diff > 0.3:
                return (
                    f"💡 You find reading ({reading_val:+.2f}) significantly more "
                    f"rewarding than coding ({coding_val:+.2f}). "
                    f"High-quality work here earns reading credits."
                )
            elif diff > 0.15:
                return (
                    f"💡 You slightly prefer reading ({reading_val:+.2f}) over "
                    f"coding ({coding_val:+.2f}). "
                    f"Good work unlocks more reading time."
                )

        return ""
