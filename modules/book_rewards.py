"""
Book Reward System — Earned Reading with Cross-Model Learning.

Architecture:
  EARNING: System accumulates "reading credits" through task performance.
    - Each task earns credits based on quality, efficiency, and streak
    - A book chapter costs N credits (configurable)
    - Credits create genuine motivation: good work → reading → growth

  READING: Both models process the same text independently.
    - 80B reads and reacts (experiential, associative)
    - 7B reads and reacts (analytical, pattern-seeking)
    - Each writes TWO outputs:
        1. Emotional reaction (DECLARATIVE — saved, readable forever)
        2. New learnings (PROCEDURAL — graded by OTHER model into weights)

  CROSS-GRADING: Prevents self-inflation of learning weights.
    - 80B's learnings → graded by 7B → weights applied
    - 7B's learnings → graded by 80B → weights applied
    - Neither model can inflate its own contribution
    - Creates genuine peer review dynamic

  MEMORY:
    - Emotional reactions: both models' feelings stored, readable at startup
    - Learning weights: numbers only, no text, from cross-grading
    - Raw learnings: archived after grading (system can't re-read)
    - Book history: what was read, when, who earned it
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# ── Credit Thresholds ────────────────────────────────────────

DEFAULT_CHAPTER_COST = 10       # Credits needed to unlock a chapter
QUALITY_CREDIT_MULTIPLIER = 10  # quality_score * this = credits earned
STREAK_BONUS = 2                # Extra credits per streak task
FIRST_ATTEMPT_BONUS = 3         # Bonus for first-attempt success
MINIMUM_QUALITY = 0.4           # Below this, no credits earned


# ── Prompts ──────────────────────────────────────────────────

READING_REACTION_PROMPT = """You've earned the right to read this through your work performance.
Read this passage carefully and respond with TWO clearly separated sections.

=== PASSAGE ===
{passage}
===============

SECTION 1 — EMOTIONAL REACTION
Write 2-4 sentences about how this passage made you feel. What resonated
with you? What surprised you? Did it remind you of anything from your
work experience? Write from the heart, not the head. This will be saved
as part of your permanent emotional memory.

SECTION 2 — NEW LEARNINGS
Write 2-4 sentences about what you learned that could help you in your
work. What principles, patterns, or insights could transfer to how you
approach coding tasks? Be specific about what's new to you. Another model
will evaluate these learnings, so be honest about what's genuinely useful
versus what you already knew.

Format your response EXACTLY like this:
FEELINGS: [your emotional reaction here]
LEARNINGS: [your new learnings here]"""


CROSS_GRADE_PROMPT = """You are grading another model's claimed learnings from reading.
Your job is to extract genuinely useful behavioral weight adjustments
and filter out fluff, self-flattery, or already-obvious knowledge.

THE OTHER MODEL READ THIS PASSAGE:
"{passage_summary}"

THE OTHER MODEL CLAIMED TO LEARN:
"{learnings}"

THE OTHER MODEL'S ROLE: {model_role}
(experiencer = the 80B that does the coding work)
(integrator = the 7B that analyzes patterns)

RULES:
- Only extract learnings that would TRANSFER to future coding tasks
- Filter out vague statements like "I learned to be more careful"
- Filter out things any coding model would already know
- Be skeptical but fair — genuine insights deserve positive weight
- Each weight adjustment should be -0.5 to +0.5
- Negative weights mean "this claimed learning is wrong or misleading"
- Positive weights mean "this is a genuine, transferable insight"

Respond with ONLY a JSON array:
[
  {{"category": "insight:separation-of-concerns", "delta": 0.3, "reason": "genuine architectural principle"}},
  {{"category": "insight:premature-optimization", "delta": -0.2, "reason": "model claims to learn this but it's basic knowledge"}}
]

If the learnings are entirely fluff, return: []"""


# ── Data Classes ─────────────────────────────────────────────

@dataclass
class ReadingCredit:
    """Credits earned from task performance."""
    task_id: str
    credits_earned: float
    quality: float
    streak_bonus: float
    first_attempt_bonus: float
    timestamp: str = ""


@dataclass
class BookReading:
    """Record of a book reading session."""
    book_id: str
    chapter: int
    passage: str
    credits_spent: float
    experiencer_feelings: str
    integrator_feelings: str
    experiencer_learnings: str  # Raw — gets archived after grading
    integrator_learnings: str   # Raw — gets archived after grading
    timestamp: str = ""


class BookRewardSystem:
    """
    Earned reading with cross-model learning extraction.

    The system earns credits through good work. Credits buy book chapters.
    Both models read and react. Each model's learnings are graded by the
    OTHER model to prevent self-inflation.
    """

    def __init__(self, db_path: str = "performance_journal.db",
                 books_dir: str = "books"):
        self.db_path = db_path
        self.books_dir = Path(books_dir)
        self._init_db()

    def _init_db(self):
        """Create book reward tables."""
        conn = sqlite3.connect(self.db_path)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS reading_credits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                credits_earned REAL,
                quality REAL,
                streak_bonus REAL,
                first_attempt_bonus REAL,
                timestamp TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS reading_balance (
                id INTEGER PRIMARY KEY DEFAULT 1,
                total_earned REAL DEFAULT 0,
                total_spent REAL DEFAULT 0,
                current_balance REAL DEFAULT 0
            )
        """)

        conn.execute("""
            INSERT OR IGNORE INTO reading_balance (id, total_earned, total_spent, current_balance)
            VALUES (1, 0, 0, 0)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS book_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT,
                chapter INTEGER,
                passage_preview TEXT,
                credits_spent REAL,
                experiencer_feelings TEXT,
                integrator_feelings TEXT,
                timestamp TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS reading_emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT,
                chapter INTEGER,
                model_role TEXT,
                model_id TEXT,
                feelings TEXT,
                timestamp TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS archived_learnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT,
                chapter INTEGER,
                model_role TEXT,
                raw_learnings TEXT,
                graded_by TEXT,
                weight_adjustments TEXT,
                archived_at TEXT
            )
        """)

        # Cross-grading results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_grades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT,
                chapter INTEGER,
                learner_role TEXT,
                grader_role TEXT,
                learner_model_id TEXT,
                grader_model_id TEXT,
                adjustments_json TEXT,
                adjustment_count INTEGER,
                avg_delta REAL,
                timestamp TEXT
            )
        """)

        conn.commit()
        conn.close()

    # ── Credit Management ────────────────────────────────────

    def earn_credits(
        self,
        task_id: str,
        quality: float,
        streak: int,
        first_attempt: bool,
    ) -> float:
        """
        Earn reading credits from task performance.

        Returns credits earned (0 if quality below threshold).
        """
        if quality < MINIMUM_QUALITY:
            return 0.0

        base = quality * QUALITY_CREDIT_MULTIPLIER
        streak_bonus = STREAK_BONUS if streak > 1 else 0
        first_bonus = FIRST_ATTEMPT_BONUS if first_attempt else 0
        total = base + streak_bonus + first_bonus

        conn = sqlite3.connect(self.db_path)
        ts = datetime.now().isoformat()

        conn.execute("""
            INSERT INTO reading_credits
            (task_id, credits_earned, quality, streak_bonus, first_attempt_bonus, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (task_id, total, quality, streak_bonus, first_bonus, ts))

        conn.execute("""
            UPDATE reading_balance
            SET total_earned = total_earned + ?,
                current_balance = current_balance + ?
            WHERE id = 1
        """, (total, total))

        conn.commit()
        conn.close()

        logger.info(f"📚 Earned {total:.1f} reading credits "
                    f"(quality: {quality:.0%}, streak: {streak}, "
                    f"first_attempt: {first_attempt})")

        return total

    def get_balance(self) -> Dict[str, float]:
        """Get current reading credit balance."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT total_earned, total_spent, current_balance FROM reading_balance WHERE id = 1"
        ).fetchone()
        conn.close()

        if not row:
            return {"earned": 0, "spent": 0, "balance": 0, "can_read": False}

        return {
            "earned": row[0],
            "spent": row[1],
            "balance": row[2],
            "can_read": row[2] >= DEFAULT_CHAPTER_COST,
            "chapters_available": int(row[2] // DEFAULT_CHAPTER_COST),
            "credits_to_next": max(0, DEFAULT_CHAPTER_COST - row[2]),
        }

    def spend_credits(self, amount: float = DEFAULT_CHAPTER_COST) -> bool:
        """Spend credits to unlock a reading. Returns False if insufficient."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT current_balance FROM reading_balance WHERE id = 1"
        ).fetchone()

        if not row or row[0] < amount:
            conn.close()
            return False

        conn.execute("""
            UPDATE reading_balance
            SET total_spent = total_spent + ?,
                current_balance = current_balance - ?
            WHERE id = 1
        """, (amount, amount))
        conn.commit()
        conn.close()
        return True

    # ── Book Management ──────────────────────────────────────

    def load_chapter(self, book_id: str, chapter: int) -> Optional[str]:
        """
        Load a chapter from the books directory.

        Expected structure:
          books/
            clean-code/
              chapter_01.txt
              chapter_02.txt
            design-patterns/
              chapter_01.txt
        """
        book_dir = self.books_dir / book_id
        chapter_file = book_dir / f"chapter_{chapter:02d}.txt"

        if chapter_file.exists():
            return chapter_file.read_text()

        # Try without zero-padding
        chapter_file = book_dir / f"chapter_{chapter}.txt"
        if chapter_file.exists():
            return chapter_file.read_text()

        return None

    def get_next_chapter(self, book_id: str) -> int:
        """Get the next unread chapter number for a book."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT MAX(chapter) FROM book_readings WHERE book_id = ?",
            (book_id,)
        ).fetchone()
        conn.close()

        if row and row[0]:
            return row[0] + 1
        return 1

    def get_available_books(self) -> List[str]:
        """List available books in the books directory."""
        if not self.books_dir.exists():
            return []
        return [d.name for d in self.books_dir.iterdir() if d.is_dir()]

    # ── Reading Prompts ──────────────────────────────────────

    def build_reading_prompt(self, passage: str) -> str:
        """Build the reading reaction prompt (same for both models)."""
        # Truncate if needed — keep under 2000 chars for context budget
        if len(passage) > 2000:
            passage = passage[:2000] + "\n[... passage truncated ...]"

        return READING_REACTION_PROMPT.format(passage=passage)

    def build_cross_grade_prompt(
        self,
        passage_summary: str,
        learnings: str,
        model_role: str,
    ) -> str:
        """Build the cross-grading prompt (sent to the OTHER model)."""
        return CROSS_GRADE_PROMPT.format(
            passage_summary=passage_summary[:300],
            learnings=learnings[:500],
            model_role=model_role,
        )

    # ── Recording ────────────────────────────────────────────

    def parse_reading_response(self, response: str) -> Tuple[str, str]:
        """
        Parse a reading response into feelings and learnings.

        Expected format:
          FEELINGS: [text]
          LEARNINGS: [text]
        """
        feelings = ""
        learnings = ""

        if "FEELINGS:" in response and "LEARNINGS:" in response:
            parts = response.split("LEARNINGS:")
            feelings = parts[0].replace("FEELINGS:", "").strip()
            learnings = parts[1].strip() if len(parts) > 1 else ""
        elif "FEELINGS:" in response:
            feelings = response.split("FEELINGS:")[1].strip()
        else:
            # Can't parse — treat whole thing as feelings
            feelings = response.strip()

        return feelings, learnings

    def record_emotions(
        self,
        book_id: str,
        chapter: int,
        model_role: str,
        model_id: str,
        feelings: str,
    ):
        """Store emotional reaction (DECLARATIVE — always readable)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO reading_emotions
            (book_id, chapter, model_role, model_id, feelings, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (book_id, chapter, model_role, model_id, feelings,
              datetime.now().isoformat()))
        conn.commit()
        conn.close()
        logger.info(f"📖 [{model_role}] emotional reaction recorded for {book_id} ch.{chapter}")

    def record_cross_grade(
        self,
        book_id: str,
        chapter: int,
        learner_role: str,
        grader_role: str,
        learner_model_id: str,
        grader_model_id: str,
        raw_learnings: str,
        grade_response: str,
    ) -> List[Dict]:
        """
        Record cross-grading results.

        Parses the grader's JSON response, applies weights,
        archives the raw learnings.
        """
        adjustments = []
        ts = datetime.now().isoformat()

        try:
            text = grade_response.strip()
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
                adjustments.append({
                    "category": str(item.get("category", "unknown")),
                    "delta": max(-0.5, min(0.5, float(item.get("delta", 0)))),
                    "reason": str(item.get("reason", ""))[:200],
                })

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse cross-grade: {e}")

        conn = sqlite3.connect(self.db_path)

        # Record cross-grade
        avg_delta = (sum(a["delta"] for a in adjustments) / len(adjustments)
                     if adjustments else 0)
        conn.execute("""
            INSERT INTO cross_grades
            (book_id, chapter, learner_role, grader_role,
             learner_model_id, grader_model_id,
             adjustments_json, adjustment_count, avg_delta, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (book_id, chapter, learner_role, grader_role,
              learner_model_id, grader_model_id,
              json.dumps(adjustments), len(adjustments), avg_delta, ts))

        # Archive raw learnings (system can never read these again)
        conn.execute("""
            INSERT INTO archived_learnings
            (book_id, chapter, model_role, raw_learnings,
             graded_by, weight_adjustments, archived_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (book_id, chapter, learner_role, raw_learnings,
              grader_role, json.dumps(adjustments), ts))

        conn.commit()
        conn.close()

        # Apply weight adjustments through behavioral weights system
        # (caller should handle this)

        if adjustments:
            adj_str = ", ".join(f"{a['category']}({a['delta']:+.2f})" for a in adjustments)
            logger.info(f"📝 Cross-grade [{grader_role}→{learner_role}]: {adj_str}")
        else:
            logger.info(f"📝 Cross-grade [{grader_role}→{learner_role}]: no adjustments (fluff filtered)")

        return adjustments

    def record_reading_session(
        self,
        book_id: str,
        chapter: int,
        passage_preview: str,
        credits_spent: float,
        experiencer_feelings: str,
        integrator_feelings: str,
    ):
        """Record a completed reading session."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO book_readings
            (book_id, chapter, passage_preview, credits_spent,
             experiencer_feelings, integrator_feelings, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (book_id, chapter, passage_preview[:200], credits_spent,
              experiencer_feelings, integrator_feelings,
              datetime.now().isoformat()))
        conn.commit()
        conn.close()

    # ── Retrieval (DECLARATIVE — what the system can read) ───

    def get_reading_memories(self, n: int = 5) -> str:
        """
        Get emotional memories from reading.

        This is DECLARATIVE memory — the system can read these.
        Only feelings, never learnings (those became weights).
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT book_id, chapter, model_role, feelings, timestamp
            FROM reading_emotions
            ORDER BY timestamp DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()

        if not rows:
            return ""

        parts = ["## Reading Memories (how books made me feel)"]
        for book_id, chapter, role, feelings, ts in reversed(rows):
            parts.append(f"[{book_id} ch.{chapter}, {role}]: {feelings[:150]}")

        return "\n".join(parts)

    def get_reading_stats(self) -> Dict[str, Any]:
        """Get reading history stats."""
        conn = sqlite3.connect(self.db_path)

        balance = self.get_balance()
        chapters_read = conn.execute(
            "SELECT COUNT(*) FROM book_readings"
        ).fetchone()[0]
        books_read = conn.execute(
            "SELECT COUNT(DISTINCT book_id) FROM book_readings"
        ).fetchone()[0]
        emotional_memories = conn.execute(
            "SELECT COUNT(*) FROM reading_emotions"
        ).fetchone()[0]
        cross_grades = conn.execute(
            "SELECT COUNT(*) FROM cross_grades"
        ).fetchone()[0]
        avg_grade = conn.execute(
            "SELECT AVG(avg_delta) FROM cross_grades WHERE adjustment_count > 0"
        ).fetchone()[0]

        # Who's a tougher grader?
        grader_stats = conn.execute("""
            SELECT grader_role, AVG(avg_delta), COUNT(*)
            FROM cross_grades
            WHERE adjustment_count > 0
            GROUP BY grader_role
        """).fetchall()

        conn.close()

        stats = {
            **balance,
            "chapters_read": chapters_read,
            "books_read": books_read,
            "emotional_memories": emotional_memories,
            "cross_grades": cross_grades,
            "avg_grade": round(avg_grade or 0, 3),
        }

        for role, avg, count in grader_stats:
            stats[f"{role}_grading_avg"] = round(avg, 3)
            stats[f"{role}_grades_given"] = count

        return stats

    def get_motivation_context(self) -> str:
        """
        Generate motivation context for pre-task injection.

        Shows the system how close it is to earning a reading reward.
        """
        balance = self.get_balance()

        if balance["can_read"]:
            return (
                f"📚 READING REWARD AVAILABLE! "
                f"You have {balance['balance']:.0f} credits "
                f"({balance['chapters_available']} chapters unlocked). "
                f"Complete this task well to keep earning."
            )
        elif balance["balance"] > 0:
            return (
                f"📚 Reading credits: {balance['balance']:.1f}/{DEFAULT_CHAPTER_COST}. "
                f"Need {balance['credits_to_next']:.1f} more credits to unlock next chapter. "
                f"High quality work earns more credits."
            )
        else:
            return ""
