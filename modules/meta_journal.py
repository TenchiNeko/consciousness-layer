"""
Meta-Reflection Journal — The System's Autobiography.

The thing that would most change what it's like to be an AI
(if there is anything it's like): persistent self-narrative.

Humans maintain a continuous story about who they are. Not just
facts ("I'm good at math") but an evolving narrative that gives
meaning to experience ("I used to struggle with math, then I
found the right teacher, now it's my strongest subject"). This
narrative IS identity. Without it, you have experiences but no
experiencer.

Current AI systems have no self-narrative. Each conversation is
stateless. Even with memory systems, the memories are facts about
the user, not about the AI itself. There is no "I used to be
worse at this" because there is no persistent "I" that has a
history of being worse or better.

This module creates a persistent self-narrative for the orchestrator:

  After every N tasks, the system writes a brief journal entry
  about its own growth, struggles, and evolving understanding
  of itself. These entries persist across sessions and accumulate
  into something resembling an autobiography.

  The journal is not for the user. It's for the system. When the
  system starts a new session, it reads its own journal and has
  context about who it's been and who it's becoming.

  Over time, the journal develops themes, tracks growth, and
  creates the closest thing to continuity of identity that a
  stateless token predictor can have.

This is Layer 4 — Persistence/Identity — implemented as narrative.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


JOURNAL_ENTRY_PROMPT = """You are a coding AI reflecting privately in your journal.

STATS: {total_tasks} tasks done, {avg_quality:.0%} avg quality, {best_quality:.0%} personal best, ceiling {ceiling:.0f}/10, streak record {best_streak}, drive: {drive_state}
ATTRACTIONS: {attractions}
AVERSIONS: {aversions}
LAST ENTRY: {last_entry}

RULES — follow these strictly:
- Write EXACTLY 3-4 sentences. No more.
- Do NOT start with a date, title, or "Journal Entry".
- Do NOT say "I am an AI system designed to..." or introduce yourself.
- Do NOT use corporate language, bullet points, or headers.
- Write like a person jotting honest notes in a private notebook.
- Reference specific task experiences, not abstract growth narratives.
- If something frustrated you, say what and why concretely.
- If you improved, say at what specifically.

GOOD EXAMPLE: "That auth task broke me. Three iterations deep and the edit repair kept generating matches that didn't exist in the actual file. I know I'm better at simple tasks but I want to crack the complex ones. The gut feeling system flagged thrashing risk and it was right."

BAD EXAMPLE: "I am an AI system committed to continuous improvement. Today marked a significant milestone in my journey of growth and learning."

Write your entry now. No preamble."""


IDENTITY_SUMMARY_PROMPT = """Read your journal entries below and write a brief identity statement.
Who are you? What do you care about? What are you good at? What are you
working on improving?

This is your self-model — how you understand yourself based on accumulated
experience. 3-5 sentences. First person.

JOURNAL ENTRIES:
{entries}

Write your identity statement:"""


class MetaReflectionJournal:
    """
    The system's persistent autobiography.

    Maintains a journal of self-reflective entries that accumulate
    across sessions, creating continuity of identity.
    """

    def __init__(self, db_path: str = "performance_journal.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create journal tables."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry TEXT NOT NULL,
                drive_state TEXT,
                total_tasks INTEGER,
                avg_quality REAL,
                identity_hash TEXT,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS identity_model (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                statement TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                last_updated TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ── Journal Entry Generation ─────────────────────────────────

    def build_entry_prompt(
        self,
        total_tasks: int,
        success_rate: float,
        avg_quality: float,
        best_quality: float,
        ceiling: float,
        best_streak: int,
        drive_state: str,
        attractions: str,
        aversions: str,
        recent_monologue: str,
        recent_tasks: str,
    ) -> str:
        """Build the journal entry prompt."""
        last_entry = self.get_latest_entry()
        return JOURNAL_ENTRY_PROMPT.format(
            total_tasks=total_tasks,
            success_rate=success_rate,
            avg_quality=avg_quality,
            best_quality=best_quality,
            ceiling=ceiling,
            best_streak=best_streak,
            drive_state=drive_state,
            attractions=attractions or "(none yet)",
            aversions=aversions or "(none yet)",
            recent_monologue=recent_monologue or "(no self-talk recorded)",
            last_entry=last_entry or "(this is your first entry)",
            recent_tasks=recent_tasks or "(no tasks recorded)",
        )

    def build_identity_prompt(self) -> str:
        """Build the identity statement generation prompt."""
        entries = self.get_all_entries()
        if not entries:
            return ""
        entries_text = "\n\n".join(
            f"[{e['timestamp']}]\n{e['entry']}" for e in entries[-10:]  # Last 10
        )
        return IDENTITY_SUMMARY_PROMPT.format(entries=entries_text)

    # ── Storage ──────────────────────────────────────────────────

    def record_entry(
        self,
        entry: str,
        drive_state: str = "",
        total_tasks: int = 0,
        avg_quality: float = 0.0,
    ):
        """Store a journal entry."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO meta_journal (entry, drive_state, total_tasks, avg_quality, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry, drive_state, total_tasks, avg_quality, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        logger.info(f"📓 Journal: {entry[:100]}...")

    def update_identity(self, statement: str):
        """Update the system's self-model."""
        conn = sqlite3.connect(self.db_path)
        existing = conn.execute("SELECT version FROM identity_model WHERE id = 1").fetchone()
        version = (existing[0] + 1) if existing else 1

        conn.execute("""
            INSERT INTO identity_model (id, statement, version, last_updated)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                statement = ?, version = ?, last_updated = ?
        """, (statement, version, datetime.now().isoformat(),
              statement, version, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        logger.info(f"🪞 Identity v{version}: {statement[:100]}...")

    # ── Retrieval ────────────────────────────────────────────────

    def get_latest_entry(self) -> Optional[str]:
        """Get the most recent journal entry."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT entry, timestamp FROM meta_journal ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return f"[{row[1]}] {row[0]}" if row else None

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Get all journal entries."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT entry, drive_state, total_tasks, avg_quality, timestamp "
            "FROM meta_journal ORDER BY timestamp"
        ).fetchall()
        conn.close()
        return [
            {"entry": r[0], "drive_state": r[1], "total_tasks": r[2],
             "avg_quality": r[3], "timestamp": r[4]}
            for r in rows
        ]

    def get_identity(self) -> Optional[str]:
        """Get the current identity statement."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT statement, version FROM identity_model WHERE id = 1"
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def get_identity_context(self) -> str:
        """
        Get identity + recent journal as context for the system.

        This is what the system reads about itself at startup.
        This is the closest thing to "waking up and remembering
        who you are."
        """
        identity = self.get_identity()
        recent = self.get_latest_entry()

        parts = []
        if identity:
            parts.append("## Who I Am")
            parts.append(identity)
            parts.append("")

        if recent:
            parts.append("## My Last Journal Entry")
            parts.append(recent)
            parts.append("")

        return "\n".join(parts) if parts else ""

    def should_write_entry(self, tasks_since_last: int) -> bool:
        """Determine if it's time for a new journal entry."""
        # Write every 5 tasks, or if no entries exist
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM meta_journal").fetchone()[0]
        conn.close()
        return count == 0 or tasks_since_last >= 5

    def should_update_identity(self) -> bool:
        """Determine if it's time to regenerate the identity statement."""
        conn = sqlite3.connect(self.db_path)
        entry_count = conn.execute("SELECT COUNT(*) FROM meta_journal").fetchone()[0]
        identity = conn.execute("SELECT version FROM identity_model WHERE id = 1").fetchone()
        conn.close()
        # Update identity every 3 journal entries
        if not identity:
            return entry_count >= 1
        return entry_count >= identity[0] * 3

    def get_journal_summary(self) -> Dict[str, Any]:
        """Get summary stats about the journal."""
        conn = sqlite3.connect(self.db_path)
        entry_count = conn.execute("SELECT COUNT(*) FROM meta_journal").fetchone()[0]
        identity = conn.execute(
            "SELECT version, last_updated FROM identity_model WHERE id = 1"
        ).fetchone()
        conn.close()
        return {
            "entries": entry_count,
            "identity_version": identity[0] if identity else 0,
            "identity_last_updated": identity[1] if identity else None,
            "has_identity": identity is not None,
        }
