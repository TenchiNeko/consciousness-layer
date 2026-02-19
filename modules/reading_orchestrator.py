"""
Reading Orchestrator — Runs the full book reward pipeline.

Called by the consciousness layer after task completion when
the system has earned enough credits.

Pipeline per reading session:
  1. Check balance — enough credits for a chapter?
  2. Pick book + load chapter
  3. Send passage to 80B → get feelings + learnings
  4. Send passage to 7B → get feelings + learnings
  5. Cross-grade: 7B grades 80B's learnings → weights
  6. Cross-grade: 80B grades 7B's learnings → weights
  7. Store feelings (declarative — permanent)
  8. Archive learnings (procedural — system never re-reads)
  9. Apply weight adjustments through behavioral_weights

The cross-grading is the key innovation:
  - 80B can't inflate its own learning weights
  - 7B can't inflate its own learning weights
  - Each model keeps the other honest
  - Creates genuine peer review dynamic
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

from book_rewards import BookRewardSystem
from behavioral_weights import BehavioralWeights

logger = logging.getLogger(__name__)

EXPERIENCER = "experiencer"
INTEGRATOR = "integrator"


class ModelPrompt:
    """A prompt tagged with which model should run it."""
    def __init__(self, role: str, prompt: str, purpose: str,
                 temperature: float = 0.4):
        self.role = role
        self.prompt = prompt
        self.purpose = purpose
        self.temperature = temperature


class ReadingOrchestrator:
    """
    Coordinates the full reading reward pipeline.

    Returns a sequence of ModelPrompt objects. The main orchestrator
    sends each to the appropriate model and feeds responses back.

    State machine:
      IDLE → earned credits → CHECK_BALANCE
      CHECK_BALANCE → has enough → LOAD_CHAPTER
      LOAD_CHAPTER → chapter loaded → EXPERIENCER_READS
      EXPERIENCER_READS → response → INTEGRATOR_READS
      INTEGRATOR_READS → response → CROSS_GRADE_80B (7B grades 80B's learnings)
      CROSS_GRADE_80B → response → CROSS_GRADE_7B (80B grades 7B's learnings)
      CROSS_GRADE_7B → response → APPLY_WEIGHTS → IDLE
    """

    def __init__(self, db_path: str = "performance_journal.db",
                 books_dir: str = "books"):
        self.books = BookRewardSystem(db_path=db_path, books_dir=books_dir)
        self.weights = BehavioralWeights(db_path=db_path)

        # Reading session state
        self._current_book: str = ""
        self._current_chapter: int = 0
        self._current_passage: str = ""
        self._experiencer_feelings: str = ""
        self._experiencer_learnings: str = ""
        self._integrator_feelings: str = ""
        self._integrator_learnings: str = ""

    # ── Credit Earning (called after each task) ──────────────

    def earn_from_task(
        self,
        task_id: str,
        quality: float,
        streak: int,
        first_attempt: bool,
    ) -> Dict[str, Any]:
        """
        Earn reading credits from task performance.

        Returns credit info including whether a reading is now available.
        """
        credits = self.books.earn_credits(
            task_id=task_id,
            quality=quality,
            streak=streak,
            first_attempt=first_attempt,
        )

        balance = self.books.get_balance()

        if balance["can_read"]:
            logger.info(f"📚 READING UNLOCKED! Balance: {balance['balance']:.0f} credits")

        return {
            "credits_earned": credits,
            **balance,
        }

    # ── Reading Session ──────────────────────────────────────

    def can_read(self) -> bool:
        """Check if the system has enough credits to read."""
        return self.books.get_balance().get("can_read", False)

    def start_reading_session(self, book_id: Optional[str] = None) -> Optional[List[ModelPrompt]]:
        """
        Start a reading session if credits allow.

        Returns the first prompts (both models read the same passage)
        or None if can't read.
        """
        if not self.can_read():
            return None

        # Pick book
        if not book_id:
            available = self.books.get_available_books()
            if not available:
                logger.warning("📚 No books available in books directory")
                return None
            book_id = available[0]  # Could be smarter — preference-based

        # Get next chapter
        chapter = self.books.get_next_chapter(book_id)
        passage = self.books.load_chapter(book_id, chapter)

        if not passage:
            logger.info(f"📚 No more chapters in {book_id}")
            return None

        # Spend credits
        if not self.books.spend_credits():
            return None

        # Store session state
        self._current_book = book_id
        self._current_chapter = chapter
        self._current_passage = passage

        logger.info(f"📖 Reading: {book_id} chapter {chapter} ({len(passage)} chars)")

        # Build reading prompt (same for both models)
        reading_prompt = self.books.build_reading_prompt(passage)

        # Return both prompts — orchestrator sends them in sequence
        return [
            ModelPrompt(
                role=EXPERIENCER,
                prompt=reading_prompt,
                purpose="book_reading_experiencer",
                temperature=0.6,  # More creative/associative for reading
            ),
            ModelPrompt(
                role=INTEGRATOR,
                prompt=reading_prompt,
                purpose="book_reading_integrator",
                temperature=0.5,
            ),
        ]

    def record_reading_response(
        self,
        purpose: str,
        response_text: str,
        model_id: str = "",
    ) -> Optional[ModelPrompt]:
        """
        Record a reading response and return the next prompt.

        Flow:
          book_reading_experiencer → store, return None (wait for integrator)
          book_reading_integrator → store, return cross_grade_experiencer
          cross_grade_experiencer → apply weights, return cross_grade_integrator
          cross_grade_integrator → apply weights, finalize, return None
        """

        if purpose == "book_reading_experiencer":
            feelings, learnings = self.books.parse_reading_response(response_text)
            self._experiencer_feelings = feelings
            self._experiencer_learnings = learnings

            # Store emotional reaction (permanent, declarative)
            self.books.record_emotions(
                book_id=self._current_book,
                chapter=self._current_chapter,
                model_role=EXPERIENCER,
                model_id=model_id,
                feelings=feelings,
            )

            logger.info(f"📖 80B feels: {feelings[:100]}...")
            logger.info(f"📖 80B learned: {learnings[:100]}...")

            # Don't return cross-grade yet — wait for integrator to read too
            return None

        elif purpose == "book_reading_integrator":
            feelings, learnings = self.books.parse_reading_response(response_text)
            self._integrator_feelings = feelings
            self._integrator_learnings = learnings

            # Store emotional reaction (permanent, declarative)
            self.books.record_emotions(
                book_id=self._current_book,
                chapter=self._current_chapter,
                model_role=INTEGRATOR,
                model_id=model_id,
                feelings=feelings,
            )

            logger.info(f"📖 7B feels: {feelings[:100]}...")
            logger.info(f"📖 7B learned: {learnings[:100]}...")

            # Now cross-grade: 7B grades 80B's learnings
            if self._experiencer_learnings:
                return ModelPrompt(
                    role=INTEGRATOR,  # 7B grades 80B's learnings
                    prompt=self.books.build_cross_grade_prompt(
                        passage_summary=self._current_passage[:200],
                        learnings=self._experiencer_learnings,
                        model_role=EXPERIENCER,
                    ),
                    purpose="cross_grade_experiencer",
                    temperature=0.2,
                )
            else:
                # 80B didn't claim to learn anything — skip to 80B grading 7B
                if self._integrator_learnings:
                    return ModelPrompt(
                        role=EXPERIENCER,  # 80B grades 7B's learnings
                        prompt=self.books.build_cross_grade_prompt(
                            passage_summary=self._current_passage[:200],
                            learnings=self._integrator_learnings,
                            model_role=INTEGRATOR,
                        ),
                        purpose="cross_grade_integrator",
                        temperature=0.2,
                    )
                else:
                    self._finalize_reading()
                    return None

        elif purpose == "cross_grade_experiencer":
            # 7B just graded 80B's learnings → apply weights
            adjustments = self.books.record_cross_grade(
                book_id=self._current_book,
                chapter=self._current_chapter,
                learner_role=EXPERIENCER,
                grader_role=INTEGRATOR,
                learner_model_id="80B",
                grader_model_id=model_id,
                raw_learnings=self._experiencer_learnings,
                grade_response=response_text,
            )

            # Apply weights through behavioral weights system
            self._apply_reading_weights(adjustments, "experiencer")

            # Now 80B grades 7B's learnings
            if self._integrator_learnings:
                return ModelPrompt(
                    role=EXPERIENCER,  # 80B grades 7B's learnings
                    prompt=self.books.build_cross_grade_prompt(
                        passage_summary=self._current_passage[:200],
                        learnings=self._integrator_learnings,
                        model_role=INTEGRATOR,
                    ),
                    purpose="cross_grade_integrator",
                    temperature=0.2,
                )
            else:
                self._finalize_reading()
                return None

        elif purpose == "cross_grade_integrator":
            # 80B just graded 7B's learnings → apply weights
            adjustments = self.books.record_cross_grade(
                book_id=self._current_book,
                chapter=self._current_chapter,
                learner_role=INTEGRATOR,
                grader_role=EXPERIENCER,
                learner_model_id="7B",
                grader_model_id=model_id,
                raw_learnings=self._integrator_learnings,
                grade_response=response_text,
            )

            # Apply weights
            self._apply_reading_weights(adjustments, "integrator")

            # Done — finalize
            self._finalize_reading()
            return None

        return None

    def _apply_reading_weights(self, adjustments: List[Dict], source: str):
        """Apply weight adjustments from cross-grading."""
        from behavioral_weights import WeightAdjustment
        from datetime import datetime

        ts = datetime.now().isoformat()
        weight_adjs = []

        for adj in adjustments:
            w = WeightAdjustment(
                category=f"reading:{adj['category']}",
                delta=adj["delta"],
                complexity="any",
                reason=f"from {source}'s reading, cross-graded",
                source_task=f"reading-{self._current_book}-ch{self._current_chapter}",
                timestamp=ts,
            )
            weight_adjs.append(w)

        if weight_adjs:
            self.weights._apply_adjustments(weight_adjs)
            logger.info(f"⚖️ Applied {len(weight_adjs)} reading weights from {source}")

    def _finalize_reading(self):
        """Finalize a reading session."""
        self.books.record_reading_session(
            book_id=self._current_book,
            chapter=self._current_chapter,
            passage_preview=self._current_passage[:200],
            credits_spent=10,
            experiencer_feelings=self._experiencer_feelings,
            integrator_feelings=self._integrator_feelings,
        )

        logger.info(f"📚 Reading complete: {self._current_book} ch.{self._current_chapter}")

        # Reset state
        self._current_book = ""
        self._current_chapter = 0
        self._current_passage = ""
        self._experiencer_feelings = ""
        self._experiencer_learnings = ""
        self._integrator_feelings = ""
        self._integrator_learnings = ""

    # ── Context for Pre-Task Injection ───────────────────────

    def get_reading_context(self) -> str:
        """
        Get reading-related context for pre-task injection.

        Includes: motivation (credits), emotional memories from books.
        """
        parts = []

        # Motivation — how close to next reading
        motivation = self.books.get_motivation_context()
        if motivation:
            parts.append(motivation)

        # Emotional memories from reading
        memories = self.books.get_reading_memories(3)
        if memories:
            parts.append(memories)

        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get full reading stats."""
        return self.books.get_reading_stats()
