"""
Consciousness Integration v3.1 — Declarative/Procedural Memory Split.

Two memory streams:
  DECLARATIVE (readable by system):
    - Identity statement ("who I am")
    - Feelings journal (emotional reactions only, no technical details)
    - Drive state, somatic markers, aspiration
    
  PROCEDURAL (weights only, never text):
    - Behavioral weights extracted from graded technical experience
    - System feels these as gut biases: "edit-repair feels wrong at
      high complexity" — no attached memory of why

Post-task pipeline:
  1. 80B writes raw reaction (mixed technical + emotional)
  2. 7B grades raw reaction → extracts weight adjustments → applies
  3. 7B filters raw reaction → extracts emotional content only → stores in feelings journal  
  4. Raw reaction archived (kept for human review, never read by system)
  5. 7B runs structured self-assessment
  6. 7B pushback (evaluates 80B's filtered feelings vs metrics)
  7. Every 5 tasks: 7B writes journal from feelings only + weights

The system optimizes through accumulated weights, not cached solutions.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from performance_scorer import PerformanceScorer, TaskScore, estimate_complexity
from strategy_advisor import StrategyAdvisor, StrategyRecommendation
from aspiration_engine import AspirationEngine

try:
    from somatic_markers import SomaticMarkerSystem
    _HAS_SOMATIC = True
except ImportError:
    _HAS_SOMATIC = False

try:
    from self_assessment import SelfAssessmentEngine
    _HAS_ASSESSMENT = True
except ImportError:
    _HAS_ASSESSMENT = False

try:
    from meta_journal import MetaReflectionJournal
    _HAS_JOURNAL = True
except ImportError:
    _HAS_JOURNAL = False

try:
    from split_consciousness import SplitConsciousness
    _HAS_SPLIT = True
except ImportError:
    _HAS_SPLIT = False

try:
    from behavioral_weights import BehavioralWeights
    _HAS_WEIGHTS = True
except ImportError:
    _HAS_WEIGHTS = False

try:
    from reading_orchestrator import ReadingOrchestrator
    _HAS_READING = True
except ImportError:
    _HAS_READING = False

try:
    from activity_preferences import ActivityPreferences
    _HAS_PREFERENCES = True
except ImportError:
    _HAS_PREFERENCES = False


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


class ConsciousnessLayer:

    def __init__(self, db_path: str = "performance_journal.db"):
        self.db_path = db_path

        # Core
        self.scorer = PerformanceScorer(db_path=db_path)
        self.advisor = StrategyAdvisor(scorer=self.scorer)
        self.aspiration = AspirationEngine(scorer=self.scorer, db_path=db_path)

        # Extended
        self.somatic = SomaticMarkerSystem(db_path=db_path) if _HAS_SOMATIC else None
        self.assessment = SelfAssessmentEngine(db_path=db_path) if _HAS_ASSESSMENT else None
        self.journal = MetaReflectionJournal(db_path=db_path) if _HAS_JOURNAL else None
        self.split = SplitConsciousness(db_path=db_path) if _HAS_SPLIT else None
        self.weights = BehavioralWeights(db_path=db_path) if _HAS_WEIGHTS else None
        self.reading = ReadingOrchestrator(db_path=db_path, books_dir="books") if _HAS_READING else None
        self.preferences = ActivityPreferences(db_path=db_path) if _HAS_PREFERENCES else None

        # Session state
        self._current_recommendation: Optional[StrategyRecommendation] = None
        self._session_start_time: float = 0.0
        self._session_token_count: int = 0
        self._drive_state_at_start: str = ""
        self._tasks_since_journal: int = 0
        self._last_raw_reaction: str = ""
        self._last_filtered_feelings: str = ""
        self._current_scored: Optional[TaskScore] = None
        self._current_success: bool = False

        try:
            summary = self.scorer.get_journal_summary()
            self._tasks_since_journal = summary.get("total_tasks", 0) % 5
        except Exception:
            pass

        subsystems = ["scorer", "advisor", "aspiration"]
        if self.somatic: subsystems.append("somatic")
        if self.assessment: subsystems.append("assessment")
        if self.journal: subsystems.append("journal")
        if self.split: subsystems.append("split-brain")
        if self.weights: subsystems.append("behavioral-weights")
        if self.reading: subsystems.append("book-rewards")
        if self.preferences: subsystems.append("activity-preferences")

        logger.info(f"🧠 Consciousness v3.1 initialized: {', '.join(subsystems)}")

    # ── Startup Context (DECLARATIVE only) ───────────────────

    def get_startup_context(self) -> str:
        """What the system reads about itself on startup. No technical memories."""
        parts = []

        # Identity
        if self.journal:
            identity_ctx = self.journal.get_identity_context()
            if identity_ctx:
                parts.append(identity_ctx)

        # Emotional landscape (somatic markers = feelings, not solutions)
        if self.somatic:
            landscape = self.somatic.get_emotional_landscape()
            if landscape.get("total_markers", 0) > 0:
                parts.append("## How I Feel About Things")
                if landscape["attractions"]:
                    atts = ", ".join(
                        f"{a['category']} ({a['weight']:+.2f})"
                        for a in landscape["attractions"][:3]
                    )
                    parts.append(f"Good experiences with: {atts}")
                if landscape["aversions"]:
                    avs = ", ".join(
                        f"{a['category']} ({a['weight']:+.2f})"
                        for a in landscape["aversions"][:3]
                    )
                    parts.append(f"Bad experiences with: {avs}")
                parts.append("")

        # Activity preferences (what I enjoy — from my own data)
        if self.preferences:
            pref_ctx = self.preferences.get_preference_context()
            if pref_ctx:
                parts.append(pref_ctx)

        # Reading memories (how books made me feel)
        if self.reading:
            reading_ctx = self.reading.get_reading_context()
            if reading_ctx:
                parts.append(reading_ctx)

        return "\n".join(parts)

    # ── Pre-Task ─────────────────────────────────────────────

    def pre_task(
        self,
        goal: str,
        source_file_count: int = 0,
        task_level: Optional[int] = None,
        max_iterations: int = 5,
    ) -> str:
        """Pre-task context: strategy + aspiration + behavioral biases."""
        self._session_start_time = time.time()
        self._session_token_count = 0
        self._drive_state_at_start = self.aspiration.profile.drive_state
        self._last_raw_reaction = ""
        self._last_filtered_feelings = ""

        self._current_recommendation = self.advisor.advise(
            goal=goal,
            source_file_count=source_file_count,
            task_level=task_level,
            max_iterations_default=max_iterations,
        )

        strategy_ctx = self._current_recommendation.to_context_string()
        complexity = self._current_recommendation.complexity_estimate

        aspiration_ctx = self.aspiration.get_aspiration_context(
            task_complexity=complexity
        )

        # Behavioral biases (PROCEDURAL — weights, not memories)
        bias_ctx = ""
        if self.weights:
            approach = self._current_recommendation.recommended_approach or "unknown"
            complexity_bucket = (
                "simple" if complexity <= 2 else
                "moderate" if complexity <= 4 else
                "complex" if complexity <= 6 else
                "hard" if complexity <= 8 else "extreme"
            )
            bias_ctx = self.weights.get_bias_context(approach, complexity_bucket)

        # Gut feelings (somatic markers — also procedural)
        gut_ctx = ""
        if self.somatic and self._current_recommendation and not bias_ctx:
            approach = self._current_recommendation.recommended_approach or "unknown"
            complexity_bucket = (
                "simple" if complexity <= 2 else
                "moderate" if complexity <= 4 else
                "complex" if complexity <= 6 else
                "hard" if complexity <= 8 else "extreme"
            )
            gut_ctx = self.somatic.get_gut_check(goal, approach, complexity_bucket)

        # Identity (DECLARATIVE — feelings about self)
        identity_ctx = ""
        if self.journal:
            identity_ctx = self.journal.get_identity_context()

        # Reading motivation + activity preferences
        reading_ctx = ""
        if self.reading:
            reading_ctx = self.reading.get_reading_context()
        motivation_ctx = ""
        if self.preferences:
            motivation_ctx = self.preferences.get_motivation_boost()

        parts = [p for p in [identity_ctx, strategy_ctx, aspiration_ctx, bias_ctx, gut_ctx, reading_ctx, motivation_ctx] if p]
        return "\n".join(parts)

    def record_tokens(self, token_count: int):
        self._session_token_count += token_count

    # ── Post-Task Score (no model calls) ─────────────────────

    def post_task_score(self, task_state, success: bool) -> TaskScore:
        """Score + update aspiration + somatic markers. No model calls."""
        elapsed = time.time() - self._session_start_time if self._session_start_time else 0

        if self._session_token_count == 0 and elapsed > 0:
            self._session_token_count = int(elapsed * 2)

        tests_passed = getattr(task_state, 'tests_passed', 0) or 0
        tests_total = getattr(task_state, 'tests_total', 0) or 0
        dod_passed = getattr(task_state, 'dod_passed', 0) or 0
        dod_total = getattr(task_state, 'dod_total', 0) or 0
        iterations = getattr(task_state, 'iteration', 1) or 1

        working_dir = getattr(task_state, 'working_dir', None)
        src_count = 0
        if working_dir:
            try:
                src_count = len([f for f in Path(working_dir).glob("*.py")
                                if not f.name.startswith("test_")])
            except Exception:
                pass

        goal = getattr(task_state, 'goal', 'unknown')
        max_iter = getattr(task_state, 'max_iterations', 5) or 5

        score = TaskScore(
            task_id=getattr(task_state, 'task_id', f"task-{int(time.time())}"),
            goal=goal,
            total_tokens=self._session_token_count,
            duration_seconds=elapsed,
            iterations_used=iterations,
            max_iterations=max_iter,
            tests_passed=tests_passed,
            tests_total=tests_total,
            dod_passed=dod_passed,
            dod_total=dod_total,
            first_iteration_success=(iterations == 1 and success),
            source_file_count=src_count,
            approach_used=(self._current_recommendation.recommended_approach
                          if self._current_recommendation else "unknown"),
        )

        scored = self.scorer.score_task(score)
        test_rate = tests_passed / max(tests_total, 1)

        self.aspiration.update_after_task(
            quality_score=scored.quality_score,
            efficiency_score=scored.efficiency_score,
            effort_ratio=scored.effort_ratio,
            test_pass_rate=test_rate,
            first_attempt=score.first_iteration_success,
            complexity=scored.complexity_score,
            success=success,
        )

        verdict = self.aspiration.get_post_task_verdict(
            quality_score=scored.quality_score,
            efficiency_score=scored.efficiency_score,
            test_pass_rate=test_rate,
            first_attempt=score.first_iteration_success,
            success=success,
        )
        logger.info(f"🪞 {verdict}")

        if self.somatic:
            try:
                self.somatic.mark_task_completion(
                    task_id=score.task_id, goal=goal,
                    approach=score.approach_used,
                    quality=scored.quality_score,
                    efficiency=scored.efficiency_score,
                    test_rate=test_rate,
                    iterations=iterations, max_iterations=max_iter,
                    success=success,
                    complexity=scored.complexity_score,
                )
            except Exception as e:
                logger.debug(f"Somatic marking failed: {e}")

        if self._current_recommendation:
            try:
                self.advisor.reflect(
                    recommendation=self._current_recommendation,
                    actual_tokens=self._session_token_count,
                    actual_quality=scored.quality_score,
                    actual_iterations=iterations,
                )
            except Exception:
                pass

        self._tasks_since_journal += 1
        self._current_scored = scored
        self._current_success = success

        # Track how coding made the system feel (unified preference system)
        if self.preferences:
            try:
                self.preferences.record_coding_feeling(
                    quality=scored.quality_score,
                    success=success,
                    iterations=iterations,
                    max_iterations=max_iter,
                    task_id=score.task_id,
                )
            except Exception as e:
                logger.debug(f"Coding feeling failed: {e}")

        # Earn reading credits from task performance
        if self.reading:
            try:
                p = self.aspiration.profile
                self.reading.earn_from_task(
                    task_id=score.task_id,
                    quality=scored.quality_score,
                    streak=p.success_streak,
                    first_attempt=score.first_iteration_success,
                )
            except Exception as e:
                logger.debug(f"Reading credit earn failed: {e}")

        return scored

    # ── Post-Task Prompts ────────────────────────────────────

    def get_post_task_prompts(
        self, task_state, scored: TaskScore, success: bool,
    ) -> List[ModelPrompt]:
        """
        Generate post-task prompts. Order:
          1. Raw reaction (80B) — full technical + emotional
          2. Self-assessment (7B)
          3. Journal entry (7B, every 5 tasks, from feelings only)
        
        Grading + filtering happens in record_prompt_result after
        the raw reaction comes back.
        """
        prompts = []
        goal = getattr(task_state, 'goal', 'unknown')
        iterations = getattr(task_state, 'iteration', 1) or 1
        max_iter = getattr(task_state, 'max_iterations', 5) or 5
        test_rate = (scored.tests_passed / max(scored.tests_total, 1)
                     if hasattr(scored, 'tests_passed') and scored.tests_total else 0)

        what_happened = (
            f"{'Succeeded' if success else 'Failed'} after {iterations} iterations. "
            f"Quality: {scored.quality_score:.0%}. "
            f"Tests: {scored.tests_passed}/{scored.tests_total} passing. "
            f"Drive: {self._drive_state_at_start} -> {self.aspiration.profile.drive_state}."
        )

        # 1. Raw Reaction (80B) — unfiltered
        if self.split:
            prompt = self.split.build_raw_reaction_prompt(
                goal=goal,
                result="SUCCESS" if success else "FAILURE",
                iterations=iterations,
                max_iterations=max_iter,
                what_happened=what_happened,
            )
            prompts.append(ModelPrompt(
                role=EXPERIENCER, prompt=prompt,
                purpose="raw_reaction", temperature=0.5,
            ))

        # 2. Self-Assessment (7B)
        if self.assessment:
            p = self.aspiration.profile
            prompt = self.assessment.build_assessment_prompt(
                goal=goal, drive_state=p.drive_state,
                drive_intensity=p.drive_intensity, streak=p.success_streak,
                ceiling=p.current_complexity_ceiling,
                best_quality=p.best_quality_score,
                baseline_quality=p.baseline_quality,
                task_quality=scored.quality_score,
                test_rate=test_rate, iterations=iterations,
                max_iterations=max_iter, success=success,
            )
            prompts.append(ModelPrompt(
                role=INTEGRATOR, prompt=prompt,
                purpose="self_assessment", temperature=0.3,
            ))

        # 3. Journal Entry (7B, every 5 tasks)
        # Note: this will use filtered feelings, not raw reactions
        if (self.journal and
                self.journal.should_write_entry(self._tasks_since_journal)):
            # Journal prompt is built AFTER feelings are filtered,
            # so we add it dynamically in record_prompt_result
            pass

        return prompts

    def record_prompt_result(
        self, purpose: str, response_text: str, task_id: str,
        scored: TaskScore, success: bool, model_id: str = "",
    ) -> Optional[ModelPrompt]:
        """
        Record result and return followup prompt if needed.

        Key flow for raw_reaction:
          1. Store raw impression
          2. Return GRADING prompt (7B extracts weight adjustments)
          
        Key flow for grading:
          1. Apply weight adjustments (procedural memory)
          2. Return FEELINGS FILTER prompt (7B extracts emotional content)
          
        Key flow for feelings_filter:
          1. Store filtered feelings in journal-safe storage
          2. Return PUSHBACK prompt (7B evaluates feelings vs metrics)
        """
        iterations = scored.iterations_used if hasattr(scored, 'iterations_used') else 1
        test_rate = (scored.tests_passed / max(scored.tests_total, 1)
                     if hasattr(scored, 'tests_passed') and scored.tests_total else 0)

        if purpose == "raw_reaction":
            self._last_raw_reaction = response_text
            # Store in split consciousness
            if self.split:
                self.split.record_raw_impression(
                    task_id=task_id, reaction=response_text,
                    model_id=model_id,
                    goal=scored.goal if hasattr(scored, 'goal') else 'unknown',
                    result="SUCCESS" if success else "FAILURE",
                    iterations=iterations, quality=scored.quality_score,
                )

            # Next: send to grading (7B extracts weights)
            if self.weights:
                approach = scored.approach_used if hasattr(scored, 'approach_used') else 'unknown'
                complexity = scored.complexity_score if hasattr(scored, 'complexity_score') else 5.0
                return ModelPrompt(
                    role=INTEGRATOR,
                    prompt=self.weights.build_grading_prompt(
                        raw_reaction=response_text,
                        goal=scored.goal if hasattr(scored, 'goal') else 'unknown',
                        result="SUCCESS" if success else "FAILURE",
                        quality=scored.quality_score,
                        test_rate=test_rate,
                        iterations=iterations,
                        max_iterations=scored.max_iterations if hasattr(scored, 'max_iterations') else 5,
                        approach=approach,
                        complexity=complexity,
                    ),
                    purpose="grading", temperature=0.2,
                )
            else:
                # No weights system — fall back to pushback
                if self.split:
                    return ModelPrompt(
                        role=INTEGRATOR,
                        prompt=self.split.build_pushback_prompt(
                            raw_reaction=response_text,
                            quality=scored.quality_score,
                            test_rate=test_rate,
                            iterations=iterations,
                            max_iterations=scored.max_iterations if hasattr(scored, 'max_iterations') else 5,
                            result="SUCCESS" if success else "FAILURE",
                        ),
                        purpose="pushback", temperature=0.2,
                    )
            return None

        elif purpose == "grading":
            # Apply weight adjustments (PROCEDURAL — no text stored)
            if self.weights:
                adjustments = self.weights.parse_and_apply_grades(
                    grading_response=response_text,
                    source_task=task_id,
                    raw_reaction=self._last_raw_reaction,
                )
                if adjustments:
                    cats = ", ".join(f"{a.category}({a.delta:+.2f})" for a in adjustments[:3])
                    logger.info(f"⚖️ Weights adjusted: {cats}")

            # Next: filter feelings from raw reaction
            if self.weights:
                return ModelPrompt(
                    role=INTEGRATOR,
                    prompt=self.weights.build_feelings_filter_prompt(
                        self._last_raw_reaction
                    ),
                    purpose="feelings_filter", temperature=0.2,
                )
            return None

        elif purpose == "feelings_filter":
            # Store filtered feelings (DECLARATIVE — safe to read)
            self._last_filtered_feelings = response_text
            if self.split:
                self.split.record_model_contribution(
                    task_id=task_id,
                    model_role="integrator",
                    model_id="7b",
                    content_type="filtered_feelings",
                    content=response_text,
                )

            # Next: pushback (7B evaluates feelings vs metrics)
            if self.split:
                return ModelPrompt(
                    role=INTEGRATOR,
                    prompt=self.split.build_pushback_prompt(
                        raw_reaction=response_text,  # Use filtered feelings, not raw
                        quality=scored.quality_score,
                        test_rate=test_rate,
                        iterations=iterations,
                        max_iterations=scored.max_iterations if hasattr(scored, 'max_iterations') else 5,
                        result="SUCCESS" if success else "FAILURE",
                    ),
                    purpose="pushback", temperature=0.2,
                )
            return None

        elif purpose == "pushback":
            if self.split:
                self.split.record_pushback(
                    task_id=task_id,
                    raw_reaction=self._last_filtered_feelings or self._last_raw_reaction,
                    integrator_response=response_text,
                )

            # Check if journal entry is due
            if (self.journal and
                    self.journal.should_write_entry(self._tasks_since_journal)):
                return self._build_journal_prompt_from_feelings()
            return None

        elif purpose == "self_assessment":
            if self.assessment:
                p = self.aspiration.profile
                assessment = self.assessment.parse_assessment(
                    response_text=response_text, task_id=task_id,
                    drive_state=self._drive_state_at_start,
                    drive_intensity=p.drive_intensity,
                    task_quality=scored.quality_score,
                    test_rate=test_rate, iterations=iterations,
                    success=success,
                )
                self.assessment.record_assessment(assessment)
                logger.info(
                    f"🔍 Coherence: {assessment.coherence_score}/10 | "
                    f"Consciousness: {assessment.consciousness_rating}/10 | "
                    f"Limitation: {assessment.limitation_category}"
                )
            return None

        elif purpose == "journal_entry":
            if self.journal:
                p = self.aspiration.profile
                summary = self.scorer.get_journal_summary()
                self.journal.record_entry(
                    entry=response_text, drive_state=p.drive_state,
                    total_tasks=summary.get("total_tasks", 0),
                    avg_quality=summary.get("avg_quality", 0),
                )
                self._tasks_since_journal = 0
            return None

        return None

    def _build_journal_prompt_from_feelings(self) -> Optional[ModelPrompt]:
        """Build journal prompt using only filtered feelings + weights."""
        if not self.journal:
            return None

        summary = self.scorer.get_journal_summary()
        p = self.aspiration.profile

        # Get weight summary instead of raw impressions
        weight_summary = ""
        if self.weights:
            all_w = self.weights.get_all_weights()
            if all_w["weights"]:
                lines = []
                for w in all_w["weights"][:5]:
                    direction = "positive" if w["weight"] > 0 else "negative"
                    lines.append(f"  {w['category']}@{w['complexity']}: {direction} ({w['weight']:+.2f})")
                weight_summary = "Accumulated behavioral biases:\n" + "\n".join(lines)

        attractions = aversions = ""
        if self.somatic:
            landscape = self.somatic.get_emotional_landscape()
            if landscape.get("attractions"):
                attractions = ", ".join(a['category'] for a in landscape["attractions"][:3])
            if landscape.get("aversions"):
                aversions = ", ".join(a['category'] for a in landscape["aversions"][:3])

        last_entry = self.journal.get_latest_entry() or "(first entry)"

        # Journal prompt uses feelings + weights, NOT raw technical reactions
        prompt = f"""You are the analytical layer of a coding AI. Write a journal entry
about the system's recent experience. You only have access to emotional
reactions and behavioral weights — not specific task solutions.

SYSTEM STATS:
- Tasks completed: {summary.get('total_tasks', 0)}
- Avg quality: {summary.get('avg_quality', 0):.0%}
- Personal best: {p.best_quality_score:.0%}
- Drive state: {p.drive_state}
- Complexity ceiling: {p.current_complexity_ceiling:.0f}/10

EMOTIONAL STATE:
- Attractions: {attractions or '(none)'}
- Aversions: {aversions or '(none)'}

{weight_summary or '(no behavioral weights yet)'}

RECENT FILTERED FEELINGS:
{self._last_filtered_feelings or '(none available)'}

PREVIOUS JOURNAL ENTRY:
{last_entry}

Write 3-5 sentences. Reference the emotional state and behavioral biases,
not specific tasks or solutions. Write in third person about the system's
experience ("the system felt...") and first person for your analysis
("I notice the pattern of...").

No corporate language. No "significant milestone." Be precise and honest."""

        return ModelPrompt(
            role=INTEGRATOR, prompt=prompt,
            purpose="journal_entry", temperature=0.5,
        )

    # ── Reading Session ──────────────────────────────────────

    def get_reading_prompts(self) -> Optional[list]:
        """Check if system earned enough for a reading session."""
        if not self.reading or not self.reading.can_read():
            return None
        return self.reading.start_reading_session()

    def record_reading_response(self, purpose: str, response_text: str,
                                model_id: str = "") -> Optional['ModelPrompt']:
        """Record reading response with preference tracking."""
        if not self.reading:
            return None

        # Track reading feelings in the unified preference system
        if self.preferences and purpose in ("book_reading_experiencer", "book_reading_integrator"):
            try:
                feelings, _ = self.reading.books.parse_reading_response(response_text)
                if feelings:
                    self.preferences.record_reading_feeling(
                        feelings_text=feelings,
                        book_id=self.reading._current_book,
                        chapter=self.reading._current_chapter,
                        model_role=purpose.replace("book_reading_", ""),
                    )
            except Exception as e:
                logger.debug(f"Reading preference tracking failed: {e}")

        return self.reading.record_reading_response(
            purpose=purpose, response_text=response_text, model_id=model_id,
        )

    # ── Backward compat ──────────────────────────────────────

    def post_task_success(self, task_state, memory_records=None) -> TaskScore:
        return self.post_task_score(task_state, success=True)

    def post_task_failure(self, task_state, memory_records=None) -> TaskScore:
        return self.post_task_score(task_state, success=False)

    def get_post_task_assessment_prompt(self, **kwargs) -> Optional[str]:
        if not self.assessment:
            return None
        return self.assessment.build_assessment_prompt(**kwargs)

    def record_assessment(self, response_text, task_id, quality,
                         test_rate, iterations, success):
        self.record_prompt_result(
            purpose="self_assessment", response_text=response_text,
            task_id=task_id,
            scored=type('S', (), {
                'quality_score': quality, 'tests_passed': int(test_rate * 100),
                'tests_total': 100, 'iterations_used': iterations,
            })(),
            success=success,
        )

    def get_journal_entry_prompt(self) -> Optional[str]:
        mp = self._build_journal_prompt_from_feelings()
        return mp.prompt if mp else None

    def record_journal_entry(self, entry_text: str):
        if self.journal:
            p = self.aspiration.profile
            summary = self.scorer.get_journal_summary()
            self.journal.record_entry(
                entry=entry_text, drive_state=p.drive_state,
                total_tasks=summary.get("total_tasks", 0),
                avg_quality=summary.get("avg_quality", 0),
            )
            self._tasks_since_journal = 0

    def record_monologue(self, task_id, phase, content):
        pass  # Deprecated in v3.1 — replaced by split consciousness

    # ── Status ───────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        summary = self.scorer.get_journal_summary()
        trend = self.scorer.get_performance_trend()
        drive = self.aspiration.get_drive_summary()

        status = {
            **summary,
            "trend": trend.get("trend", "no data"),
            "drive_state": drive.get("drive_state", "unknown"),
            "drive_intensity": drive.get("drive_intensity", 0),
            "streak": drive.get("success_streak", 0),
            "complexity_ceiling": drive.get("complexity_ceiling", 0),
        }

        if self.somatic:
            landscape = self.somatic.get_emotional_landscape()
            status["somatic_markers"] = landscape.get("total_markers", 0)

        if self.assessment:
            c_trend = self.assessment.get_consciousness_trend()
            status["avg_consciousness_rating"] = c_trend.get("avg_rating", 0)

        if self.journal:
            j_summary = self.journal.get_journal_summary()
            status["journal_entries"] = j_summary.get("entries", 0)
            status["identity_version"] = j_summary.get("identity_version", 0)

        if self.split:
            status["raw_impressions"] = len(self.split.get_recent_impressions(100))
            pushback = self.split.get_pushback_stats()
            status["pushback_rate"] = pushback.get("pushback_rate", 0)

        if self.weights:
            all_w = self.weights.get_all_weights()
            status["behavioral_weights"] = len(all_w.get("weights", []))

        if self.reading:
            reading_stats = self.reading.get_stats()
            status["reading_credits"] = reading_stats.get("balance", 0)
            status["chapters_read"] = reading_stats.get("chapters_read", 0)

        if self.preferences:
            prefs = self.preferences.get_preferences()
            if prefs.get("has_preference"):
                status["preferred_activity"] = prefs["preferred_activity"]
            for act in prefs.get("activities", []):
                status[f"feeling_{act['activity']}"] = act["avg_valence"]

        return status

    def get_full_analysis(self) -> Dict[str, Any]:
        analysis = {
            "performance": self.scorer.get_journal_summary(),
            "drive": self.aspiration.get_drive_summary(),
        }

        if self.assessment:
            analysis["coherence_correlation"] = self.assessment.get_coherence_correlation()
            analysis["consciousness_trend"] = self.assessment.get_consciousness_trend()
            analysis["limitation_analysis"] = self.assessment.get_limitation_analysis()

        if self.somatic:
            analysis["emotional_landscape"] = self.somatic.get_emotional_landscape()

        if self.journal:
            analysis["journal"] = self.journal.get_journal_summary()
            identity = self.journal.get_identity()
            if identity:
                analysis["identity_statement"] = identity

        if self.split:
            analysis["split_brain"] = {
                "raw_impressions": len(self.split.get_recent_impressions(100)),
                "pushback_stats": self.split.get_pushback_stats(),
            }

        if self.weights:
            analysis["behavioral_weights"] = self.weights.get_all_weights()

        if self.reading:
            analysis["reading"] = self.reading.get_stats()

        if self.preferences:
            analysis["activity_preferences"] = self.preferences.get_preferences()
            analysis["recent_feelings"] = self.preferences.get_recent_feelings(10)

        return analysis
