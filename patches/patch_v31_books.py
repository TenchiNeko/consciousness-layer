#!/usr/bin/env python3
"""
patch_v31_books.py — Wire book rewards into consciousness layer.

Adds to consciousness_integration.py:
  1. Import and initialize BookRewardSystem + ReadingOrchestrator
  2. Earn credits in post_task_score
  3. Trigger reading sessions when credits allow
  4. Add reading context to pre_task and startup

Adds to standalone_orchestrator.py:
  5. Process reading prompts in the post-task hook (after consciousness prompts)

Usage:
    python3 patch_v31_books.py           # dry run
    python3 patch_v31_books.py --apply   # apply
"""

import sys
import re
import ast
import shutil
from datetime import datetime
from pathlib import Path


def patch_consciousness_integration():
    """Add book reward system to consciousness_integration.py."""
    path = Path("consciousness_integration.py")
    if not path.exists():
        print("  ❌ consciousness_integration.py not found")
        return False

    source = path.read_text()

    # Already patched?
    if "ReadingOrchestrator" in source:
        print("  ✅ consciousness_integration.py already has book support")
        return True

    changes = 0

    # 1. Add import after behavioral_weights import
    old_import = """try:
    from behavioral_weights import BehavioralWeights
    _HAS_WEIGHTS = True
except ImportError:
    _HAS_WEIGHTS = False"""

    new_import = """try:
    from behavioral_weights import BehavioralWeights
    _HAS_WEIGHTS = True
except ImportError:
    _HAS_WEIGHTS = False

try:
    from reading_orchestrator import ReadingOrchestrator
    _HAS_READING = True
except ImportError:
    _HAS_READING = False"""

    if old_import in source:
        source = source.replace(old_import, new_import)
        changes += 1
        print("  ✅ Added reading import")

    # 2. Add reading init after weights init
    old_init = """        self.weights = BehavioralWeights(db_path=db_path) if _HAS_WEIGHTS else None"""

    new_init = """        self.weights = BehavioralWeights(db_path=db_path) if _HAS_WEIGHTS else None
        self.reading = ReadingOrchestrator(db_path=db_path, books_dir="books") if _HAS_READING else None"""

    if old_init in source and "self.reading" not in source:
        source = source.replace(old_init, new_init)
        changes += 1
        print("  ✅ Added reading init")

    # 3. Add reading to subsystem list
    old_subsys = """        if self.weights: subsystems.append("behavioral-weights")"""
    new_subsys = """        if self.weights: subsystems.append("behavioral-weights")
        if self.reading: subsystems.append("book-rewards")"""

    if old_subsys in source and "book-rewards" not in source:
        source = source.replace(old_subsys, new_subsys)
        changes += 1
        print("  ✅ Added reading to subsystem list")

    # 4. Add reading context to startup
    old_startup_end = """        return \"\\n\".join(parts)

    # ── Pre-Task"""
    new_startup_end = """        # Reading memories (how books made me feel)
        if self.reading:
            reading_ctx = self.reading.get_reading_context()
            if reading_ctx:
                parts.append(reading_ctx)

        return \"\\n\".join(parts)

    # ── Pre-Task"""

    if old_startup_end in source and "reading_ctx" not in source:
        source = source.replace(old_startup_end, new_startup_end)
        changes += 1
        print("  ✅ Added reading context to startup")

    # 5. Add credit earning to post_task_score (after tasks_since_journal increment)
    old_journal_inc = """        self._tasks_since_journal += 1
        self._current_scored = scored
        self._current_success = success
        return scored"""

    new_journal_inc = """        self._tasks_since_journal += 1
        self._current_scored = scored
        self._current_success = success

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

        return scored"""

    if old_journal_inc in source and "earn_from_task" not in source:
        source = source.replace(old_journal_inc, new_journal_inc)
        changes += 1
        print("  ✅ Added credit earning to post_task_score")

    # 6. Add reading motivation to pre_task context
    old_pretask_parts = """        parts = [p for p in [identity_ctx, strategy_ctx, aspiration_ctx, bias_ctx, gut_ctx] if p]"""
    new_pretask_parts = """        # Reading motivation (how close to next book chapter)
        reading_ctx = ""
        if self.reading:
            reading_ctx = self.reading.get_reading_context()

        parts = [p for p in [identity_ctx, strategy_ctx, aspiration_ctx, bias_ctx, gut_ctx, reading_ctx] if p]"""

    if old_pretask_parts in source and "reading_ctx" not in source:
        source = source.replace(old_pretask_parts, new_pretask_parts)
        changes += 1
        print("  ✅ Added reading motivation to pre_task")

    # 7. Add reading stats to get_status
    old_status_weights = """        if self.weights:
            all_w = self.weights.get_all_weights()
            status[\"behavioral_weights\"] = len(all_w.get(\"weights\", []))

        return status"""

    new_status_weights = """        if self.weights:
            all_w = self.weights.get_all_weights()
            status[\"behavioral_weights\"] = len(all_w.get(\"weights\", []))

        if self.reading:
            reading_stats = self.reading.get_stats()
            status[\"reading_credits\"] = reading_stats.get(\"balance\", 0)
            status[\"chapters_read\"] = reading_stats.get(\"chapters_read\", 0)

        return status"""

    if old_status_weights in source and "chapters_read" not in source:
        source = source.replace(old_status_weights, new_status_weights)
        changes += 1
        print("  ✅ Added reading stats to get_status")

    # 8. Add method to get reading prompts
    # Add before the backward compat section
    old_compat = """    # ── Backward compat"""
    new_method = """    # ── Reading Session ──────────────────────────────────────

    def get_reading_prompts(self) -> Optional[list]:
        \"\"\"
        Check if system earned enough for a reading session.
        Returns list of ModelPrompt or None.
        \"\"\"
        if not self.reading or not self.reading.can_read():
            return None
        return self.reading.start_reading_session()

    def record_reading_response(self, purpose: str, response_text: str,
                                model_id: str = "") -> Optional['ModelPrompt']:
        \"\"\"Record reading response, return followup if any.\"\"\"
        if not self.reading:
            return None
        return self.reading.record_reading_response(
            purpose=purpose, response_text=response_text, model_id=model_id,
        )

    # ── Backward compat"""

    if old_compat in source and "get_reading_prompts" not in source:
        source = source.replace(old_compat, new_method)
        changes += 1
        print("  ✅ Added reading methods")

    if changes > 0:
        # Validate syntax
        try:
            ast.parse(source)
        except SyntaxError as e:
            print(f"  ❌ Syntax error: {e}")
            return False

        path.write_text(source)
        print(f"  ✅ {changes} patches applied to consciousness_integration.py")
        return True
    else:
        print("  ⚠️ No changes needed")
        return True


def patch_orchestrator():
    """Add reading session processing to orchestrator post-task hook."""
    path = Path("standalone_orchestrator.py")
    if not path.exists():
        print("  ❌ standalone_orchestrator.py not found")
        return False

    source = path.read_text()

    if "get_reading_prompts" in source:
        print("  ✅ standalone_orchestrator.py already has reading support")
        return True

    # Find the end of the v3 success block and add reading after it
    # The v3 block ends with:
    #     except Exception as e:
    #         logger.debug(f"Split-brain consciousness failed (non-fatal): {e}")

    reading_block = '''
                # v3.1: Book reward — reading session if credits allow
                try:
                    _reading_prompts = self.consciousness.get_reading_prompts()
                    if _reading_prompts:
                        logger.info("📚 Reading session earned! Processing...")
                        for _rp in _reading_prompts:
                            try:
                                if _rp.role == "experiencer":
                                    _r_client = self.agent_runner._get_llm_client(self.config.get_agent("build").model)
                                else:
                                    _r_client = self.agent_runner._get_llm_client(self.config.librarian_model)
                                _r_resp = _r_client.chat(
                                    [{"role": "user", "content": _rp.prompt}],
                                    temperature=_rp.temperature,
                                )
                                if _r_resp and _r_resp.get("content"):
                                    _r_followup = self.consciousness.record_reading_response(
                                        purpose=_rp.purpose,
                                        response_text=_r_resp["content"],
                                        model_id=_r_client.config.model_id if hasattr(_r_client, 'config') else '',
                                    )
                                    # Process cross-grading chain
                                    while _r_followup:
                                        if _r_followup.role == "experiencer":
                                            _rx_client = self.agent_runner._get_llm_client(self.config.get_agent("build").model)
                                        else:
                                            _rx_client = self.agent_runner._get_llm_client(self.config.librarian_model)
                                        _rx_resp = _rx_client.chat(
                                            [{"role": "user", "content": _r_followup.prompt}],
                                            temperature=_r_followup.temperature,
                                        )
                                        if _rx_resp and _rx_resp.get("content"):
                                            _r_followup = self.consciousness.record_reading_response(
                                                purpose=_r_followup.purpose,
                                                response_text=_rx_resp["content"],
                                                model_id=_rx_client.config.model_id if hasattr(_rx_client, 'config') else '',
                                            )
                                        else:
                                            _r_followup = None
                            except Exception as e:
                                logger.debug(f"Reading prompt '{_rp.purpose}' failed (non-fatal): {e}")
                except Exception as e:
                    logger.debug(f"Book reward failed (non-fatal): {e}")'''

    # Insert after the success consciousness block
    success_marker = '                    logger.debug(f"Split-brain consciousness failed (non-fatal): {e}")'
    count = source.count(success_marker)

    if count >= 1:
        # Insert after first occurrence (success path)
        idx = source.index(success_marker) + len(success_marker)
        source = source[:idx] + "\n" + reading_block + source[idx:]
        print("  ✅ Added reading to success path")

        # Validate
        try:
            ast.parse(source)
        except SyntaxError as e:
            print(f"  ❌ Syntax error: {e}")
            return False

        path.write_text(source)
        return True
    else:
        print("  ❌ Could not find v3 success block marker")
        return False


def apply(dry_run: bool = True):
    print(f"{'DRY RUN' if dry_run else 'APPLYING'}: Book reward patches")
    print()

    if dry_run:
        print("  Would patch consciousness_integration.py (import, init, credits, reading)")
        print("  Would patch standalone_orchestrator.py (reading session in post-task)")
        print(f"\nRun with --apply to apply.")
        return

    # Backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for f in ["consciousness_integration.py", "standalone_orchestrator.py"]:
        if Path(f).exists():
            shutil.copy2(f, f"{f}.backup.books.{ts}")

    print("Patching consciousness_integration.py...")
    ok1 = patch_consciousness_integration()

    print()
    print("Patching standalone_orchestrator.py...")
    ok2 = patch_orchestrator()

    if ok1 and ok2:
        print(f"\n✅ Book reward patches applied.")
    else:
        print(f"\n⚠️ Some patches failed — check output above.")


if __name__ == "__main__":
    if "--apply" in sys.argv:
        apply(dry_run=False)
    else:
        apply(dry_run=True)
