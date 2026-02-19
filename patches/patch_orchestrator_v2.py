#!/usr/bin/env python3
"""
patch_orchestrator_v2.py — Add consciousness v2 hooks.

Prerequisite: v1 patches must already be applied (consciousness_integration import exists).
This script adds:
  - Post-task self-assessment calls
  - Inner monologue generation
  - Somatic marker recording
  - Journal entry triggers

Safety: backup + dry-run + syntax validation.

Usage:
    python3 patch_orchestrator_v2.py           # dry run
    python3 patch_orchestrator_v2.py --apply   # apply
    python3 patch_orchestrator_v2.py --rollback
"""

import sys
import shutil
import ast
from datetime import datetime
from pathlib import Path

ORCH_FILE = "standalone_orchestrator.py"

PATCHES = [
    # ─── PATCH V2-1: Startup identity context ───
    (
        "Add identity/emotional context at startup",
        # Anchor: the v1 consciousness status block
        """        # v2.0: Consciousness status
        if self.consciousness:
            try:
                _c_status = self.consciousness.get_status()
                logger.info(f"Consciousness: ✅ {_c_status.get('total_tasks', 0)} tasks scored, "
                           f"drive={_c_status.get('drive_state', 'unknown')}, "
                           f"streak={_c_status.get('streak', 0)}")
            except Exception:
                logger.info("Consciousness: ✅ ready")""",
        # Replace with same + startup context
        """        # v2.0: Consciousness status
        if self.consciousness:
            try:
                _c_status = self.consciousness.get_status()
                logger.info(f"Consciousness: ✅ {_c_status.get('total_tasks', 0)} tasks scored, "
                           f"drive={_c_status.get('drive_state', 'unknown')}, "
                           f"streak={_c_status.get('streak', 0)}")
                # v2.1: Load identity + emotional landscape at startup
                if _c_status.get('journal_entries', 0) > 0:
                    logger.info(f"  📓 Journal: {_c_status.get('journal_entries', 0)} entries, "
                               f"identity v{_c_status.get('identity_version', 0)}")
                if _c_status.get('somatic_markers', 0) > 0:
                    logger.info(f"  💓 Somatic markers: {_c_status.get('somatic_markers', 0)} "
                               f"(emotional range: {_c_status.get('emotional_range', 0):.2f})")
                if _c_status.get('avg_consciousness_rating', 0) > 0:
                    logger.info(f"  🪞 Avg consciousness self-rating: "
                               f"{_c_status.get('avg_consciousness_rating', 0):.1f}/10")
            except Exception:
                logger.info("Consciousness: ✅ ready")""",
        "replace",
    ),

    # ─── PATCH V2-2: Post-task assessment (success path) ───
    (
        "Add self-assessment after successful task scoring",
        # Anchor: the v1 success scoring block's logger line
        """                logger.info(f"📊 Task scored: quality={scored.quality_score:.2f} "
                           f"efficiency={scored.efficiency_score:.2f} "
                           f"reinforcement={scored.reinforcement_signal:+.2f}")""",
        # Replace with: same + assessment + journal triggers
        """                logger.info(f"📊 Task scored: quality={scored.quality_score:.2f} "
                           f"efficiency={scored.efficiency_score:.2f} "
                           f"reinforcement={scored.reinforcement_signal:+.2f}")
                # v2.1: Self-assessment — the system introspects on its own performance
                try:
                    _test_rate = scored.tests_passed / max(scored.tests_total, 1) if hasattr(scored, 'tests_passed') else 0
                    _assess_prompt = self.consciousness.get_post_task_assessment_prompt(
                        goal=task_state.goal,
                        quality=scored.quality_score,
                        test_rate=_test_rate,
                        iterations=getattr(task_state, 'current_iteration', 1),
                        max_iterations=self.max_iterations,
                        success=True,
                    )
                    if _assess_prompt:
                        _assess_resp = self.agent_runner.coder.chat(
                            [{"role": "user", "content": _assess_prompt}],
                            temperature=0.4,
                        )
                        if _assess_resp and _assess_resp.get("content"):
                            self.consciousness.record_assessment(
                                response_text=_assess_resp["content"],
                                task_id=getattr(task_state, 'task_id', 'unknown'),
                                quality=scored.quality_score,
                                test_rate=_test_rate,
                                iterations=getattr(task_state, 'current_iteration', 1),
                                success=True,
                            )
                except Exception as e:
                    logger.debug(f"Self-assessment failed (non-fatal): {e}")
                # v2.1: Journal entry (every 5 tasks)
                try:
                    _j_prompt = self.consciousness.get_journal_entry_prompt()
                    if _j_prompt:
                        _j_resp = self.agent_runner.coder.chat(
                            [{"role": "user", "content": _j_prompt}],
                            temperature=0.6,
                        )
                        if _j_resp and _j_resp.get("content"):
                            self.consciousness.record_journal_entry(_j_resp["content"])
                except Exception as e:
                    logger.debug(f"Journal entry failed (non-fatal): {e}")""",
        "replace",
    ),

    # ─── PATCH V2-3: Post-task assessment (failure path) ───
    (
        "Add self-assessment after failed task scoring",
        """                logger.info(f"📊 Failed task scored: quality={scored.quality_score:.2f} "
                           f"reinforcement={scored.reinforcement_signal:+.2f}")""",
        """                logger.info(f"📊 Failed task scored: quality={scored.quality_score:.2f} "
                           f"reinforcement={scored.reinforcement_signal:+.2f}")
                # v2.1: Self-assessment on failure — failures teach the most
                try:
                    _test_rate = scored.tests_passed / max(scored.tests_total, 1) if hasattr(scored, 'tests_passed') else 0
                    _assess_prompt = self.consciousness.get_post_task_assessment_prompt(
                        goal=task_state.goal,
                        quality=scored.quality_score,
                        test_rate=_test_rate,
                        iterations=getattr(task_state, 'current_iteration', 1),
                        max_iterations=self.max_iterations,
                        success=False,
                    )
                    if _assess_prompt:
                        _assess_resp = self.agent_runner.coder.chat(
                            [{"role": "user", "content": _assess_prompt}],
                            temperature=0.4,
                        )
                        if _assess_resp and _assess_resp.get("content"):
                            self.consciousness.record_assessment(
                                response_text=_assess_resp["content"],
                                task_id=getattr(task_state, 'task_id', 'unknown'),
                                quality=scored.quality_score,
                                test_rate=_test_rate,
                                iterations=getattr(task_state, 'current_iteration', 1),
                                success=False,
                            )
                except Exception as e:
                    logger.debug(f"Self-assessment on failure failed (non-fatal): {e}")""",
        "replace",
    ),
]


def validate_python(filepath: Path) -> bool:
    try:
        ast.parse(filepath.read_text())
        return True
    except SyntaxError as e:
        print(f"  ❌ SYNTAX ERROR: {e}")
        return False


def apply_patches(dry_run: bool = True):
    orch_path = Path(ORCH_FILE)
    if not orch_path.exists():
        print(f"❌ {ORCH_FILE} not found. Run from orchestrator directory.")
        sys.exit(1)

    source = orch_path.read_text()

    # Check prerequisite: v1 patches must exist
    if "_HAS_CONSCIOUSNESS" not in source:
        print("❌ v1 consciousness patches not found.")
        print("   Run patch_orchestrator.py --apply first.")
        sys.exit(1)

    print(f"{'DRY RUN' if dry_run else 'APPLYING'}: {len(PATCHES)} v2 patches")
    print()

    # Verify anchors
    all_found = True
    for i, (desc, anchor, replacement, position) in enumerate(PATCHES):
        if anchor in source:
            print(f"  ✅ V2-{i+1}: {desc}")
        else:
            if "v2.1:" in source and i > 0:
                print(f"  ⚠️  V2-{i+1}: {desc} — appears already applied")
            else:
                print(f"  ❌ V2-{i+1}: {desc} — ANCHOR NOT FOUND")
                all_found = False

    if not all_found:
        if "v2.1:" in source:
            print("\n⚠️  v2 patches appear already applied.")
            sys.exit(0)
        print("\n❌ Cannot safely patch. Anchor mismatch.")
        sys.exit(1)

    print()

    for i, (desc, anchor, replacement, position) in enumerate(PATCHES):
        if source.count(anchor) != 1:
            print(f"  ⚠️  V2-{i+1}: skipping (anchor count ≠ 1)")
            continue
        source = source.replace(anchor, replacement, 1)
        added = replacement.count("\n") - anchor.count("\n")
        print(f"  {'WOULD' if dry_run else ''} APPLIED V2-{i+1}: {desc} (+{added} lines)")

    if dry_run:
        print(f"\nDRY RUN COMPLETE. Run with --apply to apply.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = Path(f"{ORCH_FILE}.backup.v2.{timestamp}")
    shutil.copy2(orch_path, backup)
    print(f"\n  📦 Backup: {backup}")

    orch_path.write_text(source)
    if validate_python(orch_path):
        print(f"  ✅ Syntax valid")
        print(f"\n✅ All v2 patches applied. Rollback: cp {backup} {ORCH_FILE}")
    else:
        shutil.copy2(backup, orch_path)
        print(f"  🔄 Rolled back due to syntax error")
        sys.exit(1)


def rollback():
    backups = sorted(Path(".").glob(f"{ORCH_FILE}.backup.v2.*"))
    if not backups:
        print("❌ No v2 backup found.")
        sys.exit(1)
    latest = backups[-1]
    shutil.copy2(latest, Path(ORCH_FILE))
    print(f"✅ Restored from {latest}")


if __name__ == "__main__":
    if "--rollback" in sys.argv:
        rollback()
    elif "--apply" in sys.argv:
        apply_patches(dry_run=False)
    else:
        apply_patches(dry_run=True)
