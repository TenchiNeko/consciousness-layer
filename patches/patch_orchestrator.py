#!/usr/bin/env python3
"""
patch_orchestrator.py — Safe patcher for consciousness layer integration.

Applies surgical edits to standalone_orchestrator.py to wire in the
performance scorer, strategy advisor, and aspiration engine.

Safety features:
  - Creates timestamped backups before any changes
  - Verifies exact string matches before replacing
  - Dry-run mode (default) — shows changes without applying
  - Rollback support via backup files
  - Validates Python syntax after patching

Usage:
    # Preview changes (dry run)
    python3 patch_orchestrator.py

    # Apply changes
    python3 patch_orchestrator.py --apply

    # Rollback to backup
    python3 patch_orchestrator.py --rollback
"""

import sys
import shutil
import ast
from datetime import datetime
from pathlib import Path

ORCH_FILE = "standalone_orchestrator.py"

# ── Patch Definitions ────────────────────────────────────────────
# Each patch: (description, anchor_string, replacement_string, position)
# position: "after" = insert after anchor, "replace" = replace anchor

PATCHES = [
    # ─── PATCH 1: Import ───
    (
        "Add consciousness layer import",
        # Anchor: the existing librarian import line
        "from librarian_store import init_librarian_tables, get_librarian_stats, get_session_context, add_ast_chunks",
        # Replace with: same line + new import
        """from librarian_store import init_librarian_tables, get_librarian_stats, get_session_context, add_ast_chunks

# v2.0: Consciousness layer — performance scoring + aspiration drive
try:
    from consciousness_integration import ConsciousnessLayer
    _HAS_CONSCIOUSNESS = True
except ImportError:
    _HAS_CONSCIOUSNESS = False""",
        "replace",
    ),

    # ─── PATCH 2: Init ───
    (
        "Initialize consciousness layer in __init__",
        # Anchor: trace collector init (unique line)
        "        self.trace_collector = TraceCollector(working_dir)",
        # Replace with: same line + consciousness init
        """        self.trace_collector = TraceCollector(working_dir)

        # v2.0: Consciousness layer — scoring, strategy, aspiration
        self.consciousness = None
        if _HAS_CONSCIOUSNESS:
            try:
                _journal_path = str(Path(__file__).parent / 'performance_journal.db')
                self.consciousness = ConsciousnessLayer(db_path=_journal_path)
                logger.info("Consciousness: ✅ initialized (scoring + aspiration enabled)")
            except Exception as e:
                logger.warning(f"Consciousness: ⚠️ init failed: {e} (running without)")""",
        "replace",
    ),

    # ─── PATCH 3: Startup logging ───
    (
        "Add consciousness status to startup logging",
        # Anchor: librarian status logging block (unique multi-line)
        """        # v0.9.0: Librarian status
        if self.librarian:
            try:
                lib_stats = get_librarian_stats(db_path=self.librarian_db_path or "")
                logger.info(f"Librarian: ✅ {lib_stats.get('journal_entries', 0)} journal entries, "
                           f"{lib_stats.get('snippets', 0)} snippets")
            except Exception:
                logger.info("Librarian: ✅ ready (no stats available)")""",
        # Replace with: same block + consciousness status
        """        # v0.9.0: Librarian status
        if self.librarian:
            try:
                lib_stats = get_librarian_stats(db_path=self.librarian_db_path or "")
                logger.info(f"Librarian: ✅ {lib_stats.get('journal_entries', 0)} journal entries, "
                           f"{lib_stats.get('snippets', 0)} snippets")
            except Exception:
                logger.info("Librarian: ✅ ready (no stats available)")
        # v2.0: Consciousness status
        if self.consciousness:
            try:
                _c_status = self.consciousness.get_status()
                logger.info(f"Consciousness: ✅ {_c_status.get('total_tasks', 0)} tasks scored, "
                           f"drive={_c_status.get('drive_state', 'unknown')}, "
                           f"streak={_c_status.get('streak', 0)}")
            except Exception:
                logger.info("Consciousness: ✅ ready")""",
        "replace",
    ),

    # ─── PATCH 4: Pre-task strategy injection ───
    (
        "Inject strategy + aspiration context before PLAN phase",
        # Anchor: the librarian context injection block
        """        # v0.9.1: Inject librarian context (journal lessons + code snippets)
        # PREPEND so truncate_to_budget preserves it (head-biased). Exploration report
        # gets trimmed (it's regenerable) while librarian lessons (curated) survive.
        try:
            lib_ctx = get_session_context(task_state.goal, db_path=self.librarian_db_path or str(Path(__file__).parent / 'knowledge_base.db'))
            if lib_ctx:
                task_state.exploration_context = lib_ctx + "\\n" + (task_state.exploration_context or "")
                logger.info(f"  🧠 Librarian injected {len(lib_ctx)} chars of strategic context")
        except Exception as e:
            logger.debug(f"  Librarian context retrieval failed (non-fatal): {e}")""",
        # Replace with: same block + consciousness pre-task
        """        # v0.9.1: Inject librarian context (journal lessons + code snippets)
        # PREPEND so truncate_to_budget preserves it (head-biased). Exploration report
        # gets trimmed (it's regenerable) while librarian lessons (curated) survive.
        try:
            lib_ctx = get_session_context(task_state.goal, db_path=self.librarian_db_path or str(Path(__file__).parent / 'knowledge_base.db'))
            if lib_ctx:
                task_state.exploration_context = lib_ctx + "\\n" + (task_state.exploration_context or "")
                logger.info(f"  🧠 Librarian injected {len(lib_ctx)} chars of strategic context")
        except Exception as e:
            logger.debug(f"  Librarian context retrieval failed (non-fatal): {e}")

        # v2.0: Inject consciousness layer context (strategy + aspiration)
        if self.consciousness:
            try:
                _src_count = len([f for f in self.working_dir.glob("*.py")
                                  if not f.name.startswith("test_") and not f.name.startswith(".")])
                _consciousness_ctx = self.consciousness.pre_task(
                    goal=task_state.goal,
                    source_file_count=_src_count,
                    max_iterations=self.max_iterations,
                )
                if _consciousness_ctx:
                    task_state.exploration_context = (
                        _consciousness_ctx + "\\n" + (task_state.exploration_context or "")
                    )
                    logger.info(f"  🔥 Consciousness injected {len(_consciousness_ctx)} chars "
                               f"(drive={self.consciousness.aspiration.profile.drive_state})")
            except Exception as e:
                logger.debug(f"  Consciousness context failed (non-fatal): {e}")""",
        "replace",
    ),

    # ─── PATCH 5: Post-task success scoring ───
    (
        "Score task on success in _finalize_success",
        # Anchor: the self-play training data comment (unique, near end of method)
        "        # v1.2: Self-play training data collection — save (requirement → code) pairs",
        # Replace with: consciousness scoring + same line
        """        # v2.0: Score the completed task in the performance journal
        if self.consciousness:
            try:
                scored = self.consciousness.post_task_success(
                    task_state=task_state,
                    memory_records=self.memory.records,
                )
                logger.info(f"📊 Task scored: quality={scored.quality_score:.2f} "
                           f"efficiency={scored.efficiency_score:.2f} "
                           f"reinforcement={scored.reinforcement_signal:+.2f}")
            except Exception as e:
                logger.debug(f"Consciousness scoring failed (non-fatal): {e}")

        # v1.2: Self-play training data collection — save (requirement → code) pairs""",
        "replace",
    ),

    # ─── PATCH 6: Post-task failure scoring ───
    (
        "Score task on failure in _escalate",
        # Anchor: the librarian curation on failure comment (unique in _escalate)
        "        # v0.9.0: Run librarian even on failures — lessons from failures are valuable",
        # Replace with: consciousness scoring + same line
        """        # v2.0: Score the failed task (failures are valuable data for aspiration)
        if self.consciousness:
            try:
                scored = self.consciousness.post_task_failure(
                    task_state=task_state,
                    memory_records=self.memory.records,
                )
                logger.info(f"📊 Failed task scored: quality={scored.quality_score:.2f} "
                           f"reinforcement={scored.reinforcement_signal:+.2f}")
            except Exception as e:
                logger.debug(f"Consciousness failure scoring failed (non-fatal): {e}")

        # v0.9.0: Run librarian even on failures — lessons from failures are valuable""",
        "replace",
    ),
]


def validate_python(filepath: Path) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        source = filepath.read_text()
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"  ❌ SYNTAX ERROR: {e}")
        return False


def apply_patches(dry_run: bool = True):
    """Apply all patches to standalone_orchestrator.py."""
    orch_path = Path(ORCH_FILE)

    if not orch_path.exists():
        print(f"❌ {ORCH_FILE} not found in current directory.")
        print(f"   Run this from your orchestrator directory.")
        sys.exit(1)

    source = orch_path.read_text()
    original_source = source

    print(f"{'DRY RUN' if dry_run else 'APPLYING'}: {len(PATCHES)} patches to {ORCH_FILE}")
    print(f"File size: {len(source):,} bytes, {source.count(chr(10)):,} lines")
    print()

    # Verify all anchors exist before touching anything
    print("Verifying anchors...")
    all_found = True
    for i, (desc, anchor, replacement, position) in enumerate(PATCHES):
        if anchor in source:
            print(f"  ✅ Patch {i+1}: {desc}")
        else:
            # Check if already patched
            # Look for a unique string from the replacement that wouldn't be in original
            check_strs = ["_HAS_CONSCIOUSNESS", "self.consciousness", "consciousness_integration"]
            already_patched = any(cs in source for cs in check_strs) and i == 0
            if already_patched and i > 0:
                print(f"  ⚠️  Patch {i+1}: {desc} — appears already applied")
            else:
                print(f"  ❌ Patch {i+1}: {desc} — ANCHOR NOT FOUND")
                # Show what we're looking for
                lines = anchor.split("\n")
                print(f"     Looking for: {lines[0][:80]}...")
                all_found = False

    if not all_found:
        # Check if already fully patched
        if "_HAS_CONSCIOUSNESS" in source and "self.consciousness" in source:
            print()
            print("⚠️  It looks like the consciousness patches have already been applied.")
            print("   If you want to re-apply, restore from backup first:")
            print(f"   cp {ORCH_FILE}.backup.* {ORCH_FILE}")
            sys.exit(0)
        else:
            print()
            print("❌ Some anchors were not found. Cannot safely patch.")
            print("   Your orchestrator version may be different from expected.")
            sys.exit(1)

    print()

    # Apply patches
    for i, (desc, anchor, replacement, position) in enumerate(PATCHES):
        count = source.count(anchor)
        if count != 1:
            print(f"  ⚠️  Patch {i+1}: anchor appears {count} times (expected 1), skipping")
            continue

        source = source.replace(anchor, replacement, 1)
        added_lines = replacement.count("\n") - anchor.count("\n")
        print(f"  {'WOULD APPLY' if dry_run else 'APPLIED'} Patch {i+1}: {desc} (+{added_lines} lines)")

    if dry_run:
        print()
        print("=" * 60)
        print("DRY RUN COMPLETE — no files modified.")
        print(f"Run with --apply to apply {len(PATCHES)} patches.")
        print("=" * 60)
        return

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f"{ORCH_FILE}.backup.{timestamp}")
    shutil.copy2(orch_path, backup_path)
    print(f"\n  📦 Backup: {backup_path}")

    # Write patched file
    orch_path.write_text(source)
    print(f"  ✍️  Written: {ORCH_FILE}")

    # Validate syntax
    if validate_python(orch_path):
        print(f"  ✅ Syntax valid")
    else:
        print(f"  ❌ SYNTAX ERROR — rolling back!")
        shutil.copy2(backup_path, orch_path)
        print(f"  🔄 Restored from {backup_path}")
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"✅ All {len(PATCHES)} patches applied successfully.")
    print(f"   Backup at: {backup_path}")
    print(f"   Rollback:  cp {backup_path} {ORCH_FILE}")
    print("=" * 60)


def rollback():
    """Restore from most recent backup."""
    orch_path = Path(ORCH_FILE)
    backups = sorted(Path(".").glob(f"{ORCH_FILE}.backup.*"))
    if not backups:
        print("❌ No backup files found.")
        sys.exit(1)

    latest = backups[-1]
    print(f"Rolling back to: {latest}")
    shutil.copy2(latest, orch_path)
    print(f"✅ Restored {ORCH_FILE} from {latest}")


if __name__ == "__main__":
    if "--rollback" in sys.argv:
        rollback()
    elif "--apply" in sys.argv:
        apply_patches(dry_run=False)
    else:
        apply_patches(dry_run=True)
