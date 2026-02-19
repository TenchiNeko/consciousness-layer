#!/usr/bin/env python3
"""
patch_v3_split_brain.py — Replace v2 post-task blocks with split-brain routing.

Replaces the v2 self-assessment/journal blocks (which called a single model)
with v3 split-brain blocks that route prompts to the right model:
  - EXPERIENCER prompts → 80B (self.config.llama_70b)
  - INTEGRATOR prompts → 7B (self.config.librarian_model)

Prerequisites: v1 + v2 patches must be applied.

Usage:
    python3 patch_v3_split_brain.py           # dry run
    python3 patch_v3_split_brain.py --apply   # apply
"""

import sys
import shutil
import ast
import re
from datetime import datetime
from pathlib import Path

ORCH_FILE = "standalone_orchestrator.py"


def find_and_replace_v2_blocks(source: str) -> str:
    """Replace v2 post-task blocks with v3 split-brain routing."""

    # ── Replace SUCCESS path v2 block ────────────────────────
    # Find everything from "# v2.1: Self-assessment" to the end of
    # the journal try/except block in _finalize_success

    success_v2_pattern = (
        r'                # v2\.1: Self-assessment.*?'
        r'                    logger\.debug\(f"Journal entry failed \(non-fatal\): \{e\}"\)'
    )

    success_v3_block = '''                # v3.0: Split-brain consciousness — 80B experiences, 7B integrates
                try:
                    from consciousness_integration import EXPERIENCER, INTEGRATOR
                    _prompts = self.consciousness.get_post_task_prompts(
                        task_state=task_state, scored=scored, success=True,
                    )
                    for _mp in _prompts:
                        try:
                            if _mp.role == EXPERIENCER:
                                _client = self.agent_runner._get_llm_client(self.config.llama_70b)
                            else:
                                _client = self.agent_runner._get_llm_client(self.config.librarian_model)
                            _resp = _client.chat(
                                [{"role": "user", "content": _mp.prompt}],
                                temperature=_mp.temperature,
                            )
                            if _resp and _resp.get("content"):
                                _followup = self.consciousness.record_prompt_result(
                                    purpose=_mp.purpose,
                                    response_text=_resp["content"],
                                    task_id=getattr(task_state, 'task_id', 'unknown'),
                                    scored=scored, success=True,
                                    model_id=_client.config.model_id if hasattr(_client, 'config') else '',
                                )
                                # Handle followup (e.g. pushback after raw_reaction)
                                if _followup:
                                    _f_client = self.agent_runner._get_llm_client(self.config.librarian_model)
                                    _f_resp = _f_client.chat(
                                        [{"role": "user", "content": _followup.prompt}],
                                        temperature=_followup.temperature,
                                    )
                                    if _f_resp and _f_resp.get("content"):
                                        self.consciousness.record_prompt_result(
                                            purpose=_followup.purpose,
                                            response_text=_f_resp["content"],
                                            task_id=getattr(task_state, 'task_id', 'unknown'),
                                            scored=scored, success=True,
                                        )
                        except Exception as e:
                            logger.debug(f"Consciousness prompt '{_mp.purpose}' failed (non-fatal): {e}")
                except Exception as e:
                    logger.debug(f"Split-brain consciousness failed (non-fatal): {e}")'''

    match = re.search(success_v2_pattern, source, re.DOTALL)
    if match:
        source = source[:match.start()] + success_v3_block + source[match.end():]
        print("  ✅ Success path: v2 → v3 split-brain")
    else:
        print("  ⚠️  Success path: v2 block not found (checking if v3 already applied)")
        if "v3.0: Split-brain" in source:
            print("       v3 already applied ✅")
        else:
            print("       ❌ Cannot find v2 or v3 blocks — manual intervention needed")
            return None

    # ── Replace FAILURE path v2 block ────────────────────────
    failure_v2_pattern = (
        r'                # v2\.1: Self-assessment on failure.*?'
        r'                    logger\.debug\(f"Self-assessment on failure failed \(non-fatal\): \{e\}"\)'
    )

    failure_v3_block = '''                # v3.0: Split-brain consciousness on failure
                try:
                    from consciousness_integration import EXPERIENCER, INTEGRATOR
                    _prompts = self.consciousness.get_post_task_prompts(
                        task_state=task_state, scored=scored, success=False,
                    )
                    for _mp in _prompts:
                        try:
                            if _mp.role == EXPERIENCER:
                                _client = self.agent_runner._get_llm_client(self.config.llama_70b)
                            else:
                                _client = self.agent_runner._get_llm_client(self.config.librarian_model)
                            _resp = _client.chat(
                                [{"role": "user", "content": _mp.prompt}],
                                temperature=_mp.temperature,
                            )
                            if _resp and _resp.get("content"):
                                _followup = self.consciousness.record_prompt_result(
                                    purpose=_mp.purpose,
                                    response_text=_resp["content"],
                                    task_id=getattr(task_state, 'task_id', 'unknown'),
                                    scored=scored, success=False,
                                    model_id=_client.config.model_id if hasattr(_client, 'config') else '',
                                )
                                if _followup:
                                    _f_client = self.agent_runner._get_llm_client(self.config.librarian_model)
                                    _f_resp = _f_client.chat(
                                        [{"role": "user", "content": _followup.prompt}],
                                        temperature=_followup.temperature,
                                    )
                                    if _f_resp and _f_resp.get("content"):
                                        self.consciousness.record_prompt_result(
                                            purpose=_followup.purpose,
                                            response_text=_f_resp["content"],
                                            task_id=getattr(task_state, 'task_id', 'unknown'),
                                            scored=scored, success=False,
                                        )
                        except Exception as e:
                            logger.debug(f"Consciousness prompt '{_mp.purpose}' failed (non-fatal): {e}")
                except Exception as e:
                    logger.debug(f"Split-brain consciousness on failure failed (non-fatal): {e}")'''

    match = re.search(failure_v2_pattern, source, re.DOTALL)
    if match:
        source = source[:match.start()] + failure_v3_block + source[match.end():]
        print("  ✅ Failure path: v2 → v3 split-brain")
    else:
        if "v3.0: Split-brain consciousness on failure" in source:
            print("  ✅ Failure path: v3 already applied")
        else:
            print("  ⚠️  Failure path: v2 block not found")

    return source


def apply(dry_run: bool = True):
    orch_path = Path(ORCH_FILE)
    if not orch_path.exists():
        print(f"❌ {ORCH_FILE} not found. Run from orchestrator directory.")
        sys.exit(1)

    source = orch_path.read_text()

    # Check prerequisites
    if "_HAS_CONSCIOUSNESS" not in source:
        print("❌ v1 patches not found. Run patch_orchestrator.py --apply first.")
        sys.exit(1)

    print(f"{'DRY RUN' if dry_run else 'APPLYING'}: v3 split-brain patch")
    print()

    result = find_and_replace_v2_blocks(source)
    if result is None:
        sys.exit(1)

    if dry_run:
        # Count changes
        v3_count = result.count("v3.0: Split-brain")
        print(f"\n  Would add {v3_count} v3 split-brain blocks")
        print(f"\nDRY RUN COMPLETE. Run with --apply to apply.")
        return

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = Path(f"{ORCH_FILE}.backup.v3.{timestamp}")
    shutil.copy2(orch_path, backup)
    print(f"\n  📦 Backup: {backup}")

    # Write and validate
    orch_path.write_text(result)
    try:
        ast.parse(result)
        print(f"  ✅ Syntax valid")
        print(f"\n✅ v3 split-brain patch applied.")
        print(f"   Rollback: cp {backup} {ORCH_FILE}")
    except SyntaxError as e:
        shutil.copy2(backup, orch_path)
        print(f"  ❌ SYNTAX ERROR: {e}")
        print(f"  🔄 Rolled back")
        sys.exit(1)


if __name__ == "__main__":
    if "--apply" in sys.argv:
        apply(dry_run=False)
    else:
        apply(dry_run=True)
