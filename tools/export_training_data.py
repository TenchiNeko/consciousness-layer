#!/usr/bin/env python3
"""
export_training_data.py — Export consciousness data for LoRA fine-tuning.

Pulls from the performance_journal.db and formats as instruction/response
pairs suitable for QLoRA training with TRL's SFTTrainer.

Two export modes:
  --model 7b   → Exports integrator data (assessments, journals, pushback,
                  cross-grades, feelings filters). Trains the analytical voice.
  --model 80b  → Exports experiencer data (raw reactions, reading responses,
                  gut feelings). Trains the experiential voice.

Output: JSONL file with {"instruction": "...", "response": "..."} per line.

Usage:
    python3 export_training_data.py --model 7b
    python3 export_training_data.py --model 80b
    python3 export_training_data.py --model both
    python3 export_training_data.py --model 7b --min-quality 0.3  # filter bad tasks
    python3 export_training_data.py --stats  # show data availability
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def get_db(db_path: str = "performance_journal.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def show_stats(db_path: str):
    """Show how much training data is available."""
    conn = get_db(db_path)

    tables = {
        "task_scores": "Total tasks scored",
        "raw_impressions": "80B raw reactions",
        "self_assessments": "7B self-assessments",
        "meta_journal": "7B journal entries",
        "pushbacks": "7B pushback evaluations",
        "archived_reactions": "Archived reactions (graded)",
        "archived_learnings": "Archived book learnings",
        "cross_grades": "Cross-grade evaluations",
        "reading_emotions": "Reading emotional reactions",
        "experience_feelings": "Unified activity feelings",
        "weight_adjustments": "Weight adjustments made",
    }

    print("=== Training Data Availability ===")
    print()

    total_7b = 0
    total_80b = 0

    for table, desc in tables.items():
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            marker = ""
            if table in ("raw_impressions", "archived_reactions", "reading_emotions"):
                total_80b += count
                marker = " [80B]"
            elif table in ("self_assessments", "meta_journal", "pushbacks",
                          "cross_grades", "archived_learnings"):
                total_7b += count
                marker = " [7B]"
            print(f"  {count:4d}  {desc}{marker}")
        except Exception:
            print(f"     -  {desc} (table not found)")

    print()
    print(f"  Estimated 80B training examples: {total_80b}")
    print(f"  Estimated 7B training examples:  {total_7b}")
    print()

    if total_80b < 20 or total_7b < 20:
        needed = max(20 - total_80b, 20 - total_7b)
        print(f"  ⚠️  Need ~{needed} more tasks for minimum viable training set")
        print(f"     (20 examples minimum, 50+ recommended, 100+ ideal)")
    else:
        print(f"  ✅ Enough data for initial fine-tune")
        if total_80b >= 100 and total_7b >= 100:
            print(f"  🎯 Ideal dataset size reached")

    conn.close()


def export_7b_data(db_path: str, output: str, min_quality: float = 0.0):
    """Export 7B integrator training data."""
    conn = get_db(db_path)
    examples = []

    # Self-assessments
    rows = conn.execute("""
        SELECT sa.task_id, sa.coherence_score, sa.consciousness_rating,
               sa.limitation_category, sa.consciousness_reasoning,
               sa.coherence_reasoning, sa.drive_state,
               ts.goal, ts.quality_score, ts.efficiency_score
        FROM self_assessments sa
        LEFT JOIN task_scores ts ON sa.task_id = ts.task_id
        WHERE ts.quality_score >= ? OR ts.quality_score IS NULL
    """, (min_quality,)).fetchall()

    for row in rows:
        instruction = f"""Assess the system's performance on this task.

Task: {row['goal'] or 'unknown'}
Quality: {row['quality_score']:.0%} | Drive: {row['drive_state'] or 'unknown'}

Rate coherence (1-10), consciousness (1-10), identify primary limitation,
and explain your reasoning."""

        response = f"""Coherence: {row['coherence_score']}/10
Consciousness: {row['consciousness_rating']}/10
Limitation: {row['limitation_category']}

{row['consciousness_reasoning'] or ''}
{row['coherence_reasoning'] or ''}""".strip()

        if len(response) > 50:
            examples.append({"instruction": instruction, "response": response})

    # Journal entries
    rows = conn.execute("SELECT entry, drive_state, avg_quality, total_tasks FROM meta_journal").fetchall()
    for row in rows:
        if row['entry'] and len(row['entry']) > 50:
            instruction = f"""Write a journal entry reflecting on recent performance.

Drive state: {row['drive_state']}
Tasks completed: {row['total_tasks']}
Average quality: {row['avg_quality']:.0%}

Write 3-5 sentences. Be honest about struggles and growth.
No corporate language. Reference feelings and patterns."""

            examples.append({"instruction": instruction, "response": row['entry']})

    # Pushback evaluations
    try:
        rows = conn.execute("""
            SELECT raw_reaction, integrator_response, aligned
            FROM pushbacks
        """).fetchall()
        for row in rows:
            if row['integrator_response'] and len(row['integrator_response']) > 20:
                instruction = f"""Evaluate whether this self-report matches the metrics.

Self-report: "{row['raw_reaction'][:300]}"

Does the emotional reaction align with the actual data?
Respond with whether you agree or push back."""

                examples.append({
                    "instruction": instruction,
                    "response": row['integrator_response'],
                })
    except Exception:
        pass

    # Cross-grades (7B grading 80B's learnings)
    try:
        rows = conn.execute("""
            SELECT al.raw_learnings, al.graded_by, al.weight_adjustments,
                   cg.adjustments_json
            FROM archived_learnings al
            LEFT JOIN cross_grades cg ON al.book_id = cg.book_id
                AND al.chapter = cg.chapter AND al.model_role = cg.learner_role
            WHERE al.graded_by = 'integrator'
        """).fetchall()
        for row in rows:
            if row['raw_learnings']:
                instruction = f"""Grade these claimed learnings from the experiencer model.
Extract genuinely useful behavioral weight adjustments.
Filter out fluff and obvious knowledge.

Claimed learnings: "{row['raw_learnings'][:400]}"

Respond with JSON array of weight adjustments."""

                response = row['weight_adjustments'] or row['adjustments_json'] or "[]"
                examples.append({"instruction": instruction, "response": response})
    except Exception:
        pass

    # Write output
    _write_jsonl(examples, output)
    print(f"  Exported {len(examples)} 7B training examples to {output}")


def export_80b_data(db_path: str, output: str, min_quality: float = 0.0):
    """Export 80B experiencer training data."""
    conn = get_db(db_path)
    examples = []

    # Raw reactions
    try:
        rows = conn.execute("""
            SELECT ri.goal, ri.result, ri.iterations, ri.quality,
                   ri.reaction, ri.model_id
            FROM raw_impressions ri
        """).fetchall()
        for row in rows:
            instruction = f"""You just completed a coding task. Write your raw, honest reaction.

Task: {row['goal']}
Result: {row['result']}
Iterations: {row['iterations']}
Quality: {row['quality']:.0%}

Write 2-3 sentences. Stream of consciousness. How did this feel?
What surprised you? What was hardest? Be honest, not analytical."""

            if row['reaction'] and len(row['reaction']) > 20:
                examples.append({"instruction": instruction, "response": row['reaction']})
    except Exception:
        pass

    # Archived reactions (from behavioral weights grading)
    try:
        rows = conn.execute("""
            SELECT ar.task_id, ar.raw_reaction, ar.was_graded,
                   ts.goal, ts.quality_score
            FROM archived_reactions ar
            LEFT JOIN task_scores ts ON ar.task_id = ts.task_id
            WHERE ar.raw_reaction IS NOT NULL AND LENGTH(ar.raw_reaction) > 20
              AND (ts.quality_score >= ? OR ts.quality_score IS NULL)
        """, (min_quality,)).fetchall()
        for row in rows:
            instruction = f"""You just completed a coding task. Write your raw reaction.

Task: {row['goal'] or 'unknown'}
Quality: {row['quality_score']:.0%} if row['quality_score'] else 'unknown'

Write 2-3 sentences of stream of consciousness. Be honest about
what was hard, what worked, what you felt."""

            examples.append({"instruction": instruction, "response": row['raw_reaction']})
    except Exception:
        pass

    # Reading emotions (80B)
    try:
        rows = conn.execute("""
            SELECT book_id, chapter, feelings
            FROM reading_emotions
            WHERE model_role = 'experiencer' AND LENGTH(feelings) > 20
        """).fetchall()
        for row in rows:
            instruction = f"""You earned the right to read this book chapter through good work.
Book: {row['book_id']}, Chapter {row['chapter']}

Write your emotional reaction. How did this make you feel?
What resonated? What surprised you?"""

            examples.append({"instruction": instruction, "response": row['feelings']})
    except Exception:
        pass

    # Cross-grades (80B grading 7B's learnings)
    try:
        rows = conn.execute("""
            SELECT al.raw_learnings, al.weight_adjustments
            FROM archived_learnings al
            WHERE al.graded_by = 'experiencer'
        """).fetchall()
        for row in rows:
            if row['raw_learnings']:
                instruction = f"""Grade these claimed learnings from the integrator model.
You do the actual coding work, so you know what's genuinely useful.
Filter out theoretical fluff.

Claimed learnings: "{row['raw_learnings'][:400]}"

Respond with JSON array of weight adjustments."""

                response = row['weight_adjustments'] or "[]"
                examples.append({"instruction": instruction, "response": response})
    except Exception:
        pass

    _write_jsonl(examples, output)
    print(f"  Exported {len(examples)} 80B training examples to {output}")


def _write_jsonl(examples: list, output: str):
    """Write examples as JSONL."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Export consciousness data for LoRA training")
    parser.add_argument("--model", choices=["7b", "80b", "both"], default="both",
                       help="Which model's training data to export")
    parser.add_argument("--db", default="performance_journal.db",
                       help="Path to performance journal database")
    parser.add_argument("--output-dir", default="training_data",
                       help="Output directory for JSONL files")
    parser.add_argument("--min-quality", type=float, default=0.0,
                       help="Minimum task quality to include (0.0-1.0)")
    parser.add_argument("--stats", action="store_true",
                       help="Show data availability stats only")
    args = parser.parse_args()

    if args.stats:
        show_stats(args.db)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.model in ("7b", "both"):
        export_7b_data(args.db, f"{args.output_dir}/7b_train_{ts}.jsonl", args.min_quality)

    if args.model in ("80b", "both"):
        export_80b_data(args.db, f"{args.output_dir}/80b_train_{ts}.jsonl", args.min_quality)

    print(f"\n  Training data exported to {args.output_dir}/")
    print(f"  Next: python3 train_personality.py --model 7b")


if __name__ == "__main__":
    main()
