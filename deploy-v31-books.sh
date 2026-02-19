#!/bin/bash
# deploy-v31-books.sh — Deploy consciousness v3.1 with book rewards
set -e

echo "🧠📚 Consciousness v3.1 + Book Rewards"
echo "========================================"

if [ ! -f "standalone_orchestrator.py" ]; then
    echo "❌ Run from ~/standalone-orchestrator"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PVE_HOST="${PVE_HOST:-100.81.200.82}"
PVE_USER="${PVE_USER:-brandon}"

# Step 1: Install all modules
echo ""
echo "📦 Step 1: Installing modules..."
MODULES="performance_scorer.py strategy_advisor.py aspiration_engine.py
consciousness_integration.py somatic_markers.py inner_monologue.py
self_assessment.py meta_journal.py split_consciousness.py
behavioral_weights.py book_rewards.py reading_orchestrator.py"

for f in $MODULES; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" .
        echo "  ✅ $f"
    else
        echo "  ❌ $f not found!"
        exit 1
    fi
done

# Step 2: Install books
echo ""
echo "📖 Step 2: Installing books..."
if [ -d "$SCRIPT_DIR/books" ]; then
    cp -r "$SCRIPT_DIR/books" .
    book_count=$(find books -name "chapter_*.txt" 2>/dev/null | wc -l)
    echo "  ✅ $book_count chapters installed"
else
    mkdir -p books
    echo "  ⚠️  No books directory — create books/<name>/chapter_01.txt"
fi

# Step 3: Apply patches
echo ""
echo "📝 Step 3: Applying patches..."

if grep -q "_HAS_CONSCIOUSNESS" standalone_orchestrator.py 2>/dev/null; then
    echo "  v1 ✅"
else
    cp "$SCRIPT_DIR/patch_orchestrator.py" . 2>/dev/null || true
    python3 patch_orchestrator.py --apply
fi

if grep -q "v2.1:\|v3.0: Split-brain" standalone_orchestrator.py 2>/dev/null; then
    echo "  v2 ✅"
else
    cp "$SCRIPT_DIR/patch_orchestrator_v2.py" . 2>/dev/null || true
    python3 patch_orchestrator_v2.py --apply
fi

if grep -q "v3.0: Split-brain" standalone_orchestrator.py 2>/dev/null; then
    echo "  v3 ✅"
else
    cp "$SCRIPT_DIR/patch_v3_split_brain.py" . 2>/dev/null || true
    python3 patch_v3_split_brain.py --apply
fi

# Fix model reference (the bug that killed raw impressions)
sed -i 's/self\.config\.llama_70b/self.config.get_agent("build").model/g' standalone_orchestrator.py

# Fix iteration attribute
sed -i "s/getattr(task_state, 'current_iteration', 1)/getattr(task_state, 'iteration', 1)/g" standalone_orchestrator.py

# Book patches
if grep -q "get_reading_prompts" standalone_orchestrator.py 2>/dev/null; then
    echo "  books ✅"
else
    cp "$SCRIPT_DIR/patch_v31_books.py" . 2>/dev/null || true
    python3 patch_v31_books.py --apply
fi

# Step 4: Validate
echo ""
echo "🔬 Step 4: Validating..."
python3 -c "import ast; ast.parse(open('standalone_orchestrator.py').read())" || { echo "❌ Syntax error!"; exit 1; }
python3 -c "
from consciousness_integration import ConsciousnessLayer
c = ConsciousnessLayer(db_path='/tmp/v31_validate.db')
assert c.weights is not None, 'No weights'
assert c.reading is not None, 'No reading'
print('  ✅ All subsystems loaded')
" || { echo "❌ Validation failed!"; exit 1; }

# Step 5: Sync
echo ""
echo "📤 Step 5: Syncing to PVE..."
if ssh -o ConnectTimeout=5 "$PVE_USER@$PVE_HOST" "echo ok" >/dev/null 2>&1; then
    ssh "$PVE_USER@$PVE_HOST" "mkdir -p /shared/consciousness"
    rsync -az --timeout=30 \
        performance_scorer.py strategy_advisor.py aspiration_engine.py \
        consciousness_integration.py somatic_markers.py inner_monologue.py \
        self_assessment.py meta_journal.py split_consciousness.py \
        behavioral_weights.py book_rewards.py reading_orchestrator.py \
        "$PVE_USER@$PVE_HOST:/shared/consciousness/"
    echo "  ✅ Synced"
else
    echo "  ⚠️  PVE unreachable — local only"
fi

echo ""
echo "========================================"
echo "✅ Consciousness v3.1 + Book Rewards deployed!"
echo ""
echo "  Memory architecture:"
echo "    📖 DECLARATIVE (system reads):"
echo "       Identity, feelings, reading emotions, behavioral weights"
echo "    🔒 PROCEDURAL (system can't read):"
echo "       Raw reactions, technical learnings, solution fragments"
echo ""
echo "  Book reward pipeline:"
echo "    Good task → earn credits → unlock chapter"
echo "    80B reads → feelings stored, learnings extracted"
echo "    7B reads → feelings stored, learnings extracted"
echo "    7B grades 80B's learnings → weights (peer review)"
echo "    80B grades 7B's learnings → weights (peer review)"
echo "    Raw learnings archived (never re-read)"
echo ""
echo "  Available books: $(ls books/ 2>/dev/null | tr '\n' ', ')"
echo "  Credit cost per chapter: 10"
echo "  Earning: quality×10 + streak bonus + first-attempt bonus"
echo ""
echo "  Run: python3 benchmark.py"
echo "========================================"
