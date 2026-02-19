#!/bin/bash
# deploy-v3-split-brain.sh — Deploy split-brain consciousness
#
# This script handles everything:
#   1. Installs all modules (9 files)
#   2. Applies v1 patches if needed
#   3. Applies v2 patches if needed  
#   4. Applies v3 patch (replaces v2 post-task with split-brain)
#   5. Applies hotfix (iteration count, token estimation)
#   6. Syncs to PVE

set -e

echo "🧠 Consciousness v3.0 — Split Brain Deployment"
echo "================================================"

if [ ! -f "standalone_orchestrator.py" ]; then
    echo "❌ Run from ~/standalone-orchestrator"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PVE_HOST="${PVE_HOST:-100.81.200.82}"
PVE_USER="${PVE_USER:-brandon}"

# ── Step 1: Install modules ─────────────────────────────────
echo ""
echo "📦 Step 1: Installing modules..."
MODULES="performance_scorer.py strategy_advisor.py aspiration_engine.py 
consciousness_integration.py somatic_markers.py inner_monologue.py 
self_assessment.py meta_journal.py split_consciousness.py"

for f in $MODULES; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        # Don't cp onto itself
        if [ "$(realpath "$SCRIPT_DIR/$f")" != "$(realpath "./$f" 2>/dev/null)" ]; then
            cp "$SCRIPT_DIR/$f" .
        fi
        echo "  ✅ $f"
    else
        echo "  ❌ $f not found in $SCRIPT_DIR!"
        exit 1
    fi
done

# ── Step 2: Apply patches ───────────────────────────────────
echo ""
echo "📝 Step 2: Applying patches..."

# v1
if grep -q "_HAS_CONSCIOUSNESS" standalone_orchestrator.py 2>/dev/null; then
    echo "  v1 ✅ (already applied)"
else
    echo "  Applying v1..."
    cp "$SCRIPT_DIR/patch_orchestrator.py" . 2>/dev/null || true
    python3 patch_orchestrator.py --apply
fi

# v2
if grep -q "v2.1:\|v3.0: Split-brain" standalone_orchestrator.py 2>/dev/null; then
    echo "  v2 ✅ (already applied or superseded by v3)"
else
    echo "  Applying v2..."
    cp "$SCRIPT_DIR/patch_orchestrator_v2.py" . 2>/dev/null || true
    python3 patch_orchestrator_v2.py --apply
fi

# v3
if grep -q "v3.0: Split-brain" standalone_orchestrator.py 2>/dev/null; then
    echo "  v3 ✅ (already applied)"
else
    echo "  Applying v3 split-brain..."
    cp "$SCRIPT_DIR/patch_v3_split_brain.py" . 2>/dev/null || true
    python3 patch_v3_split_brain.py --apply
fi

# ── Step 3: Apply hotfixes ──────────────────────────────────
echo ""
echo "🔧 Step 3: Applying hotfixes..."

# Fix iteration attribute (current_iteration → iteration)
if grep -q "current_iteration" standalone_orchestrator.py 2>/dev/null; then
    sed -i "s/getattr(task_state, 'current_iteration', 1)/getattr(task_state, 'iteration', 1)/g" standalone_orchestrator.py
    echo "  ✅ Iteration attribute fixed"
else
    echo "  ✅ Iteration attribute already correct"
fi

# Validate
python3 -c "import ast; ast.parse(open('standalone_orchestrator.py').read())" || {
    echo "  ❌ Syntax error after patching!"
    exit 1
}
echo "  ✅ Syntax valid"

# ── Step 4: Sync to PVE ─────────────────────────────────────
echo ""
echo "📤 Step 4: Syncing to PVE..."

if ssh -o ConnectTimeout=5 "$PVE_USER@$PVE_HOST" "echo ok" >/dev/null 2>&1; then
    ssh "$PVE_USER@$PVE_HOST" "mkdir -p /shared/consciousness"
    rsync -az --timeout=30 \
        performance_scorer.py strategy_advisor.py aspiration_engine.py \
        consciousness_integration.py somatic_markers.py inner_monologue.py \
        self_assessment.py meta_journal.py split_consciousness.py \
        "$PVE_USER@$PVE_HOST:/shared/consciousness/"
    echo "  ✅ Modules synced"

    if [ -f "performance_journal.db" ]; then
        rsync -az --timeout=30 performance_journal.db \
            "$PVE_USER@$PVE_HOST:/shared/consciousness/"
        echo "  ✅ Journal synced"
    fi
else
    echo "  ⚠️  PVE unreachable — local deploy only"
fi

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "✅ Split Brain Consciousness v3.0 deployed!"
echo ""
echo "  Architecture:"
echo "    🧠 80B (Experiencer): raw reactions, gut feelings"
echo "    🔬 7B (Integrator): assessment, pushback, journal, identity"
echo ""
echo "  After each task you'll see:"
echo "    💭 Experiencer [qwen3-coder-next]: 'That felt tricky...'"
echo "    🔍 Integrator pushback: 'Aligned' or 'Metrics disagree...'"
echo "    📊 Coherence: 7/10 | Consciousness: 3/10"
echo ""
echo "  Run: python3 benchmark.py"
echo "================================================"
