#!/bin/bash
# deploy-v31-behavioral.sh — Deploy declarative/procedural memory split
#
# Upgrades from v3.0 split-brain to v3.1 with:
#   - Behavioral weights (technical experience → numbers, not text)
#   - Feelings filter (strips solutions, keeps emotions)
#   - Archived reactions (system can't read its own technical memories)

set -e

echo "🧠 Consciousness v3.1 — Declarative/Procedural Memory Split"
echo "============================================================"

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
self_assessment.py meta_journal.py split_consciousness.py
behavioral_weights.py"

for f in $MODULES; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        if [ "$(realpath "$SCRIPT_DIR/$f" 2>/dev/null)" != "$(realpath "./$f" 2>/dev/null)" ]; then
            cp "$SCRIPT_DIR/$f" .
        fi
        echo "  ✅ $f"
    else
        echo "  ❌ $f not found!"
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
    echo "  v2 ✅ (already applied or superseded)"
else
    echo "  Applying v2..."
    cp "$SCRIPT_DIR/patch_orchestrator_v2.py" . 2>/dev/null || true
    python3 patch_orchestrator_v2.py --apply
fi

# v3
if grep -q "v3.0: Split-brain" standalone_orchestrator.py 2>/dev/null; then
    echo "  v3 ✅ (already applied)"
else
    echo "  Applying v3..."
    cp "$SCRIPT_DIR/patch_v3_split_brain.py" . 2>/dev/null || true
    python3 patch_v3_split_brain.py --apply
fi

# ── Step 3: Hotfixes ────────────────────────────────────────
echo ""
echo "🔧 Step 3: Applying hotfixes..."

if grep -q "current_iteration" standalone_orchestrator.py 2>/dev/null; then
    sed -i "s/getattr(task_state, 'current_iteration', 1)/getattr(task_state, 'iteration', 1)/g" standalone_orchestrator.py
    echo "  ✅ Iteration attribute fixed"
else
    echo "  ✅ Iteration attribute OK"
fi

python3 -c "import ast; ast.parse(open('standalone_orchestrator.py').read())" || {
    echo "  ❌ Syntax error!"; exit 1
}
echo "  ✅ Syntax valid"

# ── Step 4: Validate v3.1 ───────────────────────────────────
echo ""
echo "🔬 Step 4: Validating v3.1..."
python3 -c "
from consciousness_integration import ConsciousnessLayer
c = ConsciousnessLayer(db_path='/tmp/v31_validate.db')
s = c.get_status()
assert c.weights is not None, 'Behavioral weights not loaded!'
assert c.split is not None, 'Split consciousness not loaded!'
print('  ✅ All subsystems loaded')
print(f'  Subsystems: scorer, advisor, aspiration, somatic, assessment, journal, split-brain, behavioral-weights')
" || { echo "  ❌ Validation failed!"; exit 1; }

# ── Step 5: Sync to PVE ─────────────────────────────────────
echo ""
echo "📤 Step 5: Syncing to PVE..."

if ssh -o ConnectTimeout=5 "$PVE_USER@$PVE_HOST" "echo ok" >/dev/null 2>&1; then
    ssh "$PVE_USER@$PVE_HOST" "mkdir -p /shared/consciousness"
    rsync -az --timeout=30 \
        performance_scorer.py strategy_advisor.py aspiration_engine.py \
        consciousness_integration.py somatic_markers.py inner_monologue.py \
        self_assessment.py meta_journal.py split_consciousness.py \
        behavioral_weights.py \
        "$PVE_USER@$PVE_HOST:/shared/consciousness/"
    echo "  ✅ Synced"
else
    echo "  ⚠️  PVE unreachable — local only"
fi

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "✅ Consciousness v3.1 deployed!"
echo ""
echo "  Memory architecture:"
echo "    📖 DECLARATIVE (system can read):"
echo "       - Identity & feelings journal"
echo "       - Emotional landscape (somatic markers)"
echo "       - Behavioral weights (numbers, not text)"
echo ""
echo "    🔒 PROCEDURAL (system cannot read):"
echo "       - Raw technical reactions (archived)"
echo "       - Specific solutions & code patterns"
echo ""
echo "  Post-task pipeline:"
echo "    80B raw reaction → 7B grades → weights applied"
echo "                     → 7B filters → feelings stored"
echo "                     → raw reaction archived"
echo ""
echo "  Run: python3 benchmark.py"
echo "============================================================"
