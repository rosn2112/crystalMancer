#!/bin/bash
# Crystal Mancer — Autoresearch Runner
# Run Karpathy's autoresearch overnight on macOS to optimize GNN architecture.
#
# Prerequisites:
#   1. Install autoresearch: pip install autoresearch
#   2. Set OPENROUTER_API_KEY environment variable
#   3. Have Crystal Mancer dataset built (crystalmancer --dry-run at minimum)
#
# Usage:
#   chmod +x autoresearch/run_autoresearch.sh
#   ./autoresearch/run_autoresearch.sh
#
# The agent will run ~100+ experiments overnight on a single GPU (MPS/CPU).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔═══════════════════════════════════════════════╗"
echo "║  🔮 Crystal Mancer × Autoresearch             ║"
echo "║  Overnight GNN Architecture Optimization      ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# Check dependencies
if ! command -v autoresearch &> /dev/null; then
    echo "❌ autoresearch not found. Install with: pip install autoresearch"
    exit 1
fi

# Set working directory
cd "$PROJECT_DIR"

# Run autoresearch with our program.md
echo "🚀 Starting autoresearch..."
echo "   Program: ${SCRIPT_DIR}/program.md"
echo "   Working dir: ${PROJECT_DIR}"
echo "   Time: $(date)"
echo ""

autoresearch \
    --program "${SCRIPT_DIR}/program.md" \
    --working-dir "${PROJECT_DIR}" \
    --max-experiments 200 \
    --experiment-timeout 360 \
    --log-dir "${PROJECT_DIR}/autoresearch_logs" \
    2>&1 | tee "${PROJECT_DIR}/autoresearch_logs/run_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Autoresearch complete! Check autoresearch_logs/ for results."
