#!/bin/bash
# Run all 8 CSSM ConvNeXt variants
#
# This script tests all combinations:
#   - Mode: pure, hybrid
#   - CSSM: standard, opponent
#   - Mixing: depthwise, dense
#
# Usage:
#   ./run_all.sh              # Quick test (1 epoch each)
#   ./run_all.sh --full       # Full training (100 epochs each)

set -e

# Parse arguments
EPOCHS=1
if [ "$1" == "--full" ]; then
    EPOCHS=100
    echo "Running FULL training (${EPOCHS} epochs per configuration)"
else
    echo "Running QUICK test (${EPOCHS} epoch per configuration)"
    echo "Use './run_all.sh --full' for full training"
fi

echo ""
echo "========================================"
echo "CSSM ConvNeXt - All 8 Variants"
echo "========================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Verify environment
if ! python -c "import jax" 2>/dev/null; then
    echo "Error: JAX not found. Please install dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Run all configurations
CONFIGS=(
    "pure standard depthwise"
    "pure standard dense"
    "pure opponent depthwise"
    "pure opponent dense"
    "hybrid standard depthwise"
    "hybrid standard dense"
    "hybrid opponent depthwise"
    "hybrid opponent dense"
)

TOTAL=${#CONFIGS[@]}
CURRENT=0

for config in "${CONFIGS[@]}"; do
    read -r MODE CSSM MIXING <<< "$config"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "----------------------------------------"
    echo "[$CURRENT/$TOTAL] ${MODE} | ${CSSM} | ${MIXING}"
    echo "----------------------------------------"

    python main.py \
        --mode "$MODE" \
        --cssm "$CSSM" \
        --mixing "$MIXING" \
        --epochs "$EPOCHS" \
        --batch_size 4 \
        --seq_len 4 \
        --no_wandb

    echo "Completed: ${MODE}_${CSSM}_${MIXING}"
done

echo ""
echo "========================================"
echo "All 8 configurations completed!"
echo "========================================"
