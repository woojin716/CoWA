#!/bin/bash
# CoWA: run source x target combinations sequentially on a single GPU.
# Results accumulate in one run directory (results/<YYYYMMDD>/<NNN>/).

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_NUM_THREADS=8

EXP_ID="${1:-ours}"
BATCH_SIZE="${2:-32}"
DEVICE="cuda"

# source -> target mapping (sources: chexpert, nih; each adapts to the other 3)
declare -A TARGETS
TARGETS[chexpert]="mimic,vindr,nih"
TARGETS[nih]="chexpert,mimic,vindr"

# Build experiment list
EXPERIMENTS=()
for SOURCE in chexpert nih; do
    IFS=',' read -ra TGTS <<< "${TARGETS[$SOURCE]}"
    for TARGET in "${TGTS[@]}"; do
        EXPERIMENTS+=("${SOURCE}|${TARGET}")
    done
done

TOTAL=${#EXPERIMENTS[@]}

echo "============================================"
echo "Ours Eval: All Source-Target Combos"
echo "  EXP_ID:        $EXP_ID"
echo "  DEVICE:        $DEVICE"
echo "  BATCH_SIZE:    $BATCH_SIZE"
echo "  Total runs:    $TOTAL (sequential)"
echo "============================================"

RESULT_DIR=$(python src/utils.py results)
echo "Results directory: $RESULT_DIR"

FAILED=0

for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r SOURCE TARGET <<< "${EXPERIMENTS[$i]}"

    echo "[$((i+1))/$TOTAL] source=$SOURCE → target=$TARGET  (device=$DEVICE)"

    python src/run_cowa.py \
        --source "$SOURCE" \
        --target "$TARGET" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --run-dir "$RESULT_DIR" \
        --exp_id "${EXP_ID}" || {
        echo "FAILED: ${SOURCE}→${TARGET}"
        FAILED=$((FAILED + 1))
    }
done

echo ""
echo "============================================"
if (( FAILED > 0 )); then
    echo "WARNING: $FAILED / $TOTAL experiments failed!"
else
    echo "All $TOTAL experiments completed!"
fi
echo "Results saved to: ${RESULT_DIR}/"
echo "  - tta_summary.csv"
echo "  - tta_detail.csv"
echo "============================================"

exit $FAILED
