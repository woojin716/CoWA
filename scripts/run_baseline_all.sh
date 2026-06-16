#!/bin/bash
# Baselines: run source x target combinations sequentially on a single GPU.
# Results accumulate in one run directory (results/<YYYYMMDD>/<NNN>/).

set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

EXP_ID="${1:-baseline}"
BATCH_SIZE="${2:-64}"
DEVICE="${3:-cuda}"
SEEDS="${4:-42}"  # comma-separated seed list (e.g., "42,123,456")

# source -> target mapping (sources: chexpert, nih; each adapts to the other 3)
declare -A TARGETS
TARGETS[chexpert]="mimic,vindr,nih"
TARGETS[nih]="chexpert,mimic,vindr"

# Build experiment list
JOBS=()
for SOURCE in chexpert nih; do
    IFS=',' read -ra TGTS <<< "${TARGETS[$SOURCE]}"
    for TARGET in "${TGTS[@]}"; do
        JOBS+=("${SOURCE}|${TARGET}")
    done
done

TOTAL=${#JOBS[@]}

echo "============================================"
echo "Baseline Eval: All Source-Target Combos"
echo "  EXP_ID:        $EXP_ID"
echo "  DEVICE:        $DEVICE"
echo "  BATCH_SIZE:    $BATCH_SIZE"
echo "  METHODS:       all (from Python)"
echo "  SEEDS:         $SEEDS"
echo "  Total runs:    $TOTAL (sequential)"
echo "============================================"

RESULT_DIR=$(python src/utils.py results)
echo "Results directory: $RESULT_DIR"

FAILED=0

# Parse SEEDS (comma-separated list into an array)
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
TOTAL_SEEDS=${#SEED_ARRAY[@]}

for seed_idx in "${!SEED_ARRAY[@]}"; do
    SEED="${SEED_ARRAY[$seed_idx]}"
    echo ""
    echo "========== SEED $SEED ($((seed_idx+1))/$TOTAL_SEEDS) =========="

    for i in "${!JOBS[@]}"; do
        IFS='|' read -r SOURCE TARGET <<< "${JOBS[$i]}"

        echo "[$((i+1))/$TOTAL] source=$SOURCE → target=$TARGET  (device=$DEVICE, seed=$SEED)"

        python src/run_baselines.py \
            --source "$SOURCE" \
            --target "$TARGET" \
            --method all \
            --batch-size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --run-dir "$RESULT_DIR" \
            --exp_id "${EXP_ID}_seed${SEED}" \
            --seed "$SEED" || {
            echo "FAILED: ${SOURCE}→${TARGET} (seed=$SEED)"
            FAILED=$((FAILED + 1))
        }
    done
done

echo ""
echo "============================================"
if (( FAILED > 0 )); then
    echo "WARNING: $FAILED experiments failed!"
else
    echo "All experiments completed successfully!"
fi
echo "Results saved to: ${RESULT_DIR}/"
echo "  - tta_summary.csv"
echo "  - tta_detail.csv"
echo "============================================"

exit $FAILED
