#!/bin/bash
# Baseline eval: source × target 조합 순차 실행 (single GPU)
# 결과는 하나의 exp_id 디렉토리에 tta_summary.csv / tta_detail.csv로 저장

set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

EXP_ID="${1:-baseline_online_all}"
BATCH_SIZE="${2:-64}"
DEVICE="${3:-cuda}"
SEEDS="${4:-42}"  # 쉼표로 구분된 seed 목록 (e.g., "42,123,456")

# source → target 매핑
declare -A TARGETS
TARGETS[chexpert]="mimic,vindr,nih"
TARGETS[mimic_ch]="chexpert,nih,vindr"
TARGETS[nih]="vindr"

# 실험 목록 생성
JOBS=()
for SOURCE in nih; do
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

OUTPUT_DIR="results/tta_experiments"
RESULT_DIR="${OUTPUT_DIR}/${EXP_ID}"
mkdir -p "$RESULT_DIR"

FAILED=0

# Parse SEEDS (쉼표로 구분된 seed 목록을 배열로 변환)
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
TOTAL_SEEDS=${#SEED_ARRAY[@]}

for seed_idx in "${!SEED_ARRAY[@]}"; do
    SEED="${SEED_ARRAY[$seed_idx]}"
    echo ""
    echo "========== SEED $SEED ($((seed_idx+1))/$TOTAL_SEEDS) =========="

    for i in "${!JOBS[@]}"; do
        IFS='|' read -r SOURCE TARGET <<< "${JOBS[$i]}"

        echo "[$((i+1))/$TOTAL] source=$SOURCE → target=$TARGET  (device=$DEVICE, seed=$SEED)"

        python src/run_tta_experiments_baseline_cached.py \
            --source "$SOURCE" \
            --target "$TARGET" \
            --method all \
            --batch-size "$BATCH_SIZE" \
            --device "$DEVICE" \
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
echo ""
echo "Results saved to:"
for SEED in "${SEED_ARRAY[@]}"; do
    echo "  - ${OUTPUT_DIR}/${EXP_ID}_seed${SEED}/"
    echo "    - tta_summary.csv"
    echo "    - tta_detail.csv"
done
echo "============================================"

exit $FAILED
