#!/bin/bash
# Ours online-eval: source 3개 × target 조합 순차 실행 (single GPU)
# 결과는 하나의 run 디렉토리(results/<YYYYMMDD>/<NNN>/)에 누적

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_NUM_THREADS=8

EXP_ID="${1:-ours}"
BATCH_SIZE="${2:-32}"
DEVICE="cuda"

# source → target 매핑 (padchest 제외)
declare -A TARGETS
TARGETS[chexpert]="mimic"
TARGETS[mimic_ch]="chexpert"
TARGETS[nih]="chexpert"

# 실험 목록 생성
EXPERIMENTS=()
for SOURCE in mimic_ch; do
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

RESULT_DIR=$(python src/run_utils.py results)
echo "Results directory: $RESULT_DIR"

FAILED=0

for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r SOURCE TARGET <<< "${EXPERIMENTS[$i]}"

    echo "[$((i+1))/$TOTAL] source=$SOURCE → target=$TARGET  (device=$DEVICE)"

    python src/run_tta_experiments_ours.py \
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
