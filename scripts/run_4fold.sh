#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (script located in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/train_4fold.yaml"
LOG_TS="$(date +%Y%m%d-%H%M%S)"
OVERALL_LOG="logs/4fold_run_${LOG_TS}.log"
mkdir -p logs

echo "Starting 4-fold CV (folds 0..3) using $CONFIG from root $ROOT_DIR" | tee -a "$OVERALL_LOG"
for FOLD in 0 1 2 3; do
  echo "=== Fold $FOLD ===" | tee -a "$OVERALL_LOG"
  WANDB_NAME="h200-fold${FOLD}"
  # Run training; train.py will update current_fold inside the config file each iteration
  python "$ROOT_DIR/src/train.py" \
    --train-config "$CONFIG" \
    --fold "$FOLD" \
    --wandb-name "$WANDB_NAME" 2>&1 | tee -a "logs/fold${FOLD}.log" || { echo "Fold $FOLD failed" | tee -a "$OVERALL_LOG"; exit 1; }
  echo "Completed fold $FOLD" | tee -a "$OVERALL_LOG"
  sleep 5
done

echo "All requested folds finished." | tee -a "$OVERALL_LOG"
