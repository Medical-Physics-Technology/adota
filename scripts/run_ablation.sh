#!/usr/bin/env bash
# Run the full 2x2 ablation study and aggregate results.
#
# Variants (defined by YAML configs in scripts/ablation/):
#   A  analytical       mse_idd   (baseline)
#   B  angle_broadcast  mse_idd
#   C  analytical       mse_only
#   D  angle_broadcast  mse_only
#
# Each variant runs to completion independently.  Results are aggregated
# at the end and written to the session log directory.
#
# Usage:
#   nohup bash scripts/run_ablation.sh > /tmp/ablation.out 2>&1 & echo "PID: $!"
#
# Follow progress:
#   tail -f /tmp/ablation.out

set -euo pipefail

CONFIGS=(
    scripts/ablation/config_A_analytical_mse_idd.yaml
    scripts/ablation/config_B_angle_broadcast_mse_idd.yaml
    scripts/ablation/config_C_analytical_mse_only.yaml
    scripts/ablation/config_D_angle_broadcast_mse_only.yaml
)

SESSION_DIR="runs/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SESSION_DIR"
COMBINED_LOG="$SESSION_DIR/ablation_all.log"

echo "$(date +%T) Session log: $SESSION_DIR" | tee "$COMBINED_LOG"
echo "$(date +%T) Running ${#CONFIGS[@]} variants" | tee -a "$COMBINED_LOG"

RUN_DIRS=()

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(grep '^config_name:' "$CONFIG" | awk '{print $2}')
    PER_VARIANT_LOG="$SESSION_DIR/run_${CONFIG_NAME}.log"

    echo "" | tee -a "$COMBINED_LOG"
    echo "$(date +%T) === Starting: $CONFIG_NAME ===" | tee -a "$COMBINED_LOG"
    echo "$(date +%T)     Config : $CONFIG" | tee -a "$COMBINED_LOG"

    uv run python scripts/train_adota.py \
        --config "$CONFIG" \
        2>&1 | tee "$PER_VARIANT_LOG" | tee -a "$COMBINED_LOG"

    RUN_DIR=$(grep "Run directory" "$PER_VARIANT_LOG" | tail -1 | sed 's/.*: //')
    if [ -n "$RUN_DIR" ]; then
        RUN_DIRS+=("$RUN_DIR")
        echo "$(date +%T) === Finished: $CONFIG_NAME -> $RUN_DIR ===" | tee -a "$COMBINED_LOG"
    else
        echo "$(date +%T) WARNING: Could not determine run dir for $CONFIG_NAME" | tee -a "$COMBINED_LOG"
    fi
done

echo "" | tee -a "$COMBINED_LOG"
echo "$(date +%T) === Aggregating results ===" | tee -a "$COMBINED_LOG"

SUMMARY_OUT="$SESSION_DIR/results_summary.json"
uv run python scripts/ablation/aggregate_results.py \
    "${RUN_DIRS[@]}" \
    --output "$SUMMARY_OUT" \
    2>&1 | tee -a "$COMBINED_LOG"

echo "" | tee -a "$COMBINED_LOG"
echo "$(date +%T) ALL DONE" | tee -a "$COMBINED_LOG"
echo "$(date +%T) Summary : $SUMMARY_OUT" | tee -a "$COMBINED_LOG"
