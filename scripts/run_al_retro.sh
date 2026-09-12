#!/usr/bin/env bash
# Run the whole retrospective active-learning benchmark unattended:
# cycle 0 once, then the three strategies over the free GPUs, then the comparison.
#
# Usage (from the repository root):
#   nohup bash scripts/run_al_retro.sh > /scratch/mstryja/adota_runs/al_retro/d30/pilot.out 2>&1 & echo "PID: $!"
#
# Knobs, all environment variables:
#   CONFIG      the loop config            (default scripts/config_al_retro_loop.yaml)
#   GPUS        space-separated device ids  (default "0 2"); strategies queue over them
#   STRATEGIES  space-separated strategies  (default "random score_topk stratified_score")
#   CYCLE0_RUN  an existing cycle-0 run directory, to skip training it again
#   SKIP_SPLITS set to 1 when the splits are already written
#
# Every stage writes its own log next to the runs (see `runs_dir` in the config);
# this script's stdout is the summary. A failed strategy does not stop the others,
# but the comparison runs only over the strategies that finished.

set -uo pipefail

CONFIG=${CONFIG:-scripts/config_al_retro_loop.yaml}
GPUS=${GPUS:-"0 2"}
STRATEGIES=${STRATEGIES:-"random score_topk stratified_score"}
CYCLE0_RUN=${CYCLE0_RUN:-}
SKIP_SPLITS=${SKIP_SPLITS:-0}

RUNS_DIR=$(grep -E '^runs_dir:' "$CONFIG" | awk '{print $2}')
SEED=$(grep -E '^seed:' "$CONFIG" | awk '{print $2}')
mkdir -p "$RUNS_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$RUNS_DIR/logs_$STAMP"
mkdir -p "$LOG_DIR"
stamp() { echo "$(date '+%F %T') $*"; }

stamp "config $CONFIG | runs $RUNS_DIR | gpus [$GPUS] | strategies [$STRATEGIES] | logs $LOG_DIR"

# ── 1. Splits ────────────────────────────────────────────────────────────────
if [ "$SKIP_SPLITS" != "1" ]; then
    stamp "=== splits"
    uv run python scripts/al_retro_loop.py splits --config "$CONFIG" > "$LOG_DIR/splits.log" 2>&1 \
        || { stamp "splits FAILED, see $LOG_DIR/splits.log"; exit 1; }
    grep -E '^\|' "$LOG_DIR/splits.log"
fi

# ── 2. Cycle 0 ───────────────────────────────────────────────────────────────
read -r -a GPU_LIST <<< "$GPUS"
if [ -z "$CYCLE0_RUN" ]; then
    stamp "=== cycle 0 on GPU ${GPU_LIST[0]}"
    uv run python scripts/al_retro_loop.py cycle0 --config "$CONFIG" --device-index "${GPU_LIST[0]}" \
        > "$LOG_DIR/cycle0.log" 2>&1 || { stamp "cycle 0 FAILED, see $LOG_DIR/cycle0.log"; exit 1; }
    CYCLE0_RUN=$(grep '^cycle-0 run:' "$LOG_DIR/cycle0.log" | tail -1 | sed 's/^cycle-0 run: //')
fi
[ -d "$CYCLE0_RUN" ] || { stamp "no cycle-0 run directory ($CYCLE0_RUN)"; exit 1; }
stamp "cycle 0: $CYCLE0_RUN"

# ── 3. Strategies, queued over the GPUs ──────────────────────────────────────
declare -A PID_GPU PID_STRATEGY
declare -a FREE=("${GPU_LIST[@]}")
FINISHED=()

launch() {
    local strategy=$1 gpu=$2
    stamp "=== $strategy on GPU $gpu"
    uv run python scripts/al_retro_loop.py run --config "$CONFIG" --strategy "$strategy" \
        --cycle0-run "$CYCLE0_RUN" --device-index "$gpu" > "$LOG_DIR/$strategy.log" 2>&1 &
    PID_GPU[$!]=$gpu
    PID_STRATEGY[$!]=$strategy
}

reap() {
    # Wait for any one child; free its GPU; remember the strategy if it succeeded.
    local pid status
    wait -n -p pid "${!PID_GPU[@]}" 2>/dev/null; status=$?
    [ -n "${pid:-}" ] || return 1
    local strategy=${PID_STRATEGY[$pid]}
    if [ "$status" -eq 0 ]; then
        stamp "$strategy finished: $(grep '^strategy ' "$LOG_DIR/$strategy.log" | tail -1 | sed 's/.*-> //')"
        FINISHED+=("$strategy")
    else
        stamp "$strategy FAILED (exit $status), see $LOG_DIR/$strategy.log"
    fi
    FREE+=("${PID_GPU[$pid]}")
    unset "PID_GPU[$pid]" "PID_STRATEGY[$pid]"
}

for strategy in $STRATEGIES; do
    while [ "${#FREE[@]}" -eq 0 ]; do reap || break; done
    gpu=${FREE[0]}; FREE=("${FREE[@]:1}")
    launch "$strategy" "$gpu"
done
while [ "${#PID_GPU[@]}" -gt 0 ]; do reap || break; done

# ── 4. Comparison ────────────────────────────────────────────────────────────
if [ "${#FINISHED[@]}" -lt 2 ]; then
    stamp "fewer than two strategies finished; no comparison"; exit 1
fi
RUN_ARGS=()
for strategy in "${FINISHED[@]}"; do
    RUN_ARGS+=(--run "$(ls -dt "$RUNS_DIR"/train_*_"${strategy}"_seed"${SEED}" | head -1)")
done
stamp "=== compare ${FINISHED[*]}"
uv run python scripts/al_compare.py --config scripts/config_al_compare.yaml \
    --output-dir "$RUNS_DIR/compare" "${RUN_ARGS[@]}" > "$LOG_DIR/compare.log" 2>&1 \
    || { stamp "compare FAILED, see $LOG_DIR/compare.log"; exit 1; }
grep '^comparison written' "$LOG_DIR/compare.log"
stamp "=== ALL DONE"
