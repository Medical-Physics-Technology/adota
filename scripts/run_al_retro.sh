#!/usr/bin/env bash
# Run the whole retrospective active-learning benchmark unattended:
# cycle 0 once, then every (seed, strategy) pair over the free GPUs, seed by
# seed, then one comparison over everything that finished.
#
# Usage (from the repository root):
#   nohup bash scripts/run_al_retro.sh > /scratch/mstryja/adota_runs/al_retro/d30/exp0011/launch.out 2>&1 & echo "PID: $!"
#
# Knobs, all environment variables:
#   CONFIG      the loop config            (default scripts/config_al_retro_loop.yaml)
#   GPUS        space-separated device ids  (default "0 1 2"); jobs queue over them
#   STRATEGIES  space-separated strategies  (default "random score_topk score_topk_mixed")
#   SEEDS       space-separated seeds       (default: the config's `seed`); the jobs are
#               ordered seed-major, so with three GPUs and three strategies each seed
#               is one round and the first round can be compared while the rest run
#   CYCLE0_RUN  an existing cycle-0 run directory, to skip training it again; every
#               seed resumes from this one checkpoint (the seed changes the selection
#               draws and the mini-batch order, not the starting weights)
#   CYCLE0_ARGS extra arguments for the cycle-0 stage only, e.g.
#               "--set lr_schedule=constant --set warmup_epochs=0" to train cycle 0 at
#               the constant 5e-4 of EXP-0009's cycle 0 while the strategy runs use
#               the config's schedule (EXP-0012)
#   SKIP_SPLITS set to 1 when the splits are already written
#
# Every stage writes its own log next to the runs (see `runs_dir` in the config);
# this script's stdout is the summary. A failed job does not stop the others,
# but the comparison runs only over the jobs that finished.

set -uo pipefail

CONFIG=${CONFIG:-scripts/config_al_retro_loop.yaml}
GPUS=${GPUS:-"0 1 2"}
STRATEGIES=${STRATEGIES:-"random score_topk score_topk_mixed"}
CYCLE0_RUN=${CYCLE0_RUN:-}
CYCLE0_ARGS=${CYCLE0_ARGS:-}
SKIP_SPLITS=${SKIP_SPLITS:-0}

RUNS_DIR=$(grep -E '^runs_dir:' "$CONFIG" | awk '{print $2}')
SEEDS=${SEEDS:-$(grep -E '^seed:' "$CONFIG" | awk '{print $2}')}
mkdir -p "$RUNS_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$RUNS_DIR/logs_$STAMP"
mkdir -p "$LOG_DIR"
stamp() { echo "$(date '+%F %T') $*"; }

stamp "config $CONFIG | runs $RUNS_DIR | gpus [$GPUS] | strategies [$STRATEGIES] | seeds [$SEEDS] | logs $LOG_DIR"

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
    stamp "=== cycle 0 on GPU ${GPU_LIST[0]} ${CYCLE0_ARGS:+(extra: $CYCLE0_ARGS)}"
    read -r -a CYCLE0_EXTRA <<< "$CYCLE0_ARGS"
    uv run python scripts/al_retro_loop.py cycle0 --config "$CONFIG" --device-index "${GPU_LIST[0]}" \
        "${CYCLE0_EXTRA[@]}" > "$LOG_DIR/cycle0.log" 2>&1 || { stamp "cycle 0 FAILED, see $LOG_DIR/cycle0.log"; exit 1; }
    CYCLE0_RUN=$(grep '^cycle-0 run:' "$LOG_DIR/cycle0.log" | tail -1 | sed 's/^cycle-0 run: //')
fi
[ -d "$CYCLE0_RUN" ] || { stamp "no cycle-0 run directory ($CYCLE0_RUN)"; exit 1; }
stamp "cycle 0: $CYCLE0_RUN"

# ── 3. (seed, strategy) jobs, queued over the GPUs ───────────────────────────
declare -A PID_GPU PID_JOB
declare -a FREE=("${GPU_LIST[@]}")
FINISHED=()

launch() {
    local strategy=$1 seed=$2 gpu=$3 job="${strategy}_seed${seed}"
    stamp "=== $job on GPU $gpu"
    uv run python scripts/al_retro_loop.py run --config "$CONFIG" --strategy "$strategy" \
        --seed "$seed" --cycle0-run "$CYCLE0_RUN" --device-index "$gpu" > "$LOG_DIR/$job.log" 2>&1 &
    PID_GPU[$!]=$gpu
    PID_JOB[$!]=$job
}

reap() {
    # Wait for any one child; free its GPU; remember the job if it succeeded.
    local pid status
    wait -n -p pid "${!PID_GPU[@]}" 2>/dev/null; status=$?
    [ -n "${pid:-}" ] || return 1
    local job=${PID_JOB[$pid]}
    if [ "$status" -eq 0 ]; then
        stamp "$job finished: $(grep '^strategy ' "$LOG_DIR/$job.log" | tail -1 | sed 's/.*-> //')"
        FINISHED+=("$job")
    else
        stamp "$job FAILED (exit $status), see $LOG_DIR/$job.log"
    fi
    FREE+=("${PID_GPU[$pid]}")
    unset "PID_GPU[$pid]" "PID_JOB[$pid]"
}

for seed in $SEEDS; do
    for strategy in $STRATEGIES; do
        while [ "${#FREE[@]}" -eq 0 ]; do reap || break; done
        gpu=${FREE[0]}; FREE=("${FREE[@]:1}")
        launch "$strategy" "$seed" "$gpu"
    done
done
while [ "${#PID_GPU[@]}" -gt 0 ]; do reap || break; done

# ── 4. Comparison ────────────────────────────────────────────────────────────
if [ "${#FINISHED[@]}" -lt 2 ]; then
    stamp "fewer than two jobs finished; no comparison"; exit 1
fi
RUN_ARGS=()
for job in "${FINISHED[@]}"; do
    RUN_ARGS+=(--run "$(ls -dt "$RUNS_DIR"/train_*_"${job}" | head -1)")
done
stamp "=== compare ${FINISHED[*]}"
uv run python scripts/al_compare.py --config scripts/config_al_compare.yaml \
    --output-dir "$RUNS_DIR/compare" "${RUN_ARGS[@]}" > "$LOG_DIR/compare.log" 2>&1 \
    || { stamp "compare FAILED, see $LOG_DIR/compare.log"; exit 1; }
grep '^comparison written' "$LOG_DIR/compare.log"
stamp "=== ALL DONE"
