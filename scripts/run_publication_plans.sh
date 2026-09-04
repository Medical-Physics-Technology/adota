#!/usr/bin/env bash
# Timing run over the 8 publication plans (fused STREAM stage, grid_factor=2,
# fp16, batched host<->device staging).
#
# The point of this script is a CLEAN end-to-end timing measurement, so before
# each plan runs it ARCHIVES any existing pipeline_timing.json. run_plan_opentps.py
# merges its report into the file it finds, which would otherwise leave stale
# extract/infer/accumulate stages from earlier experiments in the table and
# pollute aggregate_total_s. Archives land next to the plan as
# pipeline_timing.<timestamp>.json and nothing is deleted.
#
# Figures are SKIPPED (--no-figures): the dose-comparison/DVH plots are pure-CPU
# matplotlib work that can outweigh the dose generation itself (~60 s vs ~23 s on
# LUNG1-250), so including them would measure matplotlib rather than the pipeline.
# Dose_ADoTA.mhd is still written and the figures can be regenerated from it.
#
# This machine is shared, so the 1/5/15-minute load average is recorded before and
# after each plan into run_logs/pubtiming_load.tsv. GPU-side steps (flux, prep,
# forward) are insensitive to it; the CPU-side steps (CT cropping, deposit,
# de-rotation, write) are not, so the load column is part of the measurement.
#
# Each plan's stdout/stderr goes to run_logs/pubtiming_<PlanName>.out; a combined
# summary is assembled afterwards by scripts/summarize_publication_timing.py.
#
# Run detached:
#   nohup bash scripts/run_publication_plans.sh > run_logs/run_publication_plans.out 2>&1 &
#
# No `set -e`: one failing plan must not abort the rest.
set -u

PROJECT_ROOT="/home/mstryja/projects/adota"
PY="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="scripts/config_run_plan_opentps.yaml"
STAGES="stream"
LOG_DIR="${PROJECT_ROOT}/run_logs"
STAMP="$(date '+%Y%m%d_%H%M%S')"

PLANS=(
  "LUNG1-062_Publication_Plan_1"
  "LUNG1-195_Publication_Plan_2"
  "LUNG1-250_Publication_Plan_3"
  "LUNG1-364_Publication_Plan_5"
  "Prostate-AEC-004_Publication_Plan_1"
  "Prostate-AEC-069_Publication_Plan_2"
  "Prostate-AEC-006_Publication_Plan_3"
  "Prostate-AEC-007_Publication_Plan_4"
)
PLAN_ROOT="/scratch/mstryja/opentps_plans"

cd "${PROJECT_ROOT}" || exit 1
mkdir -p "${LOG_DIR}"

LOAD_TSV="${LOG_DIR}/pubtiming_load.tsv"
printf 'plan\twhen\tload1\tload5\tload15\n' > "${LOAD_TSV}"
log_load() {  # $1 = plan name, $2 = "before" | "after"
  read -r L1 L5 L15 _ < /proc/loadavg
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "${L1}" "${L5}" "${L15}" >> "${LOAD_TSV}"
  echo "[$(date '+%F %T')]   load ${2}: ${L1} ${L5} ${L15} (over $(nproc) cores)"
}

echo "[$(date '+%F %T')] Publication timing run ${STAMP} (${#PLANS[@]} plans, stages=${STAGES}, no figures)"

for NAME in "${PLANS[@]}"; do
  P="${PLAN_ROOT}/${NAME}"
  LOG="${LOG_DIR}/pubtiming_${NAME}.out"
  TIMING="${P}/pipeline_timing.json"

  if [ ! -d "${P}" ]; then
    echo "[$(date '+%F %T')] MISSING ${NAME} -- skipped"
    continue
  fi
  if [ -f "${TIMING}" ]; then
    mv "${TIMING}" "${P}/pipeline_timing.${STAMP}.json"
    echo "[$(date '+%F %T')]   archived previous timing -> pipeline_timing.${STAMP}.json"
  fi

  echo "[$(date '+%F %T')] START ${NAME} -> ${LOG}"
  log_load "${NAME}" before
  "${PY}" scripts/run_plan_opentps.py \
    --config "${CONFIG}" \
    --plan-dir "${P}" \
    --stages "${STAGES}" \
    --no-figures \
    --overwrite \
    > "${LOG}" 2>&1
  STATUS=$?
  log_load "${NAME}" after
  echo "[$(date '+%F %T')] FINISHED ${NAME} (exit ${STATUS})"
done

echo "[$(date '+%F %T')] ALL PLANS DONE"
"${PY}" scripts/summarize_publication_timing.py
