#!/usr/bin/env bash
# Run the full ADoTA plan pipeline (extract,infer,accumulate,gamma) over a list of
# OpenTPS plans, sequentially. After each plan finishes, the per-spot beamlets under
# <plan>/adota_beamlets are deleted to avoid running out of scratch storage.
#
# Each plan's full stdout/stderr is written to run_logs/<PlanName>.out.
# Run detached:
#   nohup bash scripts/run_all_plans.sh > run_logs/run_all_plans.out 2>&1 &
#
# Not using `set -e`: one failing plan must not abort the rest, and its beamlets
# should still be cleaned up.
set -u

PROJECT_ROOT="/home/mstryja/projects/adota"
PY="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="scripts/config_run_plan_opentps.yaml"
STAGES="extract,infer,accumulate,gamma"
LOG_DIR="${PROJECT_ROOT}/run_logs"

PLANS=(
  "/scratch/mstryja/opentps_plans/Prostate-AEC-003_Publication_Prostate-AEC-003_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-004_Publication_Prostate-AEC-004_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-005_Publication_Prostate-AEC-005_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-006_Publication_Prostate-AEC-006_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-007_Publication_Prostate-AEC-007_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-008_Publication_Prostate-AEC-008_100MperBeam"
)

cd "${PROJECT_ROOT}" || exit 1
mkdir -p "${LOG_DIR}"

for P in "${PLANS[@]}"; do
  PLAN_NAME="$(basename "${P}")"
  LOG="${LOG_DIR}/${PLAN_NAME}.out"
  echo "[$(date '+%F %T')] START ${PLAN_NAME} -> ${LOG}"

  "${PY}" scripts/run_plan_opentps.py \
    --config "${CONFIG}" \
    --plan-dir "${P}" \
    --stages "${STAGES}" \
    --overwrite \
    > "${LOG}" 2>&1
  STATUS=$?
  echo "[$(date '+%F %T')] FINISHED ${PLAN_NAME} (exit ${STATUS})"

  # Free scratch storage: drop the per-spot beamlets (Dose_ADoTA.mhd is kept).
  if [ -d "${P}/adota_beamlets" ]; then
    rm -rf "${P}/adota_beamlets"
    echo "[$(date '+%F %T')] removed ${P}/adota_beamlets"
  fi
done

echo "[$(date '+%F %T')] ALL PLANS DONE"
