#!/usr/bin/env bash
# Field-level resampling A/B: for each validation plan, run the fused STREAM +
# GAMMA pipeline twice -- grid_factor=1 (1mm per-beamlet, the reference) and
# grid_factor=2 (2mm field grid) -- and archive each mode's dose + gamma + timing
# into <plan>/grid_ab/{1mm,2mm}/ so the two can be compared directly (P6 go/no-go).
#
# Stream writes no per-beamlet files, so there is nothing to clean up afterwards.
#
# Run detached:
#   nohup bash scripts/run_grid_factor_ab.sh > run_logs/grid_factor_ab.out 2>&1 &
#
# Not using `set -e`: one failing plan/mode must not abort the rest.
set -u

PROJECT_ROOT="/home/mstryja/projects/adota"
PY="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="scripts/config_run_plan_opentps.yaml"
STAGES="stream,gamma"
LOG_DIR="${PROJECT_ROOT}/run_logs"

PLANS=(
  "/scratch/mstryja/opentps_plans/Prostate-AEC-004_4mm_target_margin"
  "/scratch/mstryja/opentps_plans/LUNG1-221_LUNG1-221_3beams_positive"
  "/scratch/mstryja/opentps_plans/LUNG1-062_LUNG1-062_3beams"
)

cd "${PROJECT_ROOT}" || exit 1
mkdir -p "${LOG_DIR}"

for P in "${PLANS[@]}"; do
  PLAN_NAME="$(basename "${P}")"
  for GF in 1 2; do
    LOG="${LOG_DIR}/${PLAN_NAME}_gf${GF}.out"
    DEST="${P}/grid_ab/${GF}mm"
    echo "[$(date '+%F %T')] START ${PLAN_NAME} grid_factor=${GF} -> ${LOG}"

    "${PY}" scripts/run_plan_opentps.py \
      --config "${CONFIG}" \
      --plan-dir "${P}" \
      --stages "${STAGES}" \
      --grid-factor "${GF}" \
      --overwrite \
      > "${LOG}" 2>&1
    STATUS=$?
    echo "[$(date '+%F %T')] FINISHED ${PLAN_NAME} grid_factor=${GF} (exit ${STATUS})"

    # Archive this mode's outputs for the A/B comparison (keep dose .mhd + its
    # raw payload together so the relative ElementDataFile reference still resolves).
    mkdir -p "${DEST}"
    cp -f "${P}/Dose_ADoTA.mhd" "${DEST}/" 2>/dev/null
    cp -f "${P}/Dose_ADoTA.raw" "${DEST}/" 2>/dev/null
    cp -f "${P}/Dose_ADoTA.zraw" "${DEST}/" 2>/dev/null
    cp -f "${P}/gamma_metrics.json" "${DEST}/" 2>/dev/null
    cp -f "${P}/pipeline_timing.json" "${DEST}/" 2>/dev/null
    cp -f "${P}/figures/"*gamma*.png "${DEST}/" 2>/dev/null
    echo "[$(date '+%F %T')] archived -> ${DEST}"
  done
done

echo "[$(date '+%F %T')] ALL A/B RUNS DONE"
