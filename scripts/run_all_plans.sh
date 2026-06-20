#!/usr/bin/env bash
# Run the FUSED STREAM pipeline on the 2x2x2 (field-level, grid_factor=2) grid
# followed by GAMMA over a list of OpenTPS plans, sequentially. Streaming writes
# no per-beamlet files, so there is nothing to clean up between plans; each plan
# leaves its Dose_ADoTA.mhd, gamma_metrics.json, pipeline_timing.json and the
# gamma figure in place.
#
# GRID_FACTOR, PRECISION and DOSE_RENDER are configured below and passed through to
# run_plan_opentps.py; all three are reflected in the per-plan log name so different
# settings do not overwrite each other's logs.
#   PRECISION=fp32     -> full-precision forward (reference)
#   PRECISION=fp16     -> CUDA autocast forward (faster; validate the dose vs MC)
#   DOSE_RENDER=image  -> filled jet overlay on the CT (dose_comparison figure)
#   DOSE_RENDER=contour-> clinical filled isodose contours + labeled lines
#
# Each plan's full stdout/stderr is written to run_logs/<PlanName>_<settings>.out.
# Run detached:
#   nohup bash scripts/run_all_plans.sh > run_logs/run_all_plans.out 2>&1 &
#
# Not using `set -e`: one failing plan must not abort the rest.
set -u

PROJECT_ROOT="/home/mstryja/projects/adota"
PY="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="scripts/config_run_plan_opentps.yaml"
STAGES="stream"
GRID_FACTOR=2
PRECISION=fp16          # fp32 | fp16  (forward-pass precision for the stream stage)
DOSE_RENDER=image       # image | contour  (dose_comparison figure style: filled
                        # overlay vs clinical filled isodose contours)
LOG_DIR="${PROJECT_ROOT}/run_logs"

PLANS=(
  "/scratch/mstryja/opentps_plans/LUNG1-195_lung1-195_1beam_neg90gantry"
  "/scratch/mstryja/opentps_plans/LUNG1-193_lung1-193_3beams"
  "/scratch/mstryja/opentps_plans/LUNG1-041_lung1-041_1beam_neg90gantry"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-001_Publication_Prostate-AEC-001_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-002_Publication_Prostate-AEC-002_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-003_Publication_Prostate-AEC-003_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-004_Prostate-AEC-004_1e9Primaries_per_beam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-005_Publication_Prostate-AEC-005_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-006_Publication_Prostate-AEC-006_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-007_Publication_Prostate-AEC-007_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-008_Publication_Prostate-AEC-008_100MperBeam"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-069_3-5mm_target_margin_dij_test"
)

cd "${PROJECT_ROOT}" || exit 1
mkdir -p "${LOG_DIR}"

for P in "${PLANS[@]}"; do
  PLAN_NAME="$(basename "${P}")"
  LOG="${LOG_DIR}/${PLAN_NAME}_field_level_${GRID_FACTOR}_${PRECISION}_${DOSE_RENDER}.out"
  echo "[$(date '+%F %T')] START ${PLAN_NAME} (stream gf=${GRID_FACTOR} ${PRECISION} ${DOSE_RENDER}) -> ${LOG}"

  "${PY}" scripts/run_plan_opentps.py \
    --config "${CONFIG}" \
    --plan-dir "${P}" \
    --stages "${STAGES}" \
    --grid-factor "${GRID_FACTOR}" \
    --precision "${PRECISION}" \
    --dose-render "${DOSE_RENDER}" \
    --overwrite \
    > "${LOG}" 2>&1
  STATUS=$?
  echo "[$(date '+%F %T')] FINISHED ${PLAN_NAME} (exit ${STATUS})"
done

echo "[$(date '+%F %T')] ALL PLANS DONE"
