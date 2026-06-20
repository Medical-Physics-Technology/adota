#!/usr/bin/env bash
# General A/B sweep for the plan pipeline. For every plan x variant it runs the
# fused STREAM + GAMMA pipeline and archives that variant's dose + gamma + timing
# into <plan>/ab/<label>/, so any number of parameter settings can be compared
# side by side (go/no-go). Streaming writes no per-beamlet files, so there is
# nothing to clean up between runs.
#
# Define the sweep by editing PLANS and VARIANTS below. Each VARIANT is
# "<label>::<extra CLI args>" -- the label names the archive subdir, the args are
# appended verbatim to run_plan_opentps.py (so any --flag works: --grid-factor,
# --precision, --dose-render, ...).
#
# Run detached:
#   nohup bash scripts/run_ab.sh > run_logs/ab.out 2>&1 &
#
# Not using `set -e`: one failing plan/variant must not abort the rest.
set -u

PROJECT_ROOT="/home/mstryja/projects/adota"
PY="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="scripts/config_run_plan_opentps.yaml"
STAGES="stream,gamma"
LOG_DIR="${PROJECT_ROOT}/run_logs"

PLANS=(
  "/scratch/mstryja/opentps_plans/LUNG1-221_LUNG1-221_3beams_positive"
  "/scratch/mstryja/opentps_plans/LUNG1-062_LUNG1-062_3beams"
  "/scratch/mstryja/opentps_plans/Prostate-AEC-004_4mm_target_margin"
)

# "<label>::<extra CLI args appended to run_plan_opentps.py>"
VARIANTS=(
  "gf2_fp16::--grid-factor 2 --precision fp16"
  "gf2_fp32::--grid-factor 2 --precision fp32"
)

cd "${PROJECT_ROOT}" || exit 1
mkdir -p "${LOG_DIR}"

for P in "${PLANS[@]}"; do
  PLAN_NAME="$(basename "${P}")"
  for V in "${VARIANTS[@]}"; do
    LABEL="${V%%::*}"          # text before "::"
    ARGS="${V#*::}"            # text after  "::"
    LOG="${LOG_DIR}/${PLAN_NAME}_${LABEL}.out"
    DEST="${P}/ab/${LABEL}"
    echo "[$(date '+%F %T')] START ${PLAN_NAME} [${LABEL}] args='${ARGS}' -> ${LOG}"

    # shellcheck disable=SC2086  # ARGS is intentionally word-split into flags
    "${PY}" scripts/run_plan_opentps.py \
      --config "${CONFIG}" \
      --plan-dir "${P}" \
      --stages "${STAGES}" \
      ${ARGS} \
      --overwrite \
      > "${LOG}" 2>&1
    STATUS=$?
    echo "[$(date '+%F %T')] FINISHED ${PLAN_NAME} [${LABEL}] (exit ${STATUS})"

    # Archive this variant's outputs (dose .mhd + its raw payload together so the
    # relative ElementDataFile reference still resolves).
    mkdir -p "${DEST}"
    cp -f "${P}/Dose_ADoTA.mhd"        "${DEST}/" 2>/dev/null
    cp -f "${P}/Dose_ADoTA.raw"        "${DEST}/" 2>/dev/null
    cp -f "${P}/Dose_ADoTA.zraw"       "${DEST}/" 2>/dev/null
    cp -f "${P}/gamma_metrics.json"    "${DEST}/" 2>/dev/null
    cp -f "${P}/pipeline_timing.json"  "${DEST}/" 2>/dev/null
    cp -f "${P}/figures/"*gamma*.png   "${DEST}/" 2>/dev/null
    echo "[$(date '+%F %T')] archived -> ${DEST}"
  done
done

echo "[$(date '+%F %T')] ALL A/B RUNS DONE"
