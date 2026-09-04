#!/usr/bin/env bash
# Provision the MCsquare engine into an independent, persistent home, decoupled
# from the (obsolete) datagenerator repo. Idempotent: safe to re-run.
#
# The engine is a self-contained, statically-linked binary + data (~53 MB):
# MCsquare_linux, BDL/, Materials/, Scanners/ (HU conversion + phantoms),
# config.txt template, src/. It is NEVER committed to git and NEVER placed in
# the adota repo; adota references it by absolute path from the YAML config.
#
# Usage:
#   scripts/mc/provision_mcsquare.sh [SOURCE_DIR] [DEST_DIR]
# Defaults:
#   SOURCE_DIR=/home/mstryja/projects/datagenerator/MCsquare   (original vendor install)
#   DEST_DIR=/home/mstryja/tools/mcsquare                      (independent engine home)
set -euo pipefail

SRC="${1:-/home/mstryja/projects/datagenerator/MCsquare}"
DST="${2:-/home/mstryja/tools/mcsquare}"

if [[ ! -x "$SRC/MCsquare_linux" ]]; then
    echo "ERROR: no MCsquare_linux under SOURCE_DIR=$SRC" >&2
    exit 1
fi

echo "Provisioning MCsquare: $SRC -> $DST"
mkdir -p "$(dirname "$DST")"
if [[ -x "$DST/MCsquare_linux" ]]; then
    echo "  destination already provisioned; leaving as-is (delete $DST to re-provision)."
else
    cp -a "$SRC" "$DST"
    echo "  copied."
fi

# Smoke test: run the bundled sample at low primaries from a scratch working dir.
W="${MC_WORK_ROOT:-/scratch/$USER/mc_work}/provision_smoketest"
rm -rf "$W"; mkdir -p "$W"
ln -sfn "$DST/Materials" "$W/Materials"
ln -sfn "$DST/Scanners" "$W/Scanners"
ln -sfn "$DST/BDL" "$W/BDL"
cp "$DST/Sample_input_data/CT.mhd" "$DST/Sample_input_data/CT.raw" \
   "$DST/Sample_input_data/PlanPencil.txt" "$W/"
sed -E 's/^Num_Primaries.*/Num_Primaries 1e5/; s/^Num_Threads.*/Num_Threads 8/;
        s#^Output_Directory.*#Output_Directory Outputs#; s#^CT_File.*#CT_File CT.mhd#;
        s#^BDL_Plan_File.*#BDL_Plan_File PlanPencil.txt#' \
    "$DST/Sample_input_data/config.txt" > "$W/config.txt"
( cd "$W" && "$DST/MCsquare_linux" config.txt >/dev/null 2>&1 )
if [[ -f "$W/Outputs/Dose.mhd" ]]; then
    echo "  smoke test PASSED (Outputs/Dose.mhd produced) -> $W"
else
    echo "  smoke test FAILED: no Outputs/Dose.mhd under $W" >&2
    exit 1
fi
echo "Done. Point the YAML 'mcsquare_install' at: $DST"
