#!/usr/bin/env bash
# Build DoseCUDA's CUDA kernels (the `dose_kernels` extension) for adota's Python
# into an independent, persistent home. Idempotent: an existing build of the same
# commit is left as-is.
#
# Why not `pip install DoseCUDA` into the adota venv: DoseCUDA pins `numpy < 2`
# (build and runtime) while adota runs numpy 2, and an extension compiled against
# numpy 1 cannot be imported under numpy 2. Installing an unlocked package into
# the venv would also be dropped by the next exact `uv sync`. So, like the MCsquare
# engine, DoseCUDA is an external, uncommitted install: only the compiled kernel
# module is built (against adota's own interpreter and numpy headers) and adota
# loads it by absolute path from the YAML key `dosecuda_kernels`. All beam-model
# handling (lookup tables, HU->RLSP, geometry) lives in adota (src/pencil_beam/),
# so DoseCUDA's Python package is never imported.
#
# Usage:
#   scripts/pencil_beam/provision_dosecuda.sh [SOURCE_REPO] [DEST_DIR] [GIT_REF]
# Defaults:
#   SOURCE_REPO=/home/mstryja/projects/DoseCUDA   (clone of jhu-som-radiation-oncology/DoseCUDA)
#   DEST_DIR=/home/mstryja/tools/dosecuda
#   GIT_REF=HEAD                                  (uncommitted edits in the clone are ignored)
# Environment:
#   PYTHON      interpreter to build for (default: this checkout's .venv/bin/python)
#   NVCC        CUDA compiler (default: nvcc on PATH, else the NVIDIA HPC SDK 12.3 one)
#   HOST_CXX    nvcc host compiler (default g++-12: nvcc 12.3 rejects gcc 13)
#   CUDA_ARCH   CMAKE_CUDA_ARCHITECTURES (default 86 = A40)
#   SMOKE_GPU   GPU index for the smoke test (default 0)
set -euo pipefail

SRC="${1:-/home/mstryja/projects/DoseCUDA}"
DST="${2:-/home/mstryja/tools/dosecuda}"
REF="${3:-HEAD}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-$HERE/.venv/bin/python}"
NVCC="${NVCC:-$(command -v nvcc || echo /opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/12.3/bin/nvcc)}"
HOST_CXX="${HOST_CXX:-$(command -v g++-12 || command -v g++)}"
ARCH="${CUDA_ARCH:-86}"

[[ -d "$SRC/.git" ]] || { echo "ERROR: $SRC is not a DoseCUDA git clone" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "ERROR: python not found: $PY (run 'uv sync' first)" >&2; exit 1; }
[[ -x "$NVCC" ]] || { echo "ERROR: nvcc not found: $NVCC" >&2; exit 1; }

SHA="$(git -C "$SRC" rev-parse --short=12 "$REF")"
OUT="$DST/$SHA"
echo "DoseCUDA $SHA -> $OUT  (python $("$PY" -c 'import sys; print(sys.version.split()[0])'), nvcc $("$NVCC" --version | tail -1))"

if [[ -f "$OUT/dose_kernels.so" ]]; then
    echo "  already built; leaving as-is (delete $OUT to rebuild)."
else
    BUILD="$(mktemp -d "${TMPDIR:-/scratch/$USER/tmp}/dosecuda_build.XXXXXX")"
    trap 'echo "  build failed; logs kept in $BUILD" >&2' ERR
    git -C "$SRC" archive "$REF" | tar -x -C "$BUILD"
    uv tool run cmake -S "$BUILD" -B "$BUILD/build" \
        -DSKBUILD_PROJECT_NAME=DoseCUDA \
        -DPython_EXECUTABLE="$PY" \
        -DCMAKE_CUDA_COMPILER="$NVCC" \
        -DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX" \
        -DCMAKE_CXX_COMPILER="$HOST_CXX" \
        -DCMAKE_C_COMPILER="${HOST_CXX/g++/gcc}" \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DCMAKE_BUILD_TYPE=Release > "$BUILD/configure.log"
    uv tool run cmake --build "$BUILD/build" -j 8 > "$BUILD/build.log"
    mkdir -p "$OUT"
    cp "$(find "$BUILD/build" -name 'dose_kernels*.so' | head -1)" "$OUT/dose_kernels.so"
    "$PY" - "$OUT" "$SHA" "$NVCC" "$ARCH" "$SRC" <<'EOF'
import json, subprocess, sys
import numpy
out, sha, nvcc, arch, src = sys.argv[1:]
info = {
    "dosecuda_commit": sha,
    "dosecuda_remote": subprocess.run(["git", "-C", src, "remote", "get-url", "origin"],
                                      capture_output=True, text=True).stdout.strip(),
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "nvcc": subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1],
    "cuda_architectures": arch,
}
open(f"{out}/BUILD_INFO.json", "w").write(json.dumps(info, indent=2) + "\n")
EOF
    trap - ERR
    rm -rf "$BUILD"
    echo "  built."
fi
ln -sfn "$OUT" "$DST/current"

# Smoke test: load the extension the way adota does and ray-trace a water cube.
CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}" "$PY" - "$OUT/dose_kernels.so" <<'EOF'
import importlib.machinery, importlib.util, sys
import numpy as np
loader = importlib.machinery.ExtensionFileLoader("dose_kernels", sys.argv[1])
spec = importlib.util.spec_from_file_location("dose_kernels", sys.argv[1], loader=loader)
dk = importlib.util.module_from_spec(spec); loader.exec_module(dk)
class O: pass
model, vol, beam = O(), O(), O()
model.VSADX, model.VSADY = 2000.0, 2500.0
vol.voxel_data = np.ones((40, 40, 40), np.float32)
vol.spacing = np.ones(3, np.float32); vol.origin = np.zeros(3, np.float32)
beam.iso = np.full(3, 20.0, np.float32); beam.gantry_angle = 270.0; beam.couch_angle = 0.0
wet = dk.proton_raytrace_cuda(model, vol, beam, 0)
# beam along +x: WET (cm) grows with x, ~0.1 cm per 1 mm voxel of water
prof = wet[20, 20, :]
assert prof[-1] > prof[0] and abs((prof[30] - prof[10]) - 2.0) < 0.05, prof
print(f"  smoke test PASSED: WET along beam {prof[0]:.2f} -> {prof[-1]:.2f} cm")
EOF
echo "Done. Point the YAML 'dosecuda_kernels' at: $DST/current/dose_kernels.so"
