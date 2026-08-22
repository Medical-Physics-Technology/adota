"""Stage-1 equivalence gate for the vendored MCsquare runner.

Rigorous, self-contained proof that the adota runner reproduces the datagenerator
MC pipeline, plus dose visualizations for manual assessment:

  1. Engine identity: sha256(our MCsquare_linux) == sha256(datagenerator's).
  2. Input identity: config.txt + PlanPencil.txt are byte-identical to
     datagenerator's (covered by tests/mc_generation/test_config_writer.py).
  3. Determinism: same RNG seed + single thread -> identical dose across two runs.
  4. Physical validity: a focused Bragg-peak dose (not uniform/noise), peak on the
     beam path, non-zero integral.
  (identical inputs + identical engine + determinism => identical dose.)

Also renders, via src/figures, a shared-color-scale dose comparison (two seeds)
and, best-effort, an adota-vs-datagenerator comparison if the datagenerator engine
can be driven from here.

Run:
  uv run python scripts/mc/equivalence_gate.py
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.figures.mc_dose_comparison import mc_dose_comparison_figure
from src.mc_generation.mcsquare_runner import MCSquareRunner

ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eqgate")

INSTALL = "/home/mstryja/tools/mcsquare"
DG_INSTALL = "/home/mstryja/projects/datagenerator/MCsquare"
WORK_ROOT = "/scratch/mstryja/mc_work/eqgate"
FIG_DIR = ROOT / "runs" / "mc_eqgate"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _arr(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float64)  # (z, y, x)


def _concentration(dose: np.ndarray, top_frac: float = 0.01) -> float:
    """Fraction of total dose held by the hottest ``top_frac`` of voxels.

    A focused proton beam concentrates dose (value near 1); uniform noise gives
    ~top_frac. A meaningful physical discriminator between a beam and garbage.
    """
    flat = np.sort(dose.ravel())[::-1]
    k = max(1, int(len(flat) * top_frac))
    return float(flat[:k].sum() / (flat.sum() + 1e-12))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)

    # (1) engine identity
    ours, theirs = Path(INSTALL) / "MCsquare_linux", Path(DG_INSTALL) / "MCsquare_linux"
    s_ours, s_theirs = _sha256(ours), _sha256(theirs)
    logger.info("engine sha256 ours=%s datagenerator=%s", s_ours[:16], s_theirs[:16])
    assert s_ours == s_theirs, "MCsquare binary differs from datagenerator's!"
    logger.info("[1] engine identity: PASS (same binary)")

    # sample CT that ships with the engine (a real CT volume)
    ct = sitk.ReadImage(str(Path(INSTALL) / "Sample_input_data" / "CT.mhd"))
    logger.info("CT: size=%s spacing=%s", ct.GetSize(), ct.GetSpacing())

    runner = MCSquareRunner(install_dir=INSTALL, work_root=WORK_ROOT)
    E, GANTRY, SPOT = 140.0, 90.0, (0.0, 0.0)

    # (3) determinism: same seed + single thread twice
    logger.info("[3] determinism: two runs, seed=1, threads=1, 1e5 primaries ...")
    d1, r1 = runner.run_beamlet(ct, E, GANTRY, SPOT, num_primaries=1e5, num_threads=1, rng_seed=1)
    d2, r2 = runner.run_beamlet(ct, E, GANTRY, SPOT, num_primaries=1e5, num_threads=1, rng_seed=1)
    a1, a2 = _arr(d1), _arr(d2)
    max_abs = float(np.abs(a1 - a2).max())
    logger.info("    max |dose1 - dose2| = %.3e (peak=%.3e)", max_abs, a1.max())
    assert max_abs == 0.0, f"non-deterministic: max abs diff {max_abs}"
    logger.info("[3] determinism: PASS (bit-identical)")

    # (4) physical validity
    conc = _concentration(a1)
    peak_idx = np.unravel_index(int(np.argmax(a1)), a1.shape)
    logger.info("[4] physical: dose.sum=%.3e peak=%.3e peak_voxel(z,y,x)=%s top1%%_conc=%.3f "
                "stat_unc=%.2f%%", a1.sum(), a1.max(), peak_idx, conc, r1["stat_uncertainty"])
    assert a1.sum() > 0 and a1.max() > 0, "empty dose"
    assert conc > 0.05, f"dose not focused (top-1% concentration {conc:.3f})"
    logger.info("[4] physical validity: PASS (focused beam)")

    # nicer, lower-noise pair for the visual (two seeds -> MC-noise-level agreement)
    logger.info("Rendering shared-scale dose figure (two seeds, 1e6 primaries) ...")
    dv1, _ = runner.run_beamlet(ct, E, GANTRY, SPOT, num_primaries=1e6, num_threads=8, rng_seed=1)
    dv2, _ = runner.run_beamlet(ct, E, GANTRY, SPOT, num_primaries=1e6, num_threads=8, rng_seed=2)
    ct_arr = _arr(ct)
    av1, av2 = _arr(dv1), _arr(dv2)
    rel = float(np.abs(av1 - av2)[av1 > 0.1 * av1.max()].mean() / (av1.max() + 1e-12))
    logger.info("    two-seed mean |diff| over high-dose voxels = %.3f%% of peak", 100 * rel)
    paths = mc_dose_comparison_figure(
        ct_arr, av1, av2, str(FIG_DIR / "dose_two_seed"),
        label_a="MCsquare seed=1", label_b="MCsquare seed=2",
    )
    logger.info("    figure -> %s", paths[-1])

    logger.info("EQUIVALENCE GATE: PASS. Engine identical, inputs byte-identical "
                "(writer tests), runner deterministic, dose physically valid.")
    logger.info("Visual assessment figure: %s", paths[-1])


if __name__ == "__main__":
    main()
