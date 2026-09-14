"""One YAML shape behind all three active-learning entry points.

The pool builder, the validation-set builder and the loop share a config file, so
the candidate space, the Monte Carlo settings and the beam model are described once
and cannot drift between the set the loop is judged on and the beamlets it buys.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from src.active_learning.candidates import CandidateConfig
from src.adota.config import DEFAULT_GAMMA_PARAMS, DEFAULT_SCALE
from src.beamlets.bdl import BeamDataLibrary
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.sweep import RobustnessConfig, robustness_config_from_dict

logger = logging.getLogger(__name__)


def candidate_config_from_dict(raw: dict) -> CandidateConfig:
    """Build a :class:`CandidateConfig` from the config's ``candidates`` block."""
    defaults = CandidateConfig()
    return CandidateConfig(
        n_gantry_per_ct=int(raw.get("n_gantry_per_ct", defaults.n_gantry_per_ct)),
        n_per_field=int(raw.get("n_per_field", defaults.n_per_field)),
        energies=[float(e) for e in raw.get("energies", defaults.energies)],
        theta_half_range_deg=float(raw.get("theta_half_range_deg",
                                           defaults.theta_half_range_deg)),
        lattice_n=int(raw.get("lattice_n", defaults.lattice_n)),
        gantry_min_deg=float(raw.get("gantry_min_deg", defaults.gantry_min_deg)),
        gantry_max_deg=float(raw.get("gantry_max_deg", defaults.gantry_max_deg)),
        seed=int(raw.get("seed", defaults.seed)),
    )


def mc_from_config(cfg: dict) -> Tuple[RobustnessConfig, MCSquareRunner, BeamDataLibrary, str]:
    """The Monte Carlo side of the config: settings, runner, beam model, BDL path.

    The generator's own ``robustness`` block is reused verbatim, so a beamlet the loop
    buys is simulated under exactly the settings the reference datasets used. The
    steering lattice is forced to agree with the candidate block, because a candidate's
    grid index is recorded against it.
    """
    engine = cfg["engine"]
    bdl_name = engine.get("bdl_file", "hptc_beam_model_rsnone.txt")
    bdl_path = str(Path(engine["mcsquare_install"]) / "BDL" / bdl_name)
    bdl = BeamDataLibrary.from_file(bdl_path)

    rob = robustness_config_from_dict(cfg.get("robustness", {}), default_prefix="al")
    candidates = candidate_config_from_dict(cfg.get("candidates", {}))
    half = candidates.theta_half_range_deg
    rob.theta_x_range = (-half, half)
    rob.theta_y_range = (-half, half)
    rob.grid_n = candidates.lattice_n

    runner = MCSquareRunner(
        install_dir=engine["mcsquare_install"], work_root=engine["mc_work_dir"],
        bdl_file=bdl_name, scanner=engine.get("scanner", "default"))
    return rob, runner, bdl, bdl_path


def scale_from_config(cfg: dict) -> dict:
    """The MinMax scaling, defaulting to the deployed one."""
    return {**DEFAULT_SCALE, **(cfg.get("scale") or {})}


def gamma_params_from_config(cfg: dict) -> dict:
    """Gamma criteria, defaulting to the repository's 2%/2mm training default.

    The headline number for the loop is 3%/3mm with a 10 percent cutoff; the config
    sets it explicitly rather than relying on this fallback.
    """
    return {**DEFAULT_GAMMA_PARAMS, **(cfg.get("gamma_params") or {})}
