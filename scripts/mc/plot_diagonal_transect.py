"""Appendix A.6 figure: the anti-diagonal transect of a beamlet-angle grid.

Builds one publication figure in which traversed geometry, dose and prediction
error share a single x axis (the diagonal position), replacing the overview map
plus per-case comparison set. Uses only beamlets that are already simulated and
already have an ADoTA prediction on disk -- no Monte Carlo, no inference.

  uv run python scripts/mc/plot_diagonal_transect.py --config scripts/mc/config_diagonal_transect.yaml
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer

from src.adota.config import load_yaml_config
from src.figures.diagonal_transect import angle_map_figure, diagonal_transect_figure
from src.mc_generation.diagonal_transect import extract_diagonal, path_composition

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diagonal_transect")
app = typer.Typer(help="Anti-diagonal transect figure for the beamlet-angle grid.")

_CSV_COLUMNS = [
    ("position", lambda t, c, k: int(t.positions[k])),
    ("stem", lambda t, c, k: t.stems[k]),
    ("theta_x_deg", lambda t, c, k: t.theta_x[k]),
    ("theta_y_deg", lambda t, c, k: t.theta_y[k]),
    ("gamma_pass_rate_pct", lambda t, c, k: t.gamma[k]),
    ("entry_mm", lambda t, c, k: t.entry_mm[k]),
    ("r100_mc_mm", lambda t, c, k: t.r100_mc[k]),
    ("r80_mc_mm", lambda t, c, k: t.r80_mc[k]),
    ("r80_adota_mm", lambda t, c, k: t.r80_ad[k]),
    ("delta_r80_mm", lambda t, c, k: t.r80_ad[k] - t.r80_mc[k]),
    ("path_mm", lambda t, c, k: c["path_mm"][k]),
    ("lung_frac_pct", lambda t, c, k: c["lung_frac"][k]),
    ("soft_frac_pct", lambda t, c, k: c["soft_frac"][k]),
    ("bone_frac_pct", lambda t, c, k: c["bone_frac"][k]),
    ("distal_hu", lambda t, c, k: c["distal_hu"][k]),
    ("peak_width_90pct_mm", lambda t, c, k: c["peak_width_mm"][k]),
]


def _write_metrics_csv(t, comp, path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([name for name, _ in _CSV_COLUMNS])
        for k in range(t.n):
            row = []
            for _, get in _CSV_COLUMNS:
                v = get(t, comp, k)
                row.append(v if isinstance(v, (str, int)) else f"{float(v):.3f}")
            w.writerow(row)


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config.")] = Path(
        "scripts/mc/config_diagonal_transect.yaml"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Override output dir.")] = None,
) -> None:
    cfg = load_yaml_config(config)
    grid_n = int(cfg.get("grid_n", 18))
    theta_range = cfg.get("theta_range", [-2.0, 2.0])
    crit = cfg.get("criterion_key", "g1_3_0p1")

    t = extract_diagonal(Path(cfg["exp_dir"]), Path(cfg["grids_npz"]), crit,
                         grid_n=grid_n, theta_range=theta_range)
    comp = path_composition(t)
    logger.info("%s %s %.0f MeV | %d positions | missing: %s",
                t.anatomy, t.patient, t.energy_mev, t.n, t.missing or "none")
    for k in range(t.n):
        logger.info("  %02d %s theta=(%+.2f,%+.2f) gamma=%.2f%% R80_MC=%.1fmm "
                    "dR80=%+.2fmm path=%.0fmm lung=%.0f%% distalHU=%.0f",
                    k, t.stems[k], t.theta_x[k], t.theta_y[k], t.gamma[k], t.r80_mc[k],
                    t.r80_ad[k] - t.r80_mc[k], comp["path_mm"][k], comp["lung_frac"][k],
                    comp["distal_hu"][k])

    out_dir = Path(output_dir or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg["figure_stem"]
    depth_lim = cfg.get("depth_lim")
    map_cfg = cfg.get("angle_map_figure", {})
    tr_cfg = cfg.get("transect_figure", {})

    # (a) and (b)-(f) are written as two independent files so each can be placed
    # on its own in the manuscript.
    paths = angle_map_figure(
        t, np.load(cfg["grids_npz"])[crit], str(out_dir / f"{stem}_angle_map"),
        theta_range=theta_range,
        width_in=float(map_cfg.get("width_in", 3.8)),
        height_in=float(map_cfg.get("height_in", 3.1)),
        dpi=int(map_cfg.get("dpi", 300)),
        font_scale=float(map_cfg.get("font_scale", 1.3)),
    )
    paths += diagonal_transect_figure(
        t, str(out_dir / f"{stem}_panels"),
        depth_lim=tuple(depth_lim) if depth_lim else None,
        transition=cfg.get("transition"),
        regimes=[tuple(r) for r in cfg.get("regimes", [])],
        annotate_positions=cfg.get("annotate_positions", (0, 6, 9, 17)),
        width_in=float(tr_cfg.get("width_in", 7.1)),
        height_in=float(tr_cfg.get("height_in", 8.6)),
        dpi=int(tr_cfg.get("dpi", 300)),
        font_scale=float(tr_cfg.get("font_scale", 1.3)),
    )
    if cfg.get("write_metrics_csv", True):
        csv_path = out_dir / f"{stem}_metrics.csv"
        _write_metrics_csv(t, comp, csv_path)
        paths.append(csv_path)
    logger.info("Wrote:\n  %s", "\n  ".join(str(p) for p in paths))


if __name__ == "__main__":
    app()
