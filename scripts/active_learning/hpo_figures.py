"""Figures of a finished per-mixer HPO run (EXP-0014, ``ensemble_hpo.py``).

    uv run python scripts/active_learning/hpo_figures.py \
        --config scripts/active_learning/config_hpo_figures.yaml
    uv run python scripts/active_learning/hpo_figures.py --config <yaml> \
        --sample-id 0023fc62-4ebc-4491-bee0-32237ff24bb7 --device-index 1

Two figures, each with its numbers beside it as CSV:

- ``learning_curves``: training and validation loss per epoch of every trial,
  one column per sequence mixer, from the trials' ``metrics.jsonl``. Trials cut
  at each rung are drawn lighter; the checkpoint each member was selected on is
  starred, and a trial stopped by a non-finite loss is crossed.
- ``finalists_beamlet_<pick>_<id>``: one validation beamlet through all the
  finalists (every top-rung trial and C1): the ground truth and each model in the
  axial and coronal planes through the ground-truth Bragg peak, the integrated
  depth doses and the lateral profiles. The beamlets are named in ``sample_ids``
  or picked from the finalists' per-record pass rates by the rules in ``picks``
  (``typical``, ``disagreement``, ``hardest``). Each model's ``best.pth`` is
  rebuilt from its trial config and run on the beamlet; its loss is checked
  against the finalist CSV, so the panel shows the model the metrics describe.

The finalists' error agreement is ``ensemble_agreement.py`` with
``config_ensemble_agreement_hpo.yaml``. Nothing is trained; one GPU (or the CPU)
runs six forward passes per beamlet.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import typer
import yaml

from src.active_learning.retrospective.trainer import validation_loader
from src.evaluation.cli import (
    load_yaml_config,
    merge_config,
    on_root_filesystem,
    resolve_device,
    setup_logging,
    setup_run_directory,
)
from src.figures.hpo import MIXER_TITLES, finalist_beamlet_figure, hpo_learning_curves_figure, model_styles
from src.hpo.trial import trial_config
from src.training.checkpoints import CheckpointManager
from src.training.factory import build_adota_model
from src.training.losses import LMSE
from src.training.run_dir import save_resolved_config, write_manifest
from src.utils.scallers import inverse_minmax

app = typer.Typer(help="Figures of a finished per-mixer HPO run.", add_completion=False)
logger = logging.getLogger("hpo_figures")

DEFAULTS: Dict[str, Any] = {
    "experiment": "EXP-0014", "hpo_dir": None, "output_dir": None,
    "sample_ids": [], "picks": ["typical", "disagreement"],
    "device_index": None, "voxel_mm": 2.0, "margin_mm": 30.0, "overlay_floor_pct": 5.0,
    "loss_check_rtol": 1e-2,
}
ROLE_ORDER = ("C0", "C1", "X0", "U0")
PICKS = ("typical", "disagreement", "hardest")
PICK_TEXT = {"typical": "mean pass rate over the finalists closest to its median",
             "disagreement": "largest spread of pass rate across the finalists",
             "hardest": "lowest mean pass rate over the finalists"}


# --------------------------------------------------------------------------- pure helpers

def pick_beamlets(gpr: pd.DataFrame, rules: List[str]) -> Dict[str, str]:
    """``rule -> sample id`` from a records-by-models frame of pass rates.

    - ``typical``: the mean over the models closest to the median of that mean;
    - ``disagreement``: the largest standard deviation across the models;
    - ``hardest``: the lowest mean over the models.
    Ties go to the lower sample id, so the pick is deterministic.
    """
    unknown = sorted(set(rules) - set(PICKS))
    if unknown:
        raise ValueError(f"unknown picks {unknown}; known {PICKS}")
    frame = gpr.dropna().sort_index()
    mean, spread = frame.mean(axis=1), frame.std(axis=1)
    score = {"typical": (mean - mean.median()).abs(), "disagreement": -spread, "hardest": mean}
    return {rule: str(score[rule].sort_values(kind="mergesort").index[0]) for rule in rules}


def finalist_order(selection: Dict[str, Any], finalists: List[str]) -> List[str]:
    """Finalists grouped by mixer in the order of :data:`MIXER_TITLES`, the role
    holders first within a mixer."""
    roles = {selection[r]["trial"]: r for r in ROLE_ORDER if r in selection}
    mixer_rank = {m: i for i, m in enumerate(MIXER_TITLES)}
    return sorted(finalists, key=lambda t: (mixer_rank.get(t.rsplit("_", 1)[0], 99),
                                            ROLE_ORDER.index(roles[t]) if t in roles else 9, t))


def selected_epoch(trial_dir: Path) -> Optional[int]:
    """Completed epochs (from 1) of the checkpoint the trial's highest finished
    rung reported, or ``None`` when no rung finished."""
    best = None
    for path in sorted(trial_dir.glob("segment_*.json")):
        seg = json.loads(path.read_text())
        if not seg.get("failed") and seg.get("objective_epoch") is not None:
            best = int(seg["objective_epoch"]) + 1
    return best


# --------------------------------------------------------------------------- inputs

def read_curves(hpo_dir: Path, trials: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for t in trials:
        path = hpo_dir / "trials" / t["trial_id"] / "metrics.jsonl"
        for line in path.read_text().splitlines() if path.exists() else []:
            m = json.loads(line)
            rows.append({"trial": t["trial_id"], "mixer": t["mixer"], "epoch": int(m["epoch"]) + 1,
                         "train_loss": m["train"]["loss_combined_mean"],
                         "val_loss": m["val_loss"]["loss_combined_mean"]})
    return pd.DataFrame(rows)


def note(record: pd.Series) -> str:
    """One model's metrics on one beamlet, for its row of the beamlet figure."""
    dr80 = record["dr80_mm"]
    return (f"pass rate {record['gpr'] * 100:.1f}%\nMAPE {record['mape_pct']:.1f}%\n"
            + ("dR80 undefined" if pd.isna(dr80) else f"dR80 {dr80:+.1f} mm"))


def gamma_text(params: Dict[str, Any], cutoff: float) -> str:
    return (f"{params['dose_percent_threshold']}%/{params['distance_mm_threshold']} mm, "
            f"{cutoff:g}% dose cutoff, {'local' if params.get('local_gamma') else 'global'}")


def run_finalists(hpo_cfg: Dict[str, Any], hpo_dir: Path, trials: Dict[str, Dict[str, Any]],
                  order: List[str], sample_id: str, device: torch.device,
                  expected_loss: Dict[str, float], rtol: float) -> Dict[str, Any]:
    """The beamlet's input, ground truth and every finalist's prediction, in the
    dataset's dose unit."""
    scale = hpo_cfg["training"]["scale"]
    out: Dict[str, Any] = {"pred": {}}
    loss_fn = LMSE()
    for trial in order:
        t = trials[trial]
        config = replace(trial_config(hpo_cfg["training"], t["params"], dataset_path=hpo_cfg["dataset_path"],
                                      excluded_indexes_file=hpo_cfg["exclude_indexes_path"], mixer=t["mixer"]),
                         compile=False, num_workers=0)
        if "x" not in out:
            x, energy, y = next(iter(validation_loader(config, [sample_id])))
            out.update(x=x.to(device), energy=energy.to(device), y=y.to(device))
        model = build_adota_model(config, device)
        CheckpointManager.load_weights_only(hpo_dir / "trials" / trial / "checkpoints" / "best.pth",
                                            model=model, device=device)
        model.eval()
        with torch.no_grad():
            pred = model(out["x"], out["energy"])[0]
        loss = float(loss_fn(pred, out["y"]).item())
        if trial in expected_loss and not np.isclose(loss, expected_loss[trial], rtol=rtol):
            raise RuntimeError(f"{trial} on {sample_id}: loss {loss:.6g} here, {expected_loss[trial]:.6g} in "
                               "its finalist CSV; the rebuilt model is not the evaluated one")
        out["pred"][trial] = np.squeeze(inverse_minmax(pred.cpu().numpy(), scale["min_ds"], scale["max_ds"]))
        del model
    out["gt"] = np.squeeze(inverse_minmax(out["y"].cpu().numpy(), scale["min_ds"], scale["max_ds"]))
    out["ct"] = np.squeeze(out["x"][0, 0].cpu().numpy())
    return out


# --------------------------------------------------------------------------- command

@app.command()
def main(
    config: Annotated[Optional[Path], typer.Option(help="YAML config.")]
    = Path("scripts/active_learning/config_hpo_figures.yaml"),
    hpo_dir: Annotated[Optional[Path], typer.Option(help="The hpo_<ts> run directory.")] = None,
    output_dir: Annotated[Optional[Path], typer.Option(help="Parent of the figure directory.")] = None,
    sample_id: Annotated[Optional[List[str]], typer.Option(help="Beamlet(s) to draw; repeatable.")] = None,
    device_index: Annotated[Optional[int], typer.Option(help="CUDA index; -1 for CPU.")] = None,
) -> None:
    raw = load_yaml_config(config) if config and Path(config).exists() else {}
    cfg = merge_config({"hpo_dir": str(hpo_dir) if hpo_dir else None,
                        "output_dir": str(output_dir) if output_dir else None,
                        "sample_ids": list(sample_id) if sample_id else None,
                        "device_index": device_index}, raw, DEFAULTS)
    if not cfg["hpo_dir"]:
        raise typer.BadParameter("`hpo_dir` is required")
    run = Path(cfg["hpo_dir"])
    parent = Path(cfg["output_dir"] or run / "analysis")
    if on_root_filesystem(parent):
        raise typer.BadParameter(f"output_dir {parent} is on the root filesystem; use /scratch")
    out = setup_run_directory(parent, prefix="hpo_figures_", subdirs=())
    setup_logging(out, log_filename="hpo_figures.log")
    save_resolved_config(cfg, out / "config.yaml")

    hpo_cfg = yaml.safe_load((run / "config.yaml").read_text())
    write_manifest(out, config=cfg, dataset_path=Path(hpo_cfg["dataset_path"]), extra={"hpo_dir": str(run)})
    trials = {t["trial_id"]: t for t in json.loads((run / "trials.json").read_text())}
    summary = json.loads((run / "summary.json").read_text())
    selection = json.loads((run / "selection.json").read_text())
    budgets = [int(b) for b in summary["budgets"]]
    roles = {selection[r]["trial"]: r for r in ROLE_ORDER if r in selection}
    gamma = gamma_text(hpo_cfg["gamma_params"], float(hpo_cfg["gamma_cutoff_percent"]))
    tag = summary["gamma_label"]

    # Learning curves.
    curves = read_curves(run, list(trials.values()))
    info = pd.DataFrame([{"trial": t, "mixer": trials[t]["mixer"], "failed": t in summary["failed"],
                          "label": roles.get(t, ""),
                          "selected_epoch": selected_epoch(run / "trials" / t) if t in roles else np.nan}
                         for t in trials])
    curves.to_csv(out / "learning_curves.csv", index=False)
    info.to_csv(out / "learning_curves_trials.csv", index=False)
    n_val = len(pd.read_csv(run / "ids" / "val_ids.csv"))
    hpo_learning_curves_figure(
        curves, info, budgets, str(out / "learning_curves"),
        title=f"{cfg['experiment']} hyperparameter optimisation: loss per epoch of every trial",
        caption=(f"Combined loss (0.7 x MSE + 0.3 x IDD, fixed weights), log scale. Validation on the fixed "
                 f"subsample of {n_val:,} beamlets of the HPO validation scans. Successive halving keeps the "
                 f"best third of each mixer at {budgets[0]} and {budgets[1]} epochs; constant learning rate. "
                 f"Stars: the best-so-far checkpoint each member (C0, C1, X0, U0) was selected on."))
    logger.info("learning curves: %d trials, %d epochs logged", len(trials), len(curves))

    # One beamlet through every finalist.
    finals = finalist_order(selection, [t for t in trials
                                        if (run / "trials" / t / f"finalist_{tag}.csv").exists()])
    errors = {t: pd.read_csv(run / "trials" / t / f"finalist_{tag}.csv", dtype={"sample_id": str})
              .set_index("sample_id") for t in finals}
    gpr = pd.DataFrame({t: e["gpr"] for t, e in errors.items()})
    picks = {f"id{k}": s for k, s in enumerate(cfg["sample_ids"] or [])} or pick_beamlets(gpr, list(cfg["picks"]))
    device = resolve_device(cfg["device_index"])
    epochs = {t: max(budgets[i] for i in range(len(budgets))
                     if (run / "trials" / t / f"segment_{budgets[i]:03d}.json").exists()) for t in finals}
    labels = {t: (f"{roles[t]}: " if t in roles else "") + t + ("" if epochs[t] == budgets[-1]
                                                                 else f" ({epochs[t]} epochs)") for t in finals}
    rows = []
    for pick, sid in picks.items():
        data = run_finalists(hpo_cfg, run, trials, finals, sid, device,
                             {t: float(errors[t].loc[sid, "loss_mse"]) for t in finals if sid in errors[t].index},
                             float(cfg["loss_check_rtol"]))
        energy = float(errors[finals[0]].loc[sid, "energy_mev"])
        notes = {labels[t]: note(errors[t].loc[sid]) for t in finals}
        notes["Ground truth"] = f"MCsquare\n{energy:.1f} MeV"
        stem = out / f"finalists_beamlet_{pick}_{sid[:8]}_{tag}"
        _, curves_1d = finalist_beamlet_figure(
            data["ct"], data["gt"], {labels[t]: data["pred"][t] for t in finals}, figure_path=str(stem),
            title=f"{pick.capitalize()} validation beamlet, {energy:.1f} MeV: the {len(finals)} HPO finalists",
            subtitle=f"Beamlet {sid}, picked as the {PICK_TEXT.get(pick, 'one named in the config')}.",
            notes=notes, styles=model_styles({labels[t]: trials[t]["mixer"] for t in finals}),
            voxel_mm=float(cfg["voxel_mm"]), overlay_floor_pct=float(cfg["overlay_floor_pct"]),
            margin_mm=float(cfg["margin_mm"]),
            caption=(f"Each model is its trial's best checkpoint (C0, C1, X0, U0: the selected members; C1 trained "
                     f"{budgets[1]} epochs, the others {budgets[-1]}). Dose in percent of the ground-truth maximum, "
                     f"faded in above {cfg['overlay_floor_pct']:g}%; white contour: the ground-truth 50% isodose; "
                     f"dotted line: the ground-truth Bragg-peak depth. Integrated depth dose in percent of the "
                     f"ground-truth peak. Pass rate: gamma {gamma}. dR80: predicted minus ground-truth R80."))
        for name, frame in curves_1d.items():
            frame.to_csv(f"{stem}_{name}.csv", index=False)
        for t in finals:
            rows.append({"pick": pick, "sample_id": sid, "energy_mev": energy, "trial": t, "label": labels[t],
                         "epochs_trained": epochs[t], **errors[t].loc[sid].to_dict()})
        logger.info("%s: %s (%.1f MeV) drawn", pick, sid, energy)
    pd.DataFrame(rows).to_csv(out / f"finalists_beamlets_{tag}.csv", index=False)
    logger.info("done: %s", out)


if __name__ == "__main__":
    app()
