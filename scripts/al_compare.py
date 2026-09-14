"""Compare the strategy runs of the retrospective benchmark (EXP-0009).

    uv run python scripts/al_compare.py --config scripts/config_al_compare.yaml
    uv run python scripts/al_compare.py --run /scratch/.../train_<ts>_al_EXP-0009_random_seed1234 \
        --run /scratch/.../train_<ts>_al_EXP-0009_score_topk_seed1234 --output-dir /scratch/...

Reads each run's ``metrics.jsonl`` and ``manifest.json``, checks that the runs
are comparable (same splits, same cycle-0 checkpoint, same evaluation subsample,
same training set size at every cycle), and writes:

- ``F1_training_curves``: loss against the cumulative epoch with cycle rules;
- ``F2_quality_vs_n_train`` and ``F3_quality_vs_epochs``: gamma pass rate, its
  tails, MAPE and dR80 against the training set size and the cumulative epoch;
- ``F4_fingerprint_score_decile`` / ``F4_fingerprint_energy``: what each
  strategy selected against the pool;
- ``summary_boundaries.csv``, ``epochs_to_quality.csv``, ``consistency.json``,
  and a CSV of the numbers beside every figure.

Figures come from :mod:`src.figures.al_curves`; every output goes to the config's
``output_dir`` on scratch.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import typer

from src.active_learning.retrospective.compare import (
    DEFAULT_THRESHOLDS,
    boundary_table,
    check_consistency,
    epochs_to_quality,
    fingerprint_table,
    fingerprints,
    quality_curves,
    read_run,
)
from src.evaluation.cli import load_yaml_config, merge_config, setup_logging, setup_run_directory
from src.figures.al_curves import (
    quality_curves_figure,
    selection_fingerprint_figure,
    training_curves_figure,
)

app = typer.Typer(help="Compare retrospective active-learning runs.", add_completion=False)
logger = logging.getLogger("al_compare")

DEFAULTS = {"runs": [], "labels": [], "output_dir": "/scratch/mstryja/adota_runs/al_retro/compare",
            "thresholds": None, "title": "Retrospective active learning (EXP-0009)"}


def _thresholds(raw: Optional[Dict]) -> Dict[str, Tuple[float, str]]:
    if not raw:
        return dict(DEFAULT_THRESHOLDS)
    return {metric: (float(spec["value"]), str(spec.get("direction", "ge")))
            for metric, spec in raw.items()}


@app.command()
def main(
    config: Annotated[Optional[Path], typer.Option(help="YAML config with runs and output_dir.")]
    = Path("scripts/config_al_compare.yaml"),
    run: Annotated[Optional[List[Path]], typer.Option(help="Run directory; repeatable.")] = None,
    label: Annotated[Optional[List[str]], typer.Option(help="Label per run; repeatable.")] = None,
    output_dir: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    raw = load_yaml_config(config) if config and Path(config).exists() else {}
    cfg = merge_config({"runs": [str(r) for r in run] if run else None,
                        "labels": label if label else None,
                        "output_dir": str(output_dir) if output_dir else None}, raw, DEFAULTS)
    if not cfg["runs"]:
        raise typer.BadParameter("no runs given: pass --run or set `runs` in the config")
    labels = list(cfg["labels"] or [])
    if labels and len(labels) != len(cfg["runs"]):
        raise typer.BadParameter("labels and runs must have the same length")

    out = setup_run_directory(Path(cfg["output_dir"]), prefix="al_compare_", subdirs=("figures",))
    setup_logging(out, log_filename="al_compare.log")
    runs = [read_run(Path(r), labels[i] if labels else None) for i, r in enumerate(cfg["runs"])]
    consistency = check_consistency(runs)
    (out / "consistency.json").write_text(json.dumps(consistency, indent=2))
    logger.info("comparing %d runs: %s", len(runs), consistency["labels"])

    gamma_label = str(consistency["gamma_label"])
    boundaries = runs[0].boundaries()
    caption = (f"Gamma {gamma_label.replace('_', ' ')}; the subsample of "
               f"{runs[0].manifest['inputs']['eval_subsample_size']} validation records every "
               f"{runs[0].manifest['config']['eval_every_n_epochs']} epochs, the full validation "
               f"set at the cycle boundaries (markers).")
    figures = out / "figures"

    summary = boundary_table(runs)
    summary.to_csv(out / f"summary_boundaries_gamma_{gamma_label}.csv", index=False)
    e2q = epochs_to_quality(runs, _thresholds(cfg["thresholds"]))
    e2q.to_csv(out / "epochs_to_quality.csv", index=False)

    losses = {r.label: r.rows[["cumulative_epoch", "cycle", "n_train", "train_loss", "val_loss"]]
              for r in runs}
    for r in runs:
        losses[r.label].assign(label=r.label).to_csv(figures / f"F1_training_curves_{r.label}.csv",
                                                     index=False)
    training_curves_figure(losses, boundaries, str(figures / "F1_training_curves"),
                           title=f"{cfg['title']}: training curves")

    curves = quality_curves(runs)
    for r in runs:
        curves[r.label].to_csv(figures / f"F2_F3_quality_{r.label}_gamma_{gamma_label}.csv",
                               index=False)
    boundary_caption = (f"Gamma {gamma_label.replace('_', ' ')}; the full validation set at the "
                        f"cycle boundaries.")
    quality_curves_figure(curves, x="n_train", x_label="Training records", prefix="full_",
                          gamma_caption=boundary_caption,
                          figure_path=str(figures / f"F2_quality_vs_n_train_gamma_{gamma_label}"),
                          title=cfg["title"])
    quality_curves_figure(curves, x="cumulative_epoch", x_label="Cumulative epoch",
                          gamma_caption=caption, boundaries=boundaries,
                          figure_path=str(figures / f"F3_quality_vs_epochs_gamma_{gamma_label}"),
                          title=cfg["title"])

    prints = fingerprints(runs)
    for key, key_label in (("score_decile", "score decile (0 easiest, 9 hardest)"),
                           ("energy", "energy bin [MeV]")):
        table = fingerprint_table(prints, key)
        if table.empty:
            logger.warning("no %s fingerprint in any run", key)
            continue
        table.to_csv(figures / f"F4_fingerprint_{key}.csv", index=False)
        selection_fingerprint_figure(table, figure_path=str(figures / f"F4_fingerprint_{key}"),
                                     key_label=key_label,
                                     title=f"{cfg['title']}: selection over {key_label}")
    typer.echo(f"comparison written to {out}")
    typer.echo(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    app()
