"""Why does a beamlet get its difficulty score? Exact feature contributions, per model.

    TMPDIR=/scratch/mstryja/tmp uv run python scripts/active_learning/difficulty_explain.py \\
        --config scripts/active_learning/config_difficulty_explain.yaml

Any beamlet can be explained: ``--example label=grid:<block>/<stem>`` or
``--example label=pool:<sample_id>`` (repeatable; replaces the config's examples),
with ``--pair harder:easier`` for the contrasts. Guide: scripts/docs/difficulty_explain.md

Outputs, in one run directory: ``tables/`` (placement, contributions per feature and
group for the ensemble, each member and the sparse variant, pair contrasts, member
agreement, sum checks, CSV and LaTeX), ``figures/`` (contributions, contrasts, one
beam-path figure per beamlet, correlation and agreement matrices, numbers beside each)
and ``summary.md`` (one data-driven sentence per pair).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import pandas as pd
import typer

from src.acquisition.explain.analysis import (
    ENSEMBLE,
    SPARSE_LABEL,
    BeamletRef,
    agreement,
    contrasts,
    decompose_all,
    draft_sentence,
    example_errors,
    example_features,
    load_reference,
)
from src.acquisition.explain.contributions import ensemble_consistency, percentile_frame, sum_check
from src.acquisition.explain.groups import feature_groups
from src.acquisition.explain.members import (
    MemberFits,
    member_weight_table,
    reconstruct_member_fits,
)
from src.acquisition.explain.placement import placement_table
from src.acquisition.explain.report import make_figures
from src.acquisition.explain.tables import example_table, write_csv_and_latex
from src.acquisition.features import FEATURE_NAMES
from src.acquisition.scorer import FULL, SPARSE, DifficultyScorer
from src.active_learning.ensemble.engine_grids import reference_percentile
from src.evaluation.cli import load_yaml_config, merge_config, setup_logging, setup_run_directory
from src.training.run_dir import save_resolved_config, write_manifest

logger = logging.getLogger(__name__)
METRICS = list(FEATURE_NAMES)
DEFAULTS = {"output_dir": None, "group_threshold": 0.7, "top_k": 3, "member_fits": None, "make_figures": True}
CONSISTENCY_ATOL = 1e-9


def _parse_examples(items: List[str]) -> Dict[str, str]:
    out = {}
    for item in items:
        label, sep, spec = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--example wants label=source:key, got {item!r}")
        out[label] = spec
    return out


def _markdown_table(frame: pd.DataFrame) -> str:
    """A GitHub markdown table without the optional ``tabulate`` dependency."""
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False)]
    return "\n".join([head, rule] + rows)


def _titles(labels: List[str], energies: Dict[str, float], score_pct: Dict[str, float],
            errors: pd.DataFrame) -> Dict[str, str]:
    """``medium: E = 135 MeV, score at the 32nd percentile, v11 GPR 98.1% (2%/2mm) / 99.6% (3%/3mm)``."""
    names = {"gpr_2pct_2mm_pct": "2%/2mm", "gpr_3pct_3mm_pct": "3%/3mm"}
    out = {}
    for lab in labels:
        v11 = errors[(errors["label"] == lab) & (errors["engine"] == "v11") & errors["metric"].isin(names)]
        gpr = " / ".join(f"{r.value:.1f}% ({names[r.metric]})" for r in v11.itertuples())
        out[lab] = (f"{lab}: E = {energies[lab]:g} MeV, score at percentile {score_pct[lab]:.0f} of pool test"
                    + (f"\nv11 GPR {gpr}, 10% cutoff" if gpr else ""))
    return out


def _load_or_fit_members(cfg: dict, deployed: DifficultyScorer, out: Path) -> MemberFits:
    """Reuse saved member weights when given (the identity is re-checked), else refit (about 80 s)."""
    independence = load_yaml_config(Path(cfg["independence_config"]))
    cached = cfg.get("member_fits")
    if cached:
        saved = json.loads(Path(cached).read_text())
        members = {k: DifficultyScorer(variant=FULL, intercept=v["intercept"], weights=v["weights"],
                                       grids=deployed.grids, source=f"{cached}::{k}") for k, v in saved.items()}
        mean = pd.DataFrame({k: pd.Series(s.weights) for k, s in members.items()}).mean(axis=1)
        w_diff = float((mean[list(deployed.weights)] - pd.Series(deployed.weights)).abs().max())
        b_diff = abs(sum(s.intercept for s in members.values()) / len(members) - deployed.intercept)
        if max(w_diff, b_diff) > 1e-8:
            raise typer.BadParameter(f"{cached}: member weights do not average to the deployed score")
        fits = MemberFits(members, deployed, w_diff, b_diff, 0.0, n_fit=-1, n_scans=-1)
    else:
        fits = reconstruct_member_fits(independence, deployed)
    (out / "member_fits.json").write_text(json.dumps(
        {k: {"intercept": s.intercept, "weights": s.weights} for k, s in fits.members.items()}, indent=1))
    return fits


def main(
    config: Annotated[Path, typer.Option(help="YAML config.")]
    = Path("scripts/active_learning/config_difficulty_explain.yaml"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Parent of the run directory.")] = None,
    example: Annotated[Optional[List[str]], typer.Option(help="label=grid:<block>/<stem> or label=pool:<id>.")]
    = None,
    pair: Annotated[Optional[List[str]], typer.Option(help="harder:easier, by label.")] = None,
    group_threshold: Annotated[Optional[float], typer.Option(help="|Spearman| at which features group.")] = None,
    top_k: Annotated[Optional[int], typer.Option(help="Top features or groups flagged per pair.")] = None,
    member_fits: Annotated[Optional[Path], typer.Option(help="member_fits.json of an earlier run.")] = None,
    no_figures: Annotated[bool, typer.Option("--no-figures", help="Tables and summary only.")] = False,
) -> None:
    cli = {"output_dir": str(output_dir) if output_dir else None, "group_threshold": group_threshold,
           "top_k": top_k, "member_fits": str(member_fits) if member_fits else None,
           "examples": _parse_examples(example) if example else None,
           "pairs": [p.split(":") for p in pair] if pair else None, "make_figures": False if no_figures else None}
    cfg = merge_config(cli, load_yaml_config(config), DEFAULTS)
    out = setup_run_directory(Path(cfg["output_dir"]), prefix="difficulty_explain_", subdirs=("tables", "figures"))
    setup_logging(out, log_filename="difficulty_explain.log")
    save_resolved_config(cfg, out / "config.yaml")
    tables, k, threshold = out / "tables", int(cfg["top_k"]), float(cfg["group_threshold"])

    refs = [BeamletRef.parse(label, spec) for label, spec in cfg["examples"].items()]
    pairs = [(str(h), str(e)) for h, e in cfg["pairs"]]
    labels = [r.label for r in refs]
    unknown = sorted({x for p in pairs for x in p} - set(labels))
    if unknown:
        raise typer.BadParameter(f"pairs name unknown beamlets {unknown}; known: {labels}")
    src = cfg["sources"]

    deployed = DifficultyScorer.load(cfg["scorer"], variant=FULL)
    sparse = DifficultyScorer.load(cfg["scorer"], variant=SPARSE)
    reference = load_reference(cfg["reference_features"])[METRICS]
    examples = example_features(refs, src["grid"]["features"], src["pool"]["features"])
    logger.info("reference: %d beamlets; examples: %s", len(reference), labels)

    fits = _load_or_fit_members(cfg, deployed, out)
    fits.check_table().to_csv(tables / "member_fit_check.csv", index=False)
    member_weight_table(fits).to_csv(tables / "member_weights.csv")
    members = list(fits.members)
    decs = decompose_all({ENSEMBLE: deployed, **fits.members, SPARSE_LABEL: sparse}, examples[METRICS], reference)
    checks = sum_check(list(decs.values()))
    consistency = ensemble_consistency(decs[ENSEMBLE], [decs[m] for m in members])
    checks.to_csv(tables / "sum_check.csv", index=False)
    logger.info("sum check: max |residual| %.2e over %d scorer-beamlet pairs; ensemble vs member mean %.2e",
                checks["residual"].abs().max(), len(checks), consistency)
    if consistency > CONSISTENCY_ATOL:
        raise RuntimeError(f"ensemble contributions differ from the member mean by {consistency:.2e}")

    placement = placement_table(reference, examples, METRICS)
    placement.to_csv(tables / "placement.csv", index=False)
    ref_pct = percentile_frame(deployed, reference, METRICS)
    groups = feature_groups(ref_pct, threshold)
    groups.table().to_csv(tables / "groups.csv", index=False)
    groups.clusters.correlation.to_csv(tables / "correlation_spearman.csv")
    grouped, by_group, by_feature = contrasts(decs, groups, pairs, k)
    pd.concat([d.table for d in decs.values()]).to_csv(tables / "contributions_features.csv", index=False)
    grouped.to_csv(tables / "contributions_groups.csv", index=False)
    by_group.to_csv(tables / "contrast_groups.csv", index=False)
    by_feature.to_csv(tables / "contrast_features.csv", index=False)
    agree = agreement(by_group, members, k)
    agree.to_csv(tables / "agreement.csv", index=False)

    ref_scores = deployed.score(reference)
    score_pct = dict(zip(labels, reference_percentile(decs[ENSEMBLE].scores[labels].to_numpy(), ref_scores)))
    errors = example_errors(refs, src["grid"]["per_beamlet"], src["pool"]["members_dir"])
    errors.to_csv(tables / "example_errors.csv", index=False)

    ens_feat = by_feature[by_feature["scorer"] == ENSEMBLE]
    top_feats = {h: list(ens_feat[(ens_feat["harder"] == h) & ens_feat["top"]]["item"]) for h, _ in pairs}
    wide = example_table(placement, decs[ENSEMBLE].table, groups.names, top_feats)
    write_csv_and_latex(wide, tables / "features_by_example", label="tab:difficulty_explain", float_format="{:.3g}",
                        caption=(f"Difficulty features of the example beamlets against the reference population "
                                 f"({len(reference)} pool-test beamlets): value, percentile, robust z "
                                 f"(median and IQR/1.349) and contribution to the ensemble score. Bold: the "
                                 f"top {k} features of each harder beamlet of a pair."))

    if cfg["make_figures"]:
        energies = {lab: float(examples.at[lab, "energy_mev"]) for lab in labels}
        make_figures(out, src, refs, pairs, energies, grouped, by_group, checks, groups, members,
                     _titles(labels, energies, score_pct, errors),
                     {lab: f"{lab}: {energies[lab]:g} MeV, percentile {score_pct[lab]:.0f}" for lab in labels})

    sentences = [draft_sentence(h, e, by_group, by_feature, groups, placement, score_pct, agree) for h, e in pairs]
    gpr = errors[errors["metric"].str.startswith("gpr")].pivot_table(index=["engine", "metric"], columns="label",
                                                                     values="value")
    lines = ["# Difficulty score explained", "",
             f"Reference: {len(reference)} pool-test beamlets with the analytic peak inside the crop. Scorer: "
             f"`{cfg['scorer']}` (ridge on 30 percentile features). Members: {', '.join(members)}. Feature "
             f"groups at |Spearman| >= {threshold:g} on the reference ({len(groups.clusters.ids)} groups).", "",
             f"Sum check: max |residual| {checks['residual'].abs().max():.1e} over every scorer and beamlet; the "
             f"ensemble's contributions equal the member mean to {consistency:.1e}.", "",
             "## Score percentiles in the reference", ""]
    lines += [f"- {lab}: {score_pct[lab]:.1f}" for lab in labels]
    lines += ["", "## Pairs (generated; numbers only)", ""] + [f"- {s}" for s in sentences]
    lines += ["", "## Stored gamma pass rates (%), 10% cutoff, global", "",
              _markdown_table(gpr.round(2).reset_index()), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_manifest(out, config=cfg, dataset_path=Path(cfg["reference_features"]), extra={
        "kind": "difficulty_explain", "examples": cfg["examples"], "pairs": pairs, "group_threshold": threshold})
    logger.info("done: %s", out)
    for s in sentences:
        logger.info("%s", s)


if __name__ == "__main__":
    typer.run(main)
