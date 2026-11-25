"""Why does a beamlet get its difficulty score? Exact feature contributions, per model.

    TMPDIR=/scratch/mstryja/tmp uv run python scripts/active_learning/difficulty_explain.py \\
        --config scripts/active_learning/config_difficulty_explain.yaml

Any beamlet can be explained: ``--example label=grid:<block>/<stem>`` or
``--example label=pool:<sample_id>`` (repeatable; replaces the config's examples),
with ``--pair harder:easier`` for the contrasts. Guide: scripts/docs/difficulty_explain.md

``pool_selection`` in the config draws the examples instead: per anatomy, the pool-test
beamlets nearest given score percentiles, each from a different patient, paired within
the anatomy. ``population`` adds the decile profiles of every reference beamlet.

Outputs, in one run directory: ``tables/`` (placement, contributions per feature and
group for the ensemble, each member and the sparse variant, pair contrasts, member
agreement, sum checks; CSV, plus LaTeX with ``latex_tables``), ``figures/``
(contributions, contrasts, one beam-path figure per beamlet, correlation and agreement
matrices, decile profiles, numbers beside each; ``figure_formats``, PNG by default),
``population/`` and ``summary.md`` (one data-driven sentence per pair).
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
from src.acquisition.explain.population import (
    ALL,
    decile_profiles,
    leading_group_share,
    pool_metadata,
    population_groups,
    profile_agreement,
    score_deciles,
    select_examples,
    within_stratum_pairs,
)
from src.acquisition.explain.report import make_figures, make_population_figures
from src.acquisition.explain.tables import example_table, write_csv_and_latex
from src.acquisition.features import FEATURE_NAMES
from src.acquisition.scorer import FULL, SPARSE, DifficultyScorer
from src.active_learning.ensemble.engine_grids import reference_percentile
from src.evaluation.cli import load_yaml_config, merge_config, setup_logging, setup_run_directory
from src.training.run_dir import save_resolved_config, write_manifest

logger = logging.getLogger(__name__)
METRICS = list(FEATURE_NAMES)
DEFAULTS = {"output_dir": None, "group_threshold": 0.7, "top_k": 3, "member_fits": None, "make_figures": True,
            "figure_formats": ["png"], "latex_tables": False, "run_prefix": "difficulty_explain_"}
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
            errors: pd.DataFrame, patients: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """``medium: E = 135 MeV, score at the 32nd percentile, v11 GPR 98.1% (2%/2mm) / 99.6% (3%/3mm)``."""
    names = {"gpr_2pct_2mm_pct": "2%/2mm", "gpr_3pct_3mm_pct": "3%/3mm"}
    out = {}
    for lab in labels:
        v11 = errors[(errors["label"] == lab) & (errors["engine"] == "v11") & errors["metric"].isin(names)]
        gpr = " / ".join(f"{r.value:.1f}% ({names[r.metric]})" for r in v11.itertuples())
        who = f" ({patients[lab]})" if patients and lab in patients else ""
        out[lab] = (f"{lab}{who}: E = {energies[lab]:g} MeV, score at percentile {score_pct[lab]:.0f} of pool test"
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


def _decile_text(name: str, rows: pd.DataFrame, largest: bool) -> str:
    """``top decile (n=745, mean +1.20) led by A +0.80, B +0.30``: the two largest (or most negative) groups."""
    lead = rows.sort_values("mean_contribution", ascending=not largest).head(2)
    groups = ", ".join(f"{r.group} {r.mean_contribution:+.2f}" for r in lead.itertuples())
    return (f"{name} decile (n={int(rows['n_beamlets'].iloc[0])}, mean {rows['mean_total'].iloc[0]:+.2f}) "
            f"led by {groups}")


def _population(out: Path, cfg: dict, deployed: DifficultyScorer, members: Dict[str, DifficultyScorer],
                reference: pd.DataFrame, groups, formats) -> List[str]:
    """Decile profiles of every reference beamlet (ensemble and members) into ``out / "population"``.

    Deciles come from the ensemble score, so every scorer is profiled on the same beamlets. Returns the
    summary lines.
    """
    pop_cfg = cfg["population"]
    pop = out / "population"
    pop.mkdir(exist_ok=True)
    meta = pool_metadata(pop_cfg["index"])
    missing = reference.index.difference(meta.index)
    if len(missing):
        raise KeyError(f"{len(missing)} reference beamlets have no patient or anatomy in {pop_cfg['index']}")
    n_bins = int(pop_cfg.get("n_bins", 10))
    ens, total, base = population_groups(deployed, reference, reference, groups.names)
    deciles = score_deciles(total, n_bins)
    ens.assign(total=total, baseline=base, decile=deciles, patient_id=meta["patient_id"],
               anatomy=meta["anatomy"]).rename_axis("sample_id").to_csv(pop / "contributions_ensemble.csv")
    profiles = {ENSEMBLE: decile_profiles(ens, total, deciles, meta)}
    for name, scorer in members.items():
        g, t, _ = population_groups(scorer, reference, reference, groups.names)
        profiles[name] = decile_profiles(g, t, deciles, meta)
    pd.concat([p.assign(scorer=n) for n, p in profiles.items()]).to_csv(pop / "decile_profiles.csv", index=False)
    share = leading_group_share(ens, deciles)
    share.to_csv(pop / "leading_group_share.csv", index=False)
    panels = list(dict.fromkeys(profiles[ENSEMBLE]["panel"]))
    agree = pd.concat([profile_agreement(profiles, ENSEMBLE, panel) for panel in panels], ignore_index=True)
    agree.to_csv(pop / "profile_agreement.csv", index=False)
    ens_prof = profiles[ENSEMBLE]
    if cfg["make_figures"]:
        ref_meta = meta.reindex(reference.index)
        counts = ref_meta.groupby("anatomy")["patient_id"].agg(["size", "nunique"])
        titles = {ALL: f"All anatomies: {len(reference)} beamlets, {ref_meta['patient_id'].nunique()} patients"}
        titles.update({a: f"{a}: {r['size']} beamlets, {r['nunique']} patients" for a, r in counts.iterrows()})
        make_population_figures(out, ens_prof, groups, titles, len(reference), formats=formats)

    lines = ["", "## Population: contributions per score decile (ensemble; generated)", "",
             f"{len(reference)} pool-test beamlets in {n_bins} equal-count deciles of the ensemble score (D1 easiest). "
             "Mean contribution of each group; within a decile they add up to the mean score minus the baseline.", ""]
    for panel in panels:
        sub = ens_prof[ens_prof["panel"] == panel]
        top, bottom = sub[sub["decile"] == sub["decile"].max()], sub[sub["decile"] == sub["decile"].min()]
        lines.append(f"- {panel}: {_decile_text('top', top, True)}; {_decile_text('bottom', bottom, False)}.")
    a_all = agree[agree["panel"] == ALL]
    rho = a_all["pearson_vs_reference"]
    lines += ["", f"Members vs ensemble (all anatomies): Pearson of the decile profiles {rho.min():.2f} to "
              f"{rho.max():.2f}; {int(a_all['same_top_decile_leader'].sum())} of {len(a_all)} members have the "
              "same leading group in the top decile.", ""]
    return lines


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
    out = setup_run_directory(Path(cfg["output_dir"]), prefix=str(cfg["run_prefix"]),
                              subdirs=("tables", "figures"))
    setup_logging(out, log_filename="difficulty_explain.log")
    tables, k, threshold = out / "tables", int(cfg["top_k"]), float(cfg["group_threshold"])
    formats = tuple(cfg["figure_formats"])
    src = cfg["sources"]

    deployed = DifficultyScorer.load(cfg["scorer"], variant=FULL)
    sparse = DifficultyScorer.load(cfg["scorer"], variant=SPARSE)
    reference = load_reference(cfg["reference_features"])[METRICS]
    selection, panels, patients = None, None, None
    if cfg.get("pool_selection") and not cfg.get("examples"):
        sel_cfg = cfg["pool_selection"]
        meta = pool_metadata(sel_cfg["index"])
        selection = select_examples(pd.Series(deployed.score(reference), index=reference.index), meta,
                                    sel_cfg["percentiles"], sel_cfg["levels"], by=sel_cfg.get("by", "anatomy"))
        selection.to_csv(tables / "selection.csv", index=False)
        cfg["examples"] = {r.label: f"pool:{r.sample_id}" for r in selection.itertuples()}
        cfg["pairs"] = cfg.get("pairs") or [list(p) for p in within_stratum_pairs(selection, sel_cfg["levels"])]
        patients = dict(zip(selection["label"], selection["patient_id"]))
        panels = {str(st): (list(sub["label"]), [tuple(p) for p in cfg["pairs"] if p[0] in set(sub["label"])])
                  for st, sub in selection.groupby("stratum", sort=False)}
    save_resolved_config(cfg, out / "config.yaml")

    refs = [BeamletRef.parse(label, spec) for label, spec in cfg["examples"].items()]
    pairs = [(str(h), str(e)) for h, e in cfg["pairs"]]
    labels = [r.label for r in refs]
    unknown = sorted({x for p in pairs for x in p} - set(labels))
    if unknown:
        raise typer.BadParameter(f"pairs name unknown beamlets {unknown}; known: {labels}")
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
    if cfg["latex_tables"]:
        write_csv_and_latex(wide, tables / "features_by_example", label="tab:difficulty_explain",
                            float_format="{:.3g}",
                            caption=(f"Difficulty features of the example beamlets against the reference population "
                                     f"({len(reference)} pool-test beamlets): value, percentile, robust z "
                                     f"(median and IQR/1.349) and contribution to the ensemble score. Bold: the "
                                     f"top {k} features of each harder beamlet of a pair."))
    else:
        wide.to_csv(tables / "features_by_example.csv", index=False)

    if cfg["make_figures"]:
        energies = {lab: float(examples.at[lab, "energy_mev"]) for lab in labels}
        make_figures(out, src, refs, pairs, energies, grouped, by_group, checks, groups, members,
                     _titles(labels, energies, score_pct, errors, patients),
                     {lab: f"{lab}: {energies[lab]:g} MeV, percentile {score_pct[lab]:.0f}" for lab in labels},
                     panels=panels, formats=formats)

    population_lines: List[str] = []
    if cfg.get("population"):
        population_lines = _population(out, cfg, deployed, fits.members, reference, groups, formats)

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
    if selection is not None:
        lines += ["", "## Examples drawn from the pool (per anatomy, each from a different patient)", "",
                  _markdown_table(selection[["label", "sample_id", "patient_id", "target_percentile",
                                             "stratum_percentile", "reference_percentile", "score"]].round(2))]
    lines += ["", "## Pairs (generated; numbers only)", ""] + [f"- {s}" for s in sentences]
    lines += ["", "## Stored gamma pass rates (%), 10% cutoff, global", "",
              _markdown_table(gpr.round(2).reset_index()), ""] + population_lines
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_manifest(out, config=cfg, dataset_path=Path(cfg["reference_features"]), extra={
        "kind": "difficulty_explain", "examples": cfg["examples"], "pairs": pairs, "group_threshold": threshold,
        "figure_formats": list(formats)})
    logger.info("done: %s", out)
    for s in sentences:
        logger.info("%s", s)


if __name__ == "__main__":
    typer.run(main)
