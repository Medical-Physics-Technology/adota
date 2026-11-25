"""Figures of the difficulty explanation, with the numbers beside each.

Feeds :mod:`src.figures.difficulty_explain` (contributions, contrasts, beam path) and
:func:`src.figures.beamlet_hardness.annotated_heatmap` (feature correlation, member
agreement) from the tables of :mod:`src.acquisition.explain.analysis`.

The beam-path marks come from the features' own definitions, so the figure shows
where a feature value comes from rather than a re-derivation: tissue-class changes
are the region boundaries of :func:`src.acquisition.features.analyse_density_regions`
inside the Bragg-peak zone, and the steepest HU step is the location of
``max_hu_gradient`` on the same flux-weighted profile.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.acquisition.explain.analysis import ENSEMBLE, BeamletRef
from src.acquisition.explain.groups import FeatureGroups
from src.acquisition.explain.profiles import (
    BeamAxisProfile,
    BeamletInputs,
    beam_axis_profile,
    load_grid_beamlet,
    load_pool_beamlet,
)
from src.acquisition.features import FeatureConfig, analyse_density_regions
from src.figures.beamlet_hardness import annotated_heatmap
from src.figures.difficulty_explain import (
    DepthMark,
    beam_path_figure,
    contrast_figure,
    contribution_figure,
    group_colors,
)

MARK_COLORS = {"interface": "#d62728", "gradient": "#ff7f0e"}


def load_inputs(ref: BeamletRef, sources: Mapping, energy_mev: float) -> BeamletInputs:
    """The model-grid inputs of a named beamlet."""
    if ref.source == "grid":
        return load_grid_beamlet(Path(sources["grid"]["root"]) / ref.block, ref.stem, energy_mev, ref.label)
    inputs = load_pool_beamlet(Path(sources["pool"]["h5"]), ref.key)
    return BeamletInputs(ref.label, "pool", inputs.ct_hu, inputs.flux, inputs.energy_mev, inputs.mc_dose)


def depth_marks(inputs: BeamletInputs, profile: BeamAxisProfile, config: FeatureConfig = FeatureConfig()
                ) -> Tuple[List[DepthMark], pd.DataFrame]:
    """Tissue-class changes in the peak zone and the steepest HU step there, as the features see them."""
    dz = profile.dz_mm
    z_min, z_max = profile.bp_zone_mm[0] / dz, profile.bp_zone_mm[1] / dz
    _, _, regions = analyse_density_regions(inputs.ct_hu, inputs.flux, z_min, z_max, config.flux_threshold_frac)
    marks, rows = [], []
    for prev, nxt in zip(regions[:-1], regions[1:]):
        depth = (nxt["start_slice"] - 0.5) * dz
        jump = nxt["mean_hu"] - prev["mean_hu"]
        label = f"{prev['label']} to {nxt['label']} ({jump:+.0f} HU)"
        marks.append(DepthMark(label, depth, depth, MARK_COLORS["interface"]))
        rows.append({"kind": "tissue-class change", "depth_mm": depth, "detail": label, "hu_step": jump})
    k0, k1 = int(np.ceil(z_min)), int(np.floor(z_max))
    if k1 > k0:
        steps = np.abs(np.diff(profile.hu_flux_weighted[k0:k1 + 1]))
        i = int(np.argmax(steps))
        depth = (k0 + i + 0.5) * dz
        label = f"steepest HU step ({steps[i]:.0f} HU per slice)"
        marks.append(DepthMark(label, depth, depth, MARK_COLORS["gradient"]))
        rows.append({"kind": "max_hu_gradient", "depth_mm": depth, "detail": label, "hu_step": float(steps[i])})
    return marks, pd.DataFrame(rows)


def _profile_frame(label: str, profile: BeamAxisProfile) -> pd.DataFrame:
    frame = pd.DataFrame({"beamlet": label, "depth_mm": profile.depth_mm, "hu_axis": profile.hu_axis,
                          "hu_flux_weighted": profile.hu_flux_weighted, "rsp_axis": profile.rsp_axis,
                          "wepl_axis_mm": profile.wepl_axis_mm, "idd_analytic": profile.idd_analytic})
    if profile.idd_mc is not None:
        frame["idd_mc"] = profile.idd_mc
    return frame


def make_figures(out: Path, sources: Mapping, refs: Sequence[BeamletRef], pairs: Sequence[Tuple[str, str]],
                 energies: Mapping[str, float], grouped: pd.DataFrame, by_group: pd.DataFrame,
                 totals: pd.DataFrame, groups: FeatureGroups, members: Sequence[str],
                 titles: Mapping[str, str], short_titles: Mapping[str, str]) -> Dict[str, List[Path]]:
    """Every figure of the run into ``out / "figures"``, a CSV of its numbers beside it.

    ``titles`` head the beam-path figures; ``short_titles`` head the columns of the contribution grid.
    """
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [r.label for r in refs]
    scorers = [ENSEMBLE] + list(members)
    colors = group_colors(sorted(set(groups.names.values())))
    written: Dict[str, List[Path]] = {}

    contrib = grouped[grouped["scorer"].isin(scorers)]
    contrib.to_csv(fig_dir / "contributions_numbers.csv", index=False)
    written["contributions"] = contribution_figure(
        contrib, totals[totals["scorer"].isin(scorers)], examples=labels, example_labels=short_titles, scorers=scorers,
        colors=colors, figure_path=str(fig_dir / "contributions_by_group"),
        title="Difficulty score minus the pool-test mean, by feature group")

    contrast = by_group[by_group["scorer"].isin(scorers)]
    contrast.to_csv(fig_dir / "contrast_numbers.csv", index=False)
    written["contrast"] = contrast_figure(
        contrast, pairs=pairs, pair_labels={p: f"{p[0]} vs {p[1]}" for p in pairs}, colors=colors,
        figure_path=str(fig_dir / "contrast_by_group"),
        title="Why the harder beamlet scores higher: ensemble (bars) and members (markers)")

    for (h, e), sub in contrast.groupby(["harder", "easier"], sort=False):
        mat = sub.pivot(index="item", columns="scorer", values="difference")[scorers]
        mat = mat.loc[mat[ENSEMBLE].sort_values(ascending=False).index]
        mat.to_csv(fig_dir / f"agreement_{h}_vs_{e}_numbers.csv")
        written[f"agreement_{h}_{e}"] = annotated_heatmap(
            mat, figure_path=str(fig_dir / f"agreement_{h}_vs_{e}"), value_label="Contribution difference",
            title=f"{h} minus {e}: group contribution difference per scorer",
            caption="Rows: feature groups. Columns: the deployed ensemble and the five member ridges it averages.")

    order = sorted(groups.names, key=lambda f: (groups.names[f], f))
    corr = groups.clusters.correlation.loc[order, order]
    corr.to_csv(fig_dir / "feature_correlation_numbers.csv")
    written["correlation"] = annotated_heatmap(
        corr, figure_path=str(fig_dir / "feature_correlation"), value_label="Spearman rho", fmt="{:.1f}",
        groups=[str(groups.clusters.assignment[f]) for f in order], vmax=1.0, cell_in=0.34,
        title=f"Feature correlation on the reference; groups at |rho| >= {groups.threshold:g}",
        caption="Row prefix: group id (tables/groups.csv).")

    profiles, marks_rows = [], []
    for ref in refs:
        inputs = load_inputs(ref, sources, energies[ref.label])
        profile = beam_axis_profile(inputs)
        marks, mark_table = depth_marks(inputs, profile)
        profiles.append(_profile_frame(ref.label, profile))
        marks_rows.append(mark_table.assign(beamlet=ref.label, analytic_bp_slice=profile.bp_slice,
                                            mc_bp_slice=profile.mc_bp_slice,
                                            bp_zone_min_mm=profile.bp_zone_mm[0], bp_zone_max_mm=profile.bp_zone_mm[1]))
        written[f"beam_path_{ref.label}"] = beam_path_figure(
            inputs, profile, title=titles[ref.label], marks=marks, figure_path=str(fig_dir / f"beam_path_{ref.label}"))
    pd.concat(profiles, ignore_index=True).to_csv(fig_dir / "beam_path_profiles_numbers.csv", index=False)
    pd.concat(marks_rows, ignore_index=True).to_csv(fig_dir / "beam_path_marks_numbers.csv", index=False)
    return written


__all__ = ["depth_marks", "load_inputs", "make_figures"]
