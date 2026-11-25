"""The difficulty-explanation pipeline: which features make one beamlet harder than another.

Inputs are named beamlets, ``grid:<block>/<stem>`` (a saved EXP-0021 grid beamlet)
or ``pool:<sample_id>`` (a beamlet-pool HDF5 record). For each, the analysis

1. places every feature against the reference population (percentile, robust z);
2. decomposes the score of the deployed ensemble, of the five per-member ridges
   behind it and of the sparse (Lasso) variant into exact contributions;
3. sums contributions within groups of correlated features;
4. ranks groups and features by their difference between pairs of beamlets;
5. checks whether the members give the same reasons as the ensemble.

The script is ``scripts/active_learning/difficulty_explain.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.acquisition.explain.contributions import (
    Decomposition,
    decompose,
    group_contributions,
    pair_contrast,
)
from src.acquisition.explain.groups import FeatureGroups
from src.active_learning.ensemble.independence import inside_crop_features

ENSEMBLE, SPARSE_LABEL = "ensemble", "sparse"
GRID_ENGINES: Mapping[str, str] = {"v11": "v11", "final_attndrop0": "EXP-0019", "x0_mlstm_004": "X0 (xLSTM)",
                                   "u0_none_029": "U0 (U-Net)", "pba": "pencil beam"}
"""Engine tags of the EXP-0021 per-beamlet table and their display names."""


@dataclass(frozen=True)
class BeamletRef:
    """A named beamlet: ``source`` is ``grid`` or ``pool``; ``key`` is ``block/stem`` or a sample id."""

    label: str
    source: str
    key: str

    @classmethod
    def parse(cls, label: str, spec: str) -> "BeamletRef":
        source, _, key = spec.partition(":")
        if source not in ("grid", "pool") or not key:
            raise ValueError(f"beamlet {label!r}: expected 'grid:<block>/<stem>' or 'pool:<sample_id>', got {spec!r}")
        if source == "grid" and "/" not in key:
            raise ValueError(f"beamlet {label!r}: a grid beamlet is 'grid:<block>/<stem>', got {spec!r}")
        return cls(label, source, key)

    @property
    def block(self) -> str:
        return self.key.split("/")[0]

    @property
    def stem(self) -> str:
        return self.key.split("/")[1]


def load_reference(path: str) -> pd.DataFrame:
    """The reference population: analytic features with the peak inside the crop, by sample id."""
    return inside_crop_features(path)


def example_features(refs: Sequence[BeamletRef], grid_features: str, pool_features: str) -> pd.DataFrame:
    """The saved feature rows of the named beamlets, indexed by label, with ``peak_inside_crop``."""
    grid = pool = None
    rows = []
    for ref in refs:
        if ref.source == "grid":
            grid = pd.read_csv(grid_features) if grid is None else grid
            hit = grid[(grid["block"] == ref.block) & (grid["stem"] == ref.stem)]
        else:
            pool = pd.read_csv(pool_features) if pool is None else pool
            hit = pool[pool["sample_id"].astype(str) == ref.key]
        if len(hit) != 1:
            raise KeyError(f"{ref.label}: {len(hit)} feature rows for {ref.source}:{ref.key}")
        row = hit.iloc[0].copy()
        if not bool(row["peak_inside_crop"]):
            raise ValueError(f"{ref.label}: the analytic peak is outside the crop, the score is not defined there")
        row.name = ref.label
        rows.append(row)
    return pd.DataFrame(rows)


def example_errors(refs: Sequence[BeamletRef], grid_per_beamlet: str, pool_members_dir: str) -> pd.DataFrame:
    """Every stored engine error of the named beamlets, long: label, engine, metric, value.

    Grid beamlets carry gamma pass rates at 2%/2mm and 3%/3mm (10% cutoff, global) and MAPE on
    the 1 mm crop; pool beamlets carry the 2%/2mm pass rate, MAPE and RDE of every engine.
    """
    rows: List[Dict] = []
    grid = None
    for ref in refs:
        if ref.source == "grid":
            grid = pd.read_csv(grid_per_beamlet) if grid is None else grid
            hit = grid[(grid["block"] == ref.block) & (grid["stem"] == ref.stem)].iloc[0]
            for tag, name in GRID_ENGINES.items():
                for col, metric in ((f"{tag}_g2_2_10", "gpr_2pct_2mm_pct"), (f"{tag}_g3_3_10", "gpr_3pct_3mm_pct"),
                                    (f"{tag}_mape_1mm", "mape_pct")):
                    if col in hit.index:
                        rows.append({"label": ref.label, "engine": name, "metric": metric, "value": float(hit[col])})
        else:
            for path in sorted(Path(pool_members_dir).glob("*_errors.csv")):
                frame = pd.read_csv(path).astype({"sample_id": str}).set_index("sample_id")
                if ref.key not in frame.index:
                    continue
                r = frame.loc[ref.key]
                engine = path.name[:-len("_errors.csv")]
                rows += [{"label": ref.label, "engine": engine, "metric": "gpr_2pct_2mm_pct",
                          "value": 100.0 * float(r["gpr"])},
                         {"label": ref.label, "engine": engine, "metric": "mape_pct", "value": float(r["mape_pct"])},
                         {"label": ref.label, "engine": engine, "metric": "rde_pct", "value": float(r["rde_pct"])}]
    return pd.DataFrame(rows)


def decompose_all(scorers: Mapping[str, object], examples: pd.DataFrame,
                  reference: pd.DataFrame) -> Dict[str, Decomposition]:
    """Decompose every scorer (``{label: DifficultyScorer}``) on the examples; each asserts its sum check."""
    return {name: decompose(s, examples, reference, name) for name, s in scorers.items()}


def contrasts(decs: Mapping[str, Decomposition], groups: FeatureGroups, pairs: Sequence[Tuple[str, str]],
              top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group contributions of every scorer, and the group- and feature-level contrasts of every pair."""
    table = pd.concat([d.table for d in decs.values()], ignore_index=True)
    grouped = group_contributions(table, groups.names)
    by_group = pd.concat([pair_contrast(grouped, h, e, "group", top_k) for h, e in pairs], ignore_index=True)
    by_feature = pd.concat([pair_contrast(table, h, e, "feature", top_k) for h, e in pairs], ignore_index=True)
    return grouped, by_group, by_feature


def agreement(by_group: pd.DataFrame, members: Sequence[str], top_k: int,
              reference: str = ENSEMBLE) -> pd.DataFrame:
    """Per pair and scorer: does it give the same reasons as ``reference``?

    Columns: the scorer's top-k groups, how many of them are in the reference's
    top-k, whether its first group is the reference's first, and the Spearman
    correlation of its per-group differences with the reference's.
    """
    rows = []
    for (h, e), sub in by_group.groupby(["harder", "easier"], sort=False):
        ref = sub[sub["scorer"] == reference].set_index("item")
        ref_top = list(ref.sort_values("rank").index[:top_k])
        for scorer in [s for s in sub["scorer"].unique() if s != reference]:
            mine = sub[sub["scorer"] == scorer].set_index("item")
            top = list(mine.sort_values("rank").index[:top_k])
            # A sparse scorer has no feature in some groups: their contribution is exactly zero.
            rho = spearmanr(mine["difference"].reindex(ref.index, fill_value=0.0), ref["difference"]).statistic
            rows.append({"harder": h, "easier": e, "scorer": scorer, "is_member": scorer in members,
                         "score_gap": float(mine["score_gap"].iloc[0]), "top_groups": " | ".join(top),
                         "overlap_with_ensemble_top": len(set(top) & set(ref_top)), "top_k": top_k,
                         "same_first_group": top[0] == ref_top[0], "spearman_vs_ensemble": float(rho)})
    return pd.DataFrame(rows)


def _fmt_pct(p: float) -> str:
    """``92nd``, ``8th``, ``41st``: a rounded percentile as an English ordinal."""
    if np.isnan(p):
        return "n/a"
    n = int(round(p))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def draft_sentence(h: str, e: str, by_group: pd.DataFrame, by_feature: pd.DataFrame, groups: FeatureGroups,
                   placement: pd.DataFrame, score_pct: Mapping[str, float], agree: pd.DataFrame,
                   n_groups: int = 2) -> str:
    """One data-driven sentence for a pair: the leading groups, the feature that drives each, the
    feature's percentile in both beamlets, what offsets them, and the members' agreement. The physical
    reading is added by hand in the summary; this sentence carries only numbers."""
    ens = by_group[(by_group["scorer"] == ENSEMBLE) & (by_group["harder"] == h) & (by_group["easier"] == e)]
    ens = ens.sort_values("rank")
    gap = float(ens["score_gap"].iloc[0])
    feats = by_feature[(by_feature["scorer"] == ENSEMBLE) & (by_feature["harder"] == h) & (by_feature["easier"] == e)]
    pct = placement.set_index(["beamlet_id", "feature"])["percentile_ref"]
    parts = []
    for g in ens.head(n_groups).itertuples():
        members = [f for f, name in groups.names.items() if name == g.item]
        lead = feats[feats["item"].isin(members)].sort_values("difference", ascending=False).iloc[0]
        parts.append(f"{g.item} ({g.difference:+.2f}, {100 * g.share_of_gap:.0f}% of the gap; led by "
                     f"{lead['item']} at the {_fmt_pct(pct[(h, lead['item'])])} vs "
                     f"{_fmt_pct(pct[(e, lead['item'])])} percentile)")
    offset = ens[ens["difference"] < 0].sort_values("difference").head(1)
    tail = "" if offset.empty else (f", partly offset by {offset['item'].iloc[0]} "
                                    f"({offset['difference'].iloc[0]:+.2f})")
    a = agree[(agree["harder"] == h) & (agree["easier"] == e) & agree["is_member"]]
    agree_txt = (f" {int(a['same_first_group'].sum())} of {len(a)} members share the first group; median "
                 f"Spearman of member vs ensemble group differences {a['spearman_vs_ensemble'].median():.2f}.")
    return (f"{h} scores higher than {e} (gap {gap:+.2f}; reference percentile {score_pct[h]:.0f} vs "
            f"{score_pct[e]:.0f}) mainly because of " + " and ".join(parts) + tail + "." + agree_txt)


__all__ = ["BeamletRef", "ENSEMBLE", "GRID_ENGINES", "SPARSE_LABEL", "agreement", "contrasts", "decompose_all",
           "draft_sentence", "example_errors", "example_features", "load_reference"]
