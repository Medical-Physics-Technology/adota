"""The difficulty-explanation glue: beamlet ids, member agreement, ordinals."""
from __future__ import annotations

import pandas as pd
import pytest

from src.acquisition.explain.analysis import ENSEMBLE, BeamletRef, _fmt_pct, agreement


def test_beamlet_ref_parses_grid_and_pool_and_rejects_the_rest():
    g = BeamletRef.parse("easy", "grid:block_a/a08_00")
    assert (g.source, g.block, g.stem) == ("grid", "block_a", "a08_00")
    p = BeamletRef.parse("x", "pool:c0abc")
    assert (p.source, p.key) == ("pool", "c0abc")
    for bad in ("block_a/a08_00", "grid:block_a", "h5:abc", "pool:"):
        with pytest.raises(ValueError):
            BeamletRef.parse("bad", bad)


def _contrast(scorer: str, diffs: dict) -> pd.DataFrame:
    order = sorted(diffs, key=diffs.get, reverse=True)
    return pd.DataFrame([{"scorer": scorer, "item": g, "harder": "h", "easier": "e", "difference": diffs[g],
                          "rank": order.index(g) + 1, "score_gap": sum(diffs.values())} for g in diffs])


def test_agreement_counts_shared_top_groups_and_treats_missing_groups_as_zero():
    frame = pd.concat([_contrast(ENSEMBLE, {"a": 0.5, "b": 0.3, "c": -0.1}),
                       _contrast("m1", {"a": 0.6, "b": 0.2, "c": -0.2}),
                       _contrast("m2", {"b": 0.4, "c": 0.3, "a": 0.1}),
                       _contrast("sparse", {"a": 0.7})], ignore_index=True)
    out = agreement(frame, members=["m1", "m2"], top_k=2).set_index("scorer")
    assert out.loc["m1", "same_first_group"] and out.loc["m1", "overlap_with_ensemble_top"] == 2
    assert not out.loc["m2", "same_first_group"] and out.loc["m2", "overlap_with_ensemble_top"] == 1
    assert out.loc["m1", "spearman_vs_ensemble"] == pytest.approx(1.0)
    assert not out.loc["sparse", "is_member"] and out.loc["sparse", "same_first_group"]


@pytest.mark.parametrize("p, text", [(92.0, "92nd"), (8.4, "8th"), (41.0, "41st"), (13.0, "13th"),
                                     (3.0, "3rd"), (float("nan"), "n/a")])
def test_percentile_ordinals(p, text):
    assert _fmt_pct(p) == text
