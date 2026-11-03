"""The frozen validation set: the score-free stratified draw, the patient mask that
leaves out a couch, and the HU histogram summaries of the statistics script."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from src.active_learning.validation import _water_fill, select_stratified
from src.active_learning.validation_stats import bin_index, histogram_summary, hu_edges, summarise_histograms
from src.mc_generation.geometry import body_mask
from tests.test_active_learning import make_table

STRATA = ["anatomy", "energy_mev", "patient_id", "gantry_deg"]


def test_water_fill_is_even_and_passes_shortfalls_on():
    assert _water_fill([10, 10, 10], 9) == [3, 3, 3]
    assert _water_fill([1, 10, 10], 9) == [1, 4, 4]
    assert sum(_water_fill([5, 7, 2], 11)) == 11
    assert _water_fill([2, 2], 10) == [2, 2]


def test_stratified_draw_balances_every_level_and_ignores_the_score():
    table = make_table(n_patients=6, n_energies=4, per_cell=40, n_invalid=0)
    chosen = select_stratified(table, 480, STRATA, rng=np.random.default_rng(1))
    assert len(chosen) == 480 and chosen["candidate_id"].is_unique
    assert chosen.groupby("anatomy").size().tolist() == [240, 240]
    assert set(chosen.groupby(["anatomy", "energy_mev"]).size()) == {60}
    assert set(chosen.groupby(["patient_id", "energy_mev"]).size()) == {20}
    # The same draw whatever the scores are.
    shuffled = table.assign(score_full=np.random.default_rng(7).uniform(size=len(table)))
    again = select_stratified(shuffled, 480, STRATA, rng=np.random.default_rng(1))
    assert set(again["candidate_id"]) == set(chosen["candidate_id"])


def test_stratified_draw_keeps_the_anatomy_balance_when_one_energy_is_short():
    table = make_table(n_patients=6, n_energies=4, per_cell=40, n_invalid=0)
    short = (table["anatomy"] == "thoracic") & (table["energy_mev"] == table["energy_mev"].max())
    table.loc[short & (table.groupby(["patient_id", "energy_mev"]).cumcount() >= 2), "valid"] = False
    chosen = select_stratified(table, 480, STRATA, rng=np.random.default_rng(2))
    by = chosen.groupby(["anatomy", "energy_mev"]).size()
    assert chosen.groupby("anatomy").size().tolist() == [240, 240]
    assert by[("thoracic", table["energy_mev"].max())] == 6          # all it had
    assert chosen["valid"].all()


def test_stratified_draw_refuses_more_than_the_valid_candidates():
    with pytest.raises(ValueError):
        select_stratified(make_table(n_patients=2, n_energies=1, per_cell=5), 1000, STRATA,
                          rng=np.random.default_rng(0))


def phantom_with_couch() -> sitk.Image:
    """Air everywhere, a water cylinder with a lung-like hole, and a separate slab below."""
    arr = np.full((6, 40, 40), -1000.0, dtype=np.float32)
    yy, xx = np.mgrid[:40, :40]
    arr[:, (yy - 18) ** 2 + (xx - 20) ** 2 < 12 ** 2] = 0.0
    arr[:, (yy - 18) ** 2 + (xx - 20) ** 2 < 4 ** 2] = -850.0          # interior lung
    arr[:, 35:37, 5:35] = 300.0                                        # the couch
    return sitk.GetImageFromArray(arr)


def test_largest_component_drops_the_couch_and_keeps_the_lung():
    ct = phantom_with_couch()
    every = body_mask(ct) > 0.5
    patient = body_mask(ct, largest_component=True) > 0.5
    assert every[:, 36, 20].all() and not patient[:, 36, 20].any()     # couch: in, then out
    assert patient[:, 18, 20].all()                                    # lung stays inside
    assert patient.sum() == every.sum() - 6 * 2 * 30


def test_histogram_summary_reads_percentiles_and_tissue_fractions():
    edges = hu_edges(10.0)
    values = np.concatenate([np.full(300, -850.0), np.full(700, 40.0)])
    counts = np.bincount(bin_index(values, edges), minlength=len(edges) - 1)
    summary = histogram_summary(counts, edges)
    assert summary["frac_lung"] == pytest.approx(0.3)
    assert summary["frac_soft_tissue"] == pytest.approx(0.7)
    assert summary["p50_hu"] == pytest.approx(45.0)                    # bin centre of 40 HU
    frame = pd.DataFrame({"anatomy": "a", "bin_lo": edges[:-1], "weight": counts})
    assert summarise_histograms(frame, ["anatomy"], edges)["frac_lung"].iloc[0] == pytest.approx(0.3)


def test_values_outside_the_scale_land_in_the_end_bins():
    edges = hu_edges(10.0)
    assert bin_index(np.array([-3000.0, 5000.0]), edges).tolist() == [0, len(edges) - 2]
