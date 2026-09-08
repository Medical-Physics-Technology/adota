"""Unit tests for the active-learning loop's decision logic.

Everything here runs on CPU with a synthetic candidate table: no dataset, no
checkpoint, no MCsquare. The parts that need those (candidate scoring on a real CT,
Monte Carlo labelling, a training cycle) are exercised by the loop's own dry-run and
smoke paths, not here.

The properties worth pinning are the ones a silent regression would make invisible in
a run: that a strategy spends exactly its budget, that the per-patient cap and the
per-energy quota actually bind, that invalid candidates never reach the oracle, that
candidate ids are stable across processes (the loop resumes on them), and that the
oversampling sampler delivers the share of new beamlets it promises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.active_learning.candidates import (
    CandidateConfig,
    _stable_seed,
    candidate_id,
    draw_gantries,
    field_dir_name,
    generate_candidates,
)
from src.active_learning.dataset import union_sampler, write_training_sources
from src.active_learning.oracle import batch_cost_estimate, group_sizes
from src.active_learning.pool import PoolEntry, read_pool, write_pool
from src.active_learning.sampling import STRATEGIES, select, selection_fingerprint
from src.active_learning.validation import select_balanced, summarise


def make_table(n_patients: int = 6, n_energies: int = 4, per_cell: int = 40,
               n_invalid: int = 25, seed: int = 0) -> pd.DataFrame:
    """A synthetic scored candidate table with the columns the loop relies on."""
    rng = np.random.default_rng(seed)
    rows = []
    energies = [80.0 + 25.0 * i for i in range(n_energies)]
    for p in range(n_patients):
        for e in energies:
            for k in range(per_cell):
                rows.append({
                    "candidate_id": f"c{p:02d}_{e:g}_{k:03d}",
                    "patient_id": f"P{p:02d}",
                    "anatomy": "thoracic" if p % 2 else "abdominal",
                    "gantry_deg": float(30 * (k % 2)),
                    "energy_mev": e,
                    "theta_x_deg": float(rng.uniform(-1.5, 1.5)),
                    "theta_y_deg": float(rng.uniform(-1.5, 1.5)),
                    "field_dir": f"al_P{p:02d}_e{e:g}",
                    "valid": True,
                    "score_full": float(rng.uniform(0, 1)),
                })
    table = pd.DataFrame(rows)
    table.loc[table.index[:n_invalid], "valid"] = False
    table.loc[table.index[:n_invalid], "score_full"] = np.nan
    return table


# ── Sampling ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_spends_exactly_its_budget(strategy):
    table = make_table()
    chosen = select(table, strategy, 200, rng=np.random.default_rng(1))
    assert len(chosen) == 200
    assert chosen["candidate_id"].is_unique


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_invalid_candidates_are_never_selected(strategy):
    table = make_table()
    invalid = set(table.loc[~table["valid"].astype(bool), "candidate_id"])
    chosen = select(table, strategy, 300, rng=np.random.default_rng(2))
    assert not (set(chosen["candidate_id"]) & invalid)


def test_per_energy_quota_spreads_the_budget_over_the_layers():
    table = make_table(n_energies=4)
    chosen = select(table, "score", 200, rng=np.random.default_rng(3))
    counts = chosen["energy_mev"].value_counts()
    assert len(counts) == 4
    assert counts.max() - counts.min() <= 1


def test_score_topk_collapses_onto_the_deepest_layers():
    """The ablation is meant to show the collapse the quotas prevent."""
    table = make_table(n_energies=4)
    table.loc[table["valid"], "score_full"] = (
        table.loc[table["valid"], "energy_mev"] / 200.0)
    quota = select(table, "score", 200, rng=np.random.default_rng(4))
    topk = select(table, "score_topk", 200, rng=np.random.default_rng(4))
    assert topk["energy_mev"].nunique() < quota["energy_mev"].nunique()


def test_per_patient_cap_binds():
    table = make_table(n_patients=6)
    chosen = select(table, "score", 240, rng=np.random.default_rng(5),
                    max_per_patient_frac=0.25)
    per_layer_budget = 240 / table.loc[table["valid"], "energy_mev"].nunique()
    cap = int(np.ceil(per_layer_budget * 0.25))
    worst = chosen.groupby(["energy_mev", "patient_id"]).size().max()
    assert worst <= cap


def test_score_strategy_prefers_high_scores_over_random():
    table = make_table()
    scored = select(table, "score", 200, rng=np.random.default_rng(6))
    uniform = select(table, "random", 200, rng=np.random.default_rng(6))
    assert scored["score_full"].mean() > uniform["score_full"].mean()


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        select(make_table(), "greedy", 10, rng=np.random.default_rng(0))


def test_budget_larger_than_the_pool_is_an_error_not_a_short_batch():
    table = make_table(n_patients=1, n_energies=1, per_cell=10, n_invalid=0)
    with pytest.raises(ValueError, match="valid candidates"):
        select(table, "random", 50, rng=np.random.default_rng(0))


def test_selection_fingerprint_reports_what_was_bought():
    chosen = select(make_table(), "score", 100, rng=np.random.default_rng(7))
    print_out = selection_fingerprint(chosen)
    assert print_out["n"] == 100
    assert sum(print_out["per_energy"].values()) == 100
    assert sum(print_out["per_anatomy"].values()) == 100


# ── Candidate identity and generation ───────────────────────────────────────


def test_candidate_ids_are_stable_and_parameter_specific():
    a = candidate_id("P1", 30.0, 120.0, 0.5, -0.5)
    assert a == candidate_id("P1", 30.0, 120.0, 0.5, -0.5)
    assert a != candidate_id("P1", 30.0, 120.0, 0.5, 0.5)
    assert a != candidate_id("P2", 30.0, 120.0, 0.5, -0.5)
    assert a.isalnum()


def test_seeds_do_not_depend_on_the_interpreter_hash_salt():
    """Gantries are drawn inside worker processes; ``hash()`` would differ there."""
    assert _stable_seed(1, "patient", "gantry") == _stable_seed(1, "patient", "gantry")
    assert _stable_seed(1, "patient", "gantry") != _stable_seed(2, "patient", "gantry")


def test_generated_candidates_sit_on_the_lattice_and_in_the_window():
    cfg = CandidateConfig(n_gantry_per_ct=3, n_per_field=20)
    candidates = generate_candidates("P1", "uid-1", cfg)
    assert len(candidates) == 60
    lattice = np.linspace(-cfg.theta_half_range_deg, cfg.theta_half_range_deg,
                          cfg.lattice_n)
    for candidate in candidates:
        assert np.isclose(lattice, candidate.theta_x_deg).any()
        assert np.isclose(lattice, candidate.theta_y_deg).any()
        assert candidate.energy_mev in cfg.energies
    assert len({c.gantry_deg for c in candidates}) == 3


def test_the_same_patient_always_draws_the_same_gantries():
    cfg = CandidateConfig(n_gantry_per_ct=4)
    assert draw_gantries(cfg, "uid-1") == draw_gantries(cfg, "uid-1")
    assert draw_gantries(cfg, "uid-1") != draw_gantries(cfg, "uid-2")


def test_field_dir_name_is_filename_safe():
    name = field_dir_name("thoracic", "Lung_Dx-G0035", 102.6, 247.34)
    assert name == "al_thoracic_Lung_Dx-G0035_e102p6_g247p3_v1"


# ── Validation set ──────────────────────────────────────────────────────────


def test_balanced_validation_set_is_spread_over_the_deciles():
    table = make_table(n_patients=8, n_energies=4, per_cell=60, n_invalid=0)
    chosen = select_balanced(table, 400, rng=np.random.default_rng(11))
    assert len(chosen) == 400
    counts = chosen["score_decile"].value_counts()
    assert len(counts) == 10
    # No decile may dominate: a balanced set is the point of the recipe.
    assert counts.max() <= 3 * counts.min()


def test_balanced_validation_set_covers_every_anatomy_and_energy():
    table = make_table(n_patients=8, n_energies=4, per_cell=60, n_invalid=0)
    chosen = select_balanced(table, 400, rng=np.random.default_rng(12))
    assert chosen["anatomy"].nunique() == 2
    assert chosen["energy_mev"].nunique() == 4


def test_summarise_reports_the_tail_and_ignores_failed_gammas():
    frame = pd.DataFrame({
        "gpr": [1.0, 0.99, 0.90, np.nan],
        "mape_pct": [3.0, 4.0, 9.0, 5.0],
        "rde_pct": [0.1, 0.2, 0.4, 0.2],
        "dr80_mm": [0.5, -0.6, 2.0, np.nan],
    })
    out = summarise(frame)
    assert out["n"] == 4 and out["n_gpr"] == 3
    assert out["gpr_frac_below_95"] == pytest.approx(1 / 3)
    assert out["n_dr80"] == 3
    assert out["abs_dr80_median_mm"] == pytest.approx(0.6)


# ── Union sampling ──────────────────────────────────────────────────────────


def test_union_sampler_delivers_the_promised_share_of_new_beamlets():
    n_h5, n_dir, fraction = 5000, 200, 0.25
    generator = torch.Generator().manual_seed(0)
    sampler = union_sampler(n_h5, n_dir, fraction, steps=200, batch_size=56,
                            generator=generator)
    drawn = np.array(list(sampler))
    assert len(drawn) == 200 * 56
    share = float(np.mean(drawn >= n_h5))
    assert share == pytest.approx(fraction, abs=0.02)


def test_union_sampler_refuses_an_empty_new_set():
    with pytest.raises(ValueError, match="at least one new beamlet"):
        union_sampler(100, 0, 0.25, steps=2, batch_size=4, generator=torch.Generator())


# ── Bookkeeping ─────────────────────────────────────────────────────────────


def test_training_sources_round_trip_and_deduplicate(tmp_path):
    from src.active_learning.dataset import read_training_sources

    path = tmp_path / "sources.csv"
    write_training_sources([("/a", "c1"), ("/a", "c2"), ("/a", "c1")], path)
    assert read_training_sources(path) == [("/a", "c1"), ("/a", "c2")]


def test_pool_csv_round_trip_and_role_filter(tmp_path):
    entries = [
        PoolEntry("validation", "thoracic", "D", "/root", "C", "P1", "uid1", 200),
        PoolEntry("pool", "thoracic", "D", "/root", "C", "P2", "uid2", 210),
    ]
    path = tmp_path / "pool.csv"
    write_pool(entries, path)
    assert read_pool(path) == entries
    assert [e.patient_id for e in read_pool(path, role="pool")] == ["P2"]


def test_cost_estimate_counts_groups_not_just_beamlets():
    chosen = select(make_table(), "random", 240, rng=np.random.default_rng(13))
    cost = batch_cost_estimate(chosen)
    sizes = group_sizes(chosen)
    assert cost["n_beamlets"] == 240
    assert cost["n_groups"] == sizes["n_groups"] > 0
    assert cost["estimated_setup_seconds"] > 0
    assert cost["estimated_total_hours"] > 0
