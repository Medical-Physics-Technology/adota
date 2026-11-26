"""Statistics behind the CT-feature-versus-error figures (synthetic, CPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine_drivers.ct_vs_error import (
    ERRORS,
    _bins,
    correlations,
    join_errors,
    partial_spearman,
    verdict,
)


def _points(n_pat: int = 60, per: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_pat * per
    energy = rng.uniform(70, 200, n)
    confounded = energy + rng.normal(0, 5, n)          # tracks energy only
    own = rng.normal(0, 1, n)                           # independent of energy
    site = np.array(["thoracic", "abdominal", "pelvic"])[np.repeat(np.arange(n_pat) % 3, per)]
    mape = np.expm1(0.01 * energy + 0.5 * own + rng.normal(0, 0.1, n))
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "patient_id": np.repeat(np.arange(n_pat), per),
                         "site": site, "energy_mev": energy, "confounded": confounded, "own": own,
                         "mape_pct": mape, "gpr_2pct_2mm_10pct": 100 - mape})


def test_join_errors_refuses_lost_rows():
    table = pd.DataFrame({"sample_id": ["a", "b", "c"], "x": [1, 2, 3]})
    errors = pd.DataFrame({"sample_id": ["a", "b", "c"], "engine": "e", "mape_pct": [1.0, 2.0, 3.0]})
    assert len(join_errors(table, errors)) == 3
    with pytest.raises(ValueError, match="lost rows"):
        join_errors(table, errors.iloc[:2])


def test_tie_aware_bins_keep_ties_together():
    values = pd.Series([100.0] * 50 + list(np.linspace(90, 99, 50)))
    b = _bins(values, 10)
    assert b[values == 100.0].nunique() == 1
    assert b.max() < 10


def test_partial_spearman_removes_a_pure_energy_relation_and_keeps_an_independent_one():
    p = _points()
    out = partial_spearman(p, ["confounded", "own", "energy_mev"], ERRORS["mape_pct"], "energy_mev",
                           p["patient_id"].to_numpy(), n_boot=100, seed=1).set_index("feature")
    assert abs(out.loc["confounded", "spearman"]) < 0.15
    assert out.loc["own", "spearman"] > 0.5
    assert np.isnan(out.loc["energy_mev", "spearman"])


def test_correlations_orient_gpr_as_failure_and_cover_every_scope():
    p = _points()
    st = correlations(p, ["own"], "gpr_2pct_2mm_10pct", n_boot=100, seed=1)
    assert set(st["scope"]) == {"pooled", "thoracic", "abdominal", "pelvic", "energy_partial"}
    pooled = st[st["scope"] == "pooled"].iloc[0]
    assert pooled["spearman"] > 0.3      # GPR = 100 - MAPE, so more failure with larger 'own'


def test_verdict_says_disappears_and_between_sites():
    p = _points()
    st = correlations(p, ["confounded", "own"], "mape_pct", n_boot=100, seed=1)
    text, flags = verdict(st, "mape_pct", "confounded")
    assert flags["pooled"] and flags["energy_partial"] is False
    assert "disappears after controlling for beam energy" in text
    fake = pd.DataFrame([{"error": "mape_pct", "feature": "f", "scope": s, "spearman": v, "ci_low": lo,
                          "ci_high": hi, "ci_excludes_zero": lo > 0 or hi < 0}
                         for s, v, lo, hi in (("pooled", 0.3, 0.2, 0.4), ("thoracic", 0.1, -0.1, 0.3),
                                              ("abdominal", 0.1, -0.1, 0.3), ("pelvic", 0.0, -0.2, 0.2),
                                              ("energy_partial", 0.3, 0.2, 0.4))])
    text, _ = verdict(fake, "mape_pct", "f")
    assert "differences between sites" in text


def _fake(rows):
    return pd.DataFrame([{"error": "mape_pct", "feature": "f", "scope": s, "spearman": v, "ci_low": lo,
                          "ci_high": hi, "ci_excludes_zero": lo > 0 or hi < 0} for s, v, lo, hi in rows])


def test_verdict_names_a_site_that_runs_the_other_way():
    text, _ = verdict(_fake([("pooled", -0.15, -0.2, -0.1), ("thoracic", -0.2, -0.3, -0.1),
                             ("abdominal", 0.3, 0.2, 0.4), ("pelvic", 0.0, -0.1, 0.1),
                             ("energy_partial", -0.15, -0.2, -0.1)]), "mape_pct", "f")
    assert "it holds within thoracic (-0.20)" in text
    assert "within abdominal (+0.30) it runs the other way" in text
    assert "holds within abdominal" not in text


def test_verdict_says_sites_cancel_when_pooled_is_null():
    text, _ = verdict(_fake([("pooled", 0.05, -0.01, 0.11), ("thoracic", 0.23, 0.17, 0.28),
                             ("abdominal", -0.3, -0.4, -0.2), ("pelvic", 0.25, 0.13, 0.34),
                             ("energy_partial", 0.6, 0.5, 0.7)]), "mape_pct", "f")
    assert "the sites disagree" in text and "cancel when pooled" in text
