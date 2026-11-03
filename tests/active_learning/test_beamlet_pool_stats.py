"""Reporting statistics of the beamlet pool: tables from the index, figures render."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.active_learning.beamlet_pool.stats import (
    anatomy_tables,
    concentration,
    gantry_frame,
    gantry_tables,
    gini,
    load_pool_index,
    patient_tables,
)
from src.figures.beamlet_pool import anatomy_figure, gantry_frame_figure, patient_figure

LATTICE = np.linspace(-1.5, 1.5, 4)


def write_store(root, n_per_patient=12, seed=0):
    """Two batches, three anatomies, one patient per split each; a few beamlets lost."""
    rng = np.random.default_rng(seed)
    rows, plan = [], []
    patients = [(f"{a[:3]}-{s}", a, s, c) for a, c in [("thoracic", "Lung"), ("abdominal", "Colo"),
                                                       ("pelvic", "Pros")]
                for s in ("train", "validation", "test")]
    for k, (pid, anatomy, split, collection) in enumerate(patients):
        gantries = rng.uniform(0, 360, 2)
        for i in range(n_per_patient):
            cid = f"c{k:02d}{i:03d}"
            status = "stored" if i % 6 else ("missing" if i % 12 == 0 else "rejected:mc_overshoot_crop")
            rows.append({"candidate_id": cid, "batch": k % 2, "split": split, "patient_id": pid,
                         "anatomy": anatomy, "energy_mev": float(rng.uniform(70, 200)),
                         "gantry_deg": float(gantries[i % 2]), "theta_x_deg": float(rng.choice(LATTICE)),
                         "theta_y_deg": float(rng.choice(LATTICE)), "r80_mm": float(rng.uniform(50, 300)),
                         "dose_in_body_fraction": 0.9, "status": status})
            plan.append({"candidate_id": cid, "dataset_name": collection})
    index = pd.DataFrame(rows)
    (root / "index").mkdir(parents=True)
    (root / "plan").mkdir()
    for b, part in index.groupby("batch"):
        part.to_csv(root / "index" / f"batch_{b:04d}.csv", index=False)
    pd.DataFrame(plan).to_csv(root / "plan" / "plan.csv", index=False)
    registry = pd.DataFrame({"patient_id": [p for p, *_ in patients] + ["Excluded-1"],
                             "split": [s for _, _, s, _ in patients] + ["excluded"],
                             "manufacturer": "X", "kvp": 120, "z_extent_mm": 300.0})
    registry.to_csv(root / "registry.csv", index=False)
    return root


@pytest.fixture
def pool(tmp_path):
    return load_pool_index(write_store(tmp_path / "store"), tmp_path / "store" / "registry.csv")


def test_index_carries_collection_and_registry_fields(pool):
    assert len(pool) == 9 * 12
    assert pool["collection"].notna().all() and pool["z_extent_mm"].notna().all()
    assert set(pool["status"]) == {"stored", "missing", "rejected:mc_overshoot_crop"}


def test_anatomy_tables_count_stored_beamlets_and_fates(pool):
    tables = anatomy_tables(pool)
    stored = int((pool["status"] == "stored").sum())
    assert tables["by_anatomy_split"].loc["total", "total"] == stored
    fates = tables["fates"]
    assert int(fates["planned"].sum()) == len(pool)
    assert np.allclose(fates["stored_fraction"], 10 / 12)


def test_gini_and_concentration():
    assert gini(np.full(10, 5)) == pytest.approx(0.0)
    assert gini(np.r_[np.zeros(9), 10.0]) == pytest.approx(0.9)
    c = concentration(np.array([10, 10, 10, 10]))
    assert c["effective_patients"] == pytest.approx(4.0) and c["top_decile_share"] == pytest.approx(0.25)


def test_patient_tables_one_row_per_patient(pool):
    tables = patient_tables(pool)
    per = tables["per_patient"]
    assert len(per) == 9 and (per["stored"] == 10).all() and (per["fields"] == 2).all()
    assert tables["concentration"].loc["all", "gini"] == pytest.approx(0.0)


def test_spot_positions_follow_the_crossed_bdl_helper(pool):
    """theta_x steers along the gantry axis through SMY, theta_y laterally through SMX."""
    frame = gantry_frame(pool, d_smx=2000.0, d_smy=2500.0)
    row = frame.iloc[0]
    assert row["y_g_mm"] == pytest.approx(2500.0 * np.tan(np.deg2rad(row["theta_x_deg"])))
    assert row["x_g_mm"] == pytest.approx(2000.0 * np.tan(np.deg2rad(row["theta_y_deg"])))
    assert len(frame) == int((pool["status"] == "stored").sum())


def test_gantry_tables_sum_to_the_stored_beamlets(pool):
    frame = gantry_frame(pool, 2000.0, 2500.0)
    tables = gantry_tables(frame, gantry_bin_deg=30.0)
    assert int(tables["by_gantry_sector"].to_numpy().sum()) == len(frame)
    assert len(tables["by_gantry_sector"]) == 12
    assert int(tables["spot_positions"]["beamlets"].sum()) == len(frame)
    uni = tables["uniformity"]
    assert int(uni.loc[("all", "gantry_sector_fields"), "mean_per_cell"] * 12) == frame.drop_duplicates(
        ["patient_id", "gantry_deg"]).shape[0]


def test_figures_render(pool, tmp_path):
    frame = gantry_frame(pool, 2000.0, 2500.0)
    per = patient_tables(pool)["per_patient"]
    for fn, args in [(anatomy_figure, pool), (patient_figure, per), (gantry_frame_figure, frame)]:
        paths = fn(args, figure_path=str(tmp_path / fn.__name__), title="t", caption="c")
        assert all(p.exists() for p in paths)
