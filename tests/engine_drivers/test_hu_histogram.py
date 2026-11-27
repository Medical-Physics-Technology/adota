"""HU histograms of export CT crops (synthetic, CPU, small HDF5 in tmp_path)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pytest

from src.engine_drivers.hu_histogram import (
    CLASS_LABELS,
    HU_EDGES,
    accumulate_hu,
    class_frame,
    clipping_frame,
    export_hu,
    histogram_frame,
)

SCALE = {"min_ct": -1024, "max_ct": 3071}
SHAPE = (30, 30, 160)


def _normalise(hu: np.ndarray) -> np.ndarray:
    return ((np.asarray(hu, dtype=np.float64) - SCALE["min_ct"]) / (SCALE["max_ct"] - SCALE["min_ct"])).astype(
        np.float32)


def _crop(values: List[float], fill: float = 0.0) -> np.ndarray:
    hu = np.full(int(np.prod(SHAPE)), fill, dtype=np.float64)
    hu[:len(values)] = values
    return _normalise(hu).reshape(SHAPE)


def _write(path: Path, crops: Dict[str, np.ndarray]) -> Path:
    with h5py.File(path, "w") as f:
        f.attrs["scale_json"] = json.dumps({**SCALE, "min_ds": 0.0, "max_ds": 1.0})
        for sid, ct in crops.items():
            f.create_group(sid).create_dataset("ct", data=ct)
    return path


def _bin_of(per_site, site: str, hu: float) -> int:
    """Count in the bin whose low edge is the largest edge <= hu (last bin closed)."""
    i = min(int(np.searchsorted(HU_EDGES, hu, side="right")) - 1, len(HU_EDGES) - 2)
    return int(per_site[site].counts[i])


def test_edges_shape_and_last_bin_closed(tmp_path):
    assert HU_EDGES.shape == (411,)
    assert HU_EDGES[0] == -1024 and HU_EDGES[-2] == 3066 and HU_EDGES[-1] == 3071
    h5 = _write(tmp_path / "e.h5", {"a": _crop([3071.0, 3066.0, 3065.875], fill=-1024.0)})
    acc = accumulate_hu(h5, {"a": "thoracic"})["thoracic"]
    assert acc.counts[-1] == 2          # 3071 (right edge, closed) and 3066
    assert acc.counts[-2] == 1          # 3065.875 just below the last low edge
    assert acc.counts[0] == np.prod(SHAPE) - 3
    assert acc.counts.sum() == acc.n_voxels == np.prod(SHAPE)


def test_eighth_rounding_keeps_edge_in_its_bin():
    edge = -1014.0
    v = np.float32((edge - SCALE["min_ct"]) / (SCALE["max_ct"] - SCALE["min_ct"]))
    v_below = np.nextafter(v, np.float32(0))            # one float32 ulp low
    raw = float(v_below) * (SCALE["max_ct"] - SCALE["min_ct"]) + SCALE["min_ct"]
    assert raw < edge                                   # without rounding it would fall into the bin below
    assert export_hu(np.array([v_below]), SCALE)[0] == edge
    # 300 HU does not survive the float32 round trip exactly either (it lands just above 300)
    assert export_hu(_normalise([300.0]), SCALE)[0] == 300.0
    assert export_hu(_normalise([12.375]), SCALE)[0] == 12.375


def test_eighth_rounding_in_accumulation(tmp_path):
    edge = -1014.0
    v = np.float32((edge - SCALE["min_ct"]) / (SCALE["max_ct"] - SCALE["min_ct"]))
    ct = np.full(SHAPE, np.nextafter(v, np.float32(0)), dtype=np.float32)
    acc = accumulate_hu(_write(tmp_path / "r.h5", {"a": ct}), {"a": "pelvic"})["pelvic"]
    assert acc.counts[1] == np.prod(SHAPE) and acc.counts[0] == 0


def test_class_sides_and_body_threshold(tmp_path):
    values = [-500.0, -500.125, 300.0, 300.125, -499.875]
    h5 = _write(tmp_path / "c.h5", {"a": _crop(values, fill=-1000.0)})
    acc = accumulate_hu(h5, {"a": "abdominal"})["abdominal"]
    n = int(np.prod(SHAPE))
    below = (n - len(values)) + 1        # fill -1000 and -500.125
    above = 1                            # 300.125
    assert acc.class_counts.tolist() == [below, n - below - above, above]
    assert acc.class_counts.sum() == acc.n_voxels
    # body is HU > -500: excludes -500 exactly, keeps -499.875, 300, 300.125
    assert acc.body_counts.sum() == acc.class_counts[1] + acc.class_counts[2] - 1
    assert acc.body_counts.sum() == 3
    i_m500 = int(np.searchsorted(HU_EDGES, -500.0, side="right")) - 1
    assert acc.counts[i_m500] == 3 and acc.body_counts[i_m500] == 1   # -500, -500.125 out; -499.875 in
    cf = class_frame({"abdominal": acc})
    assert cf["hu_class"].tolist() == list(CLASS_LABELS)
    assert cf["share"].sum() == pytest.approx(1.0)


def test_clipping_counts(tmp_path):
    ct = _crop([-1030.0, -1100.0, 3080.0], fill=0.0)
    acc = accumulate_hu(_write(tmp_path / "k.h5", {"a": ct}), {"a": "thoracic"})["thoracic"]
    assert (acc.n_clipped_low, acc.n_clipped_high) == (2, 1)
    assert acc.counts[0] == 2 and acc.counts[-1] == 1
    assert acc.counts.sum() == acc.n_voxels
    row = clipping_frame({"thoracic": acc}).iloc[0]
    assert (row.n_crops, row.n_clipped_low, row.n_clipped_high) == (1, 2, 1)


def test_missing_id_raises(tmp_path):
    h5 = _write(tmp_path / "m.h5", {"a": _crop([])})
    with pytest.raises(KeyError, match="nope"):
        accumulate_hu(h5, {"a": "thoracic", "nope": "thoracic"})


def test_frames_site_order_and_columns(tmp_path):
    crops = {s: _crop([float(i * 100)]) for i, s in enumerate(["p", "a", "t"])}
    per = accumulate_hu(_write(tmp_path / "o.h5", crops), {"p": "pelvic", "a": "abdominal", "t": "thoracic"})
    hf = histogram_frame(per)
    assert list(hf.columns) == ["site", "bin_low_hu", "count"]
    assert list(dict.fromkeys(hf["site"])) == ["thoracic", "abdominal", "pelvic"]
    assert len(hf) == 3 * 410
    np.testing.assert_array_equal(hf[hf.site == "pelvic"]["bin_low_hu"].to_numpy(), HU_EDGES[:-1])
    hb = histogram_frame(per, body=True)
    assert hb["count"].sum() == sum(int(a.body_counts.sum()) for a in per.values())
    assert _bin_of(per, "abdominal", 100.0) == 1                       # its one voxel at 100 HU
    assert _bin_of(per, "abdominal", 0.0) == int(np.prod(SHAPE)) - 1    # the fill
    assert list(clipping_frame(per)["site"]) == ["thoracic", "abdominal", "pelvic"]
