"""Verify a rebuilt beamlet HDF5 (v3) against its source (v2).

    uv run python scripts/check_beamlet_h5.py --old v2.h5 --new v3.h5 \\
        --exclusion-list <path> --out report.json

Checks, each a top-level section of the JSON report keyed by letter (spec
``docs/dev/h5_v3_spec.md`` section 6.3):

(a) ``set(new) == set(old) - set(exclusion)``.
(b) no exclusion id present in ``new`` (subsumed by (a), reported separately).
(c) for the sampled ids: dataset storage parameters and array equality
    (``ct``, ``dose`` exact; ``flux`` exact or, failing that, ``allclose``),
    plus the six v2 attrs (value and type/dtype).
(d) the v3-only attrs (section 4) present and non-empty on every sampled id.
(e) the whole-file ``spot_key`` multiplicity histogram (informational).
(f) for up to 5 sampled ids, ``H5PYGenerator`` output is ``torch.equal``
    between old and new.

``verdict`` is "pass" only if (a), (b), (c), (d) and (f) all pass; (e) never
fails. Exit code 1 on a "fail" verdict.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Annotated, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import typer

from src.datasets.beamlet_h5 import V3_ATTRS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("check_beamlet_h5")
app = typer.Typer(help="Verify a v3 beamlet HDF5 against its v2 source.")

V2_ATTRS: Tuple[str, ...] = (
    "dose_deposition_ratio", "gantry_angle", "id", "initial_energy", "beamlet_angles", "stat_uncertainty",
)
DATASET_NAMES: Tuple[str, ...] = ("ct", "dose", "flux")
STORAGE_PARAMS: Tuple[str, ...] = ("dtype", "shape", "chunks", "compression", "compression_opts")


def _read_lines(path: Path) -> List[str]:
    """One id per line, blank lines dropped, order preserved."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def ulp_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Max ULP distance between two float32 arrays' bit patterns viewed as int32."""
    a_bits = np.asarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    b_bits = np.asarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    return int(np.max(np.abs(a_bits - b_bits)))


def _attr_value_equal(old_value: object, new_value: object) -> bool:
    """Value equality with NaN == NaN, for both scalar and array-valued attrs."""
    if isinstance(old_value, np.ndarray) or isinstance(new_value, np.ndarray):
        old_arr, new_arr = np.asarray(old_value), np.asarray(new_value)
        if old_arr.shape != new_arr.shape:
            return False
        if np.issubdtype(old_arr.dtype, np.floating) or np.issubdtype(new_arr.dtype, np.floating):
            return bool(np.array_equal(old_arr, new_arr, equal_nan=True))
        return bool(np.array_equal(old_arr, new_arr))
    if isinstance(old_value, float) and isinstance(new_value, float):
        if math.isnan(old_value) and math.isnan(new_value):
            return True
        return bool(old_value == new_value)
    return bool(old_value == new_value)


def _attr_type_equal(old_value: object, new_value: object) -> bool:
    """type/dtype/shape agreement, per spec section 6.3 (c)."""
    if type(old_value).__name__ != type(new_value).__name__:
        return False
    if getattr(old_value, "dtype", None) != getattr(new_value, "dtype", None):
        return False
    if getattr(old_value, "shape", None) != getattr(new_value, "shape", None):
        return False
    return True


def _attr_nonempty(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, str):
        return len(value) > 0
    return value is not None


def compare_ids(old_ids: Iterable[str], new_ids: Iterable[str], excluded: Iterable[str],
                allow_new: Iterable[str] = ()) -> dict:
    """Checks (a): the id-set arithmetic, plus the informational and unexpected sets.

    ``allow_new`` lists ids that may be in ``new`` without being in ``old`` (records
    the old build lost to read errors and the new one recovered); they are reported
    under ``allowed_new`` and do not fail the check.
    """
    old_set, new_set, excluded_set, allowed = set(old_ids), set(new_ids), set(excluded), set(allow_new)
    retained = old_set - excluded_set
    listed_but_present = sorted(new_set & excluded_set)
    missing_from_new = sorted(retained - new_set)
    listed_not_in_old = sorted(excluded_set - old_set)
    unexpected_in_new = sorted(new_set - old_set - excluded_set - allowed)
    passed = not (listed_but_present or missing_from_new or unexpected_in_new)
    return {
        "pass": passed,
        "n_old": len(old_set),
        "n_new": len(new_set),
        "n_excluded": len(excluded_set),
        "n_retained_expected": len(retained),
        "listed_but_present": listed_but_present,
        "missing_from_new": missing_from_new,
        "listed_not_in_old": listed_not_in_old,
        "unexpected_in_new": unexpected_in_new,
        "allowed_new": sorted(new_set & allowed),
    }


def check_exclusion_absent(ids_report: Mapping[str, object]) -> dict:
    """Check (b): no exclusion id present in new (a subset of (a)'s evidence)."""
    listed_but_present = ids_report["listed_but_present"]
    return {"pass": not listed_but_present, "excluded_present_in_new": listed_but_present}


def compare_record(old_group: h5py.Group, new_group: h5py.Group) -> dict:
    """Check (c) for one record: dataset storage/array equality, six v2 attrs."""
    dataset_reports: Dict[str, dict] = {}
    for name in DATASET_NAMES:
        old_ds, new_ds = old_group[name], new_group[name]
        storage_mismatches = [p for p in STORAGE_PARAMS if getattr(old_ds, p) != getattr(new_ds, p)]
        old_arr, new_arr = old_ds[()], new_ds[()]
        report: dict = {"storage_mismatches": storage_mismatches}
        if name == "flux":
            if bool(np.array_equal(old_arr, new_arr)):
                report["verdict"] = "array_equal"
                report["pass"] = not storage_mismatches
            else:
                old64, new64 = old_arr.astype(np.float64), new_arr.astype(np.float64)
                abs_diff = np.abs(old64 - new64)
                denom = np.where(old64 == 0, 1.0, np.abs(old64))
                report["max_abs"] = float(np.max(abs_diff))
                report["max_rel"] = float(np.max(abs_diff / denom))
                report["max_ulp"] = ulp_distance(old_arr, new_arr)
                if bool(np.allclose(old_arr, new_arr, rtol=1e-6, atol=0)):
                    report["verdict"] = "allclose"
                    report["pass"] = not storage_mismatches
                else:
                    report["verdict"] = "fail"
                    report["pass"] = False
        else:
            equal = bool(np.array_equal(old_arr, new_arr))
            report["array_equal"] = equal
            report["pass"] = equal and not storage_mismatches
        dataset_reports[name] = report

    attr_reports: Dict[str, dict] = {}
    for attr in V2_ATTRS:
        old_value, new_value = old_group.attrs[attr], new_group.attrs[attr]
        value_equal = _attr_value_equal(old_value, new_value)
        type_equal = _attr_type_equal(old_value, new_value)
        attr_reports[attr] = {"pass": value_equal and type_equal, "value_equal": value_equal, "type_equal": type_equal}

    record_pass = all(r["pass"] for r in dataset_reports.values()) and all(r["pass"] for r in attr_reports.values())
    return {
        "pass": record_pass,
        "datasets": dataset_reports,
        "attrs": attr_reports,
        "flux_verdict": dataset_reports["flux"].get("verdict", "array_equal"),
    }


def check_v3_attrs(new_group: h5py.Group, attr_names: Sequence[str]) -> dict:
    """Check (d) for one record: every v3-only attr present and non-empty."""
    missing = [a for a in attr_names if a not in new_group.attrs or not _attr_nonempty(new_group.attrs[a])]
    return {"pass": not missing, "missing": missing}


def spot_key_histogram(new_path: Path) -> dict:
    """Check (e): multiplicity histogram of spot_key over the whole new file (informational)."""
    index_csv = new_path.with_name(f"{new_path.stem}_index.csv")
    spot_keys: List[str] = []
    if index_csv.exists():
        source = "index_csv"
        with open(index_csv, newline="", encoding="utf-8") as fh:
            spot_keys = [row["spot_key"] for row in csv.DictReader(fh)]
    else:
        # informational: a record missing spot_key (e.g. an incomplete build) is
        # skipped rather than crashing the whole checker.
        source = "attrs"
        with h5py.File(new_path, "r") as h5f:
            spot_keys = [str(spot_key) for key in h5f.keys()
                         if (spot_key := h5f[key].attrs.get("spot_key")) is not None]
    counts = Counter(spot_keys)
    multiplicity_counts = Counter(counts.values())
    mode = max(multiplicity_counts.items(), key=lambda kv: kv[1])[0] if multiplicity_counts else None
    return {
        "pass": True,
        "source": source,
        "n_spot_keys": len(counts),
        "histogram": {str(k): v for k, v in sorted(multiplicity_counts.items())},
        "mode": mode,
    }


def check_generator_equivalence(
    old_path: Path, new_path: Path, exclusion_list: Path, ids: Sequence[str], depths: Mapping[str, int],
) -> dict:
    """Check (f): H5PYGenerator((x, e, y)) matches between old and new, per id."""
    from src.loaders.generator import H5PYGenerator  # local: pulls in torch + augmentation deps

    mismatched: List[str] = []
    for sample_id in ids:
        expected_shape = (depths[sample_id], 40, 40)
        kwargs = dict(indexes=[sample_id], augmentation=False, cropp=False, normalize=False,
                      indexes_to_exclude_list=str(exclusion_list), expected_shape=expected_shape)
        x_old, e_old, y_old = H5PYGenerator(str(old_path), **kwargs)[0]
        x_new, e_new, y_new = H5PYGenerator(str(new_path), **kwargs)[0]
        if not (torch.equal(x_old, x_new) and torch.equal(e_old, e_new) and torch.equal(y_old, y_new)):
            mismatched.append(sample_id)
    return {"pass": not mismatched, "checked_ids": list(ids), "mismatched_ids": mismatched}


def run_checks(
    old_path: Path,
    new_path: Path,
    exclusion_list: Path,
    sample: int = 200,
    seed: int = 0,
    ids_file: Optional[Path] = None,
    allow_new: Optional[Path] = None,
) -> dict:
    """Run checks (a)-(f) and assemble the report dict. Raises nothing on a fail verdict."""
    old_path, new_path, exclusion_list = Path(old_path), Path(new_path), Path(exclusion_list)
    excluded = set(_read_lines(exclusion_list))
    allowed_new = _read_lines(allow_new) if allow_new is not None else []

    with h5py.File(old_path, "r") as old_h5, h5py.File(new_path, "r") as new_h5:
        old_ids, new_ids = list(old_h5.keys()), list(new_h5.keys())
        # An ids-file scopes the whole comparison: a partial build (a smoke test) holds
        # only the listed candidates, so the id-set arithmetic runs on old ∩ ids-file.
        scope = set(_read_lines(ids_file)) if ids_file is not None else None
        if scope is not None:
            old_ids = [i for i in old_ids if i in scope]
        ids_report = compare_ids(old_ids, new_ids, excluded, allow_new=allowed_new)
        ids_report["scoped_to_ids_file"] = scope is not None
        exclusion_report = check_exclusion_absent(ids_report)

        retained_sorted = sorted(set(old_ids) - excluded)
        if ids_file is not None:
            sample_ids = retained_sorted
        elif sample == 0:
            sample_ids = retained_sorted
        else:
            sample_ids = random.Random(seed).sample(retained_sorted, min(sample, len(retained_sorted)))

        record_failures: List[dict] = []
        attr_failures: List[dict] = []
        flux_verdicts: List[str] = []
        depths: Dict[str, int] = {}
        for sample_id in sample_ids:
            if sample_id not in new_h5:
                record_failures.append({"id": sample_id, "dataset": None, "detail": "missing_from_new"})
                continue
            old_group, new_group = old_h5[sample_id], new_h5[sample_id]
            depths[sample_id] = old_group["ct"].shape[2]

            record_report = compare_record(old_group, new_group)
            flux_verdicts.append(record_report["flux_verdict"])
            if not record_report["pass"]:
                for name, ds_report in record_report["datasets"].items():
                    if not ds_report["pass"]:
                        record_failures.append({"id": sample_id, "dataset": name, "detail": ds_report})
                for attr, attr_report in record_report["attrs"].items():
                    if not attr_report["pass"]:
                        record_failures.append({"id": sample_id, "dataset": f"attr:{attr}", "detail": attr_report})

            v3_report = check_v3_attrs(new_group, V3_ATTRS)
            for attr in v3_report["missing"]:
                attr_failures.append({"id": sample_id, "attr": attr})

        c_report = {"pass": not record_failures, "n_sampled": len(sample_ids), "failures": record_failures}
        d_report = {"pass": not attr_failures, "missing": attr_failures}

        if any(v == "fail" for v in flux_verdicts):
            flux_equivalence = "fail"
        elif any(v == "allclose" for v in flux_verdicts):
            flux_equivalence = "allclose"
        else:
            flux_equivalence = "array_equal"

        e_report = spot_key_histogram(new_path)

        f_ids = [sample_id for sample_id in sample_ids if sample_id in depths][:5]
        f_report = check_generator_equivalence(old_path, new_path, exclusion_list, f_ids, depths)

    verdict = "pass" if all(
        r["pass"] for r in (ids_report, exclusion_report, c_report, d_report, f_report)
    ) else "fail"

    return {
        "old_path": str(old_path),
        "new_path": str(new_path),
        "exclusion_list": str(exclusion_list),
        "seed": seed,
        "sample_size": sample,
        "sampled_ids": sample_ids,
        "a": ids_report,
        "b": exclusion_report,
        "c": c_report,
        "d": d_report,
        "e": e_report,
        "f": f_report,
        "verdict": verdict,
        "flux_equivalence": flux_equivalence,
    }


@app.command()
def main(
    old: Annotated[Path, typer.Option(help="v2 HDF5 file.")],
    new: Annotated[Path, typer.Option(help="v3 HDF5 file to verify.")],
    exclusion_list: Annotated[Path, typer.Option("--exclusion-list", help="Ids excluded from v3, one per line.")],
    sample: Annotated[int, typer.Option(help="Ids to sample for checks (c)-(f); 0 = all retained ids.")] = 200,
    seed: Annotated[int, typer.Option(help="Seed for random.Random when drawing the sample.")] = 0,
    ids_file: Annotated[Optional[Path], typer.Option(
        help="Explicit id list, one per line; overrides --sample.")] = None,
    out: Annotated[Path, typer.Option(help="Report JSON output path.")] = Path("report.json"),
    allow_new: Annotated[Optional[Path], typer.Option(
        "--allow-new", help="Ids allowed in new but absent from old (records old lost to read "
                            "errors), one per line; reported under a.allowed_new.")] = None,
) -> None:
    report = run_checks(old, new, exclusion_list, sample=sample, seed=seed, ids_file=ids_file,
                        allow_new=allow_new)
    out.write_text(json.dumps(report, indent=2))
    logger.info("verdict=%s flux_equivalence=%s report=%s", report["verdict"], report["flux_equivalence"], out)
    if report["verdict"] != "pass":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
