"""HDF5 writer, attribute schema and CSV side outputs for the v3 beamlet build.

Pure helpers (spec ``docs/dev/h5_v3_spec.md`` section 6.2): turning a
:class:`~src.datasets.beamlet_record.PreprocessedRecord` plus the raw metadata into
an HDF5 group with the v2 + v3 attribute schema, the index/skip CSV rows, and the
exclusion-aware candidate list. No process pool, no CLI, no torch device handling
here -- that orchestration is :mod:`scripts.build_beamlet_h5`, which is the only
importer of :func:`plan_candidates` and the writer functions below.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import h5py
import numpy as np

from src.datasets.beamlet_record import PreprocessedRecord, list_record_ids

SCHEMA_VERSION = 3
V2_ATTRS: Tuple[str, ...] = (
    "dose_deposition_ratio", "gantry_angle", "id", "initial_energy", "beamlet_angles", "stat_uncertainty",
)
V3_ATTRS: Tuple[str, ...] = (
    "schema_version", "source_dataset", "energy_mev", "gantry_angle_sim", "isocenter_mm", "image_origin_mm",
    "image_size_vox", "image_spacing_mm", "roi_size_vox", "bixel_shift_xy_mm", "ray_entrance_mm",
    "ray_entrance_proj_mm", "num_primaries", "flux_model", "flux_compute", "bdl_file", "bdl_sha256",
    "flux_sigma_xy_mm", "downsample_method", "patient_key", "spot_key", "metadata_json",
)
DATASET_NAMES: Tuple[str, ...] = ("ct", "dose", "flux")
DATASET_KWARGS: Dict[str, object] = dict(compression="gzip", chunks=True)
SKIP_REASONS: Tuple[str, ...] = (
    "excluded", "zero_dose", "bad_shape", "short_depth", "load_error", "json_error", "n_spots",
)


@dataclass(frozen=True)
class RecordProvenance:
    source_dataset: str
    flux_compute: str
    bdl_file: str
    bdl_sha256: str
    downsample_method: str
    flux_model: str = "SingleGaussian"


def derive_keys(metadata: dict) -> Tuple[str, str]:
    """``(patient_key, spot_key)`` per spec section 4.

    ``gantry_angle`` in ``spot_key`` is the top-level, patient-frame angle (the one
    that varies between spots sharing a patient geometry), not
    ``simulation_log.gantry_angle``.
    """
    patient_str = "|".join([
        *(f"{v:.3f}" for v in metadata["image_origin"]),
        *(str(int(v)) for v in metadata["image_size"]),
        *(f"{v:.3f}" for v in metadata["image_spacing"]),
    ])
    patient_key = hashlib.sha1(patient_str.encode("utf-8")).hexdigest()[:16]

    sim = metadata["simulation_log"]
    spot_str = "|".join([
        patient_key,
        *(f"{v:.3f}" for v in sim["isocenter"]),
        f"{metadata['gantry_angle']:.3f}",
        *(f"{v:.3f}" for v in sim["bixelgrid_shifts_xy"][0]),
    ])
    spot_key = hashlib.sha1(spot_str.encode("utf-8")).hexdigest()[:16]
    return patient_key, spot_key


def v2_attrs(metadata: dict, initial_energy_norm: float) -> dict:
    """The six v2 attrs, JSON-native values, lists converted with ``np.array``.

    No casts: v2 stored whatever type the JSON parsed to, and the checker compares
    dtypes, so a cast here would show up as a v2/v3 mismatch.
    """
    return {
        "dose_deposition_ratio": metadata["dose_deposition_ratio"],
        "gantry_angle": metadata["gantry_angle"],
        "id": metadata["id"],
        "initial_energy": float(initial_energy_norm),
        "beamlet_angles": np.array(metadata["simulation_log"]["beamlet_angles"]),
        "stat_uncertainty": metadata["stat_uncertainty"],
    }


def v3_attrs(metadata: dict, metadata_text: str, pre: PreprocessedRecord, prov: RecordProvenance) -> dict:
    """The v3-only attrs of spec section 4, in table order, with the listed dtypes."""
    sim = metadata["simulation_log"]
    patient_key, spot_key = derive_keys(metadata)
    return {
        "schema_version": np.int64(SCHEMA_VERSION),
        "source_dataset": prov.source_dataset,
        "energy_mev": np.float64(pre.energy_mev),
        "gantry_angle_sim": np.float64(sim["gantry_angle"]),
        "isocenter_mm": np.array(sim["isocenter"], dtype=np.float64),
        "image_origin_mm": np.array(metadata["image_origin"], dtype=np.float64),
        "image_size_vox": np.array(metadata["image_size"], dtype=np.int64),
        "image_spacing_mm": np.array(metadata["image_spacing"], dtype=np.float64),
        "roi_size_vox": np.array(metadata["roi_size"], dtype=np.int64),
        "bixel_shift_xy_mm": np.array(sim["bixelgrid_shifts_xy"][0], dtype=np.float64),
        "ray_entrance_mm": np.array(metadata["rays_entrence_point"], dtype=np.float64),
        "ray_entrance_proj_mm": np.array(metadata["rays_entrence_point_proj"], dtype=np.float64),
        "num_primaries": np.float64(sim["sim_params"]["Num_Primaries"]),
        "flux_model": prov.flux_model,
        "flux_compute": prov.flux_compute,
        "bdl_file": prov.bdl_file,
        "bdl_sha256": prov.bdl_sha256,
        "flux_sigma_xy_mm": np.array(pre.sigmas_xy, dtype=np.float64),
        "downsample_method": prov.downsample_method,
        "patient_key": patient_key,
        "spot_key": spot_key,
        "metadata_json": metadata_text,
    }


def write_record(
    h5: h5py.File, pre: PreprocessedRecord, raw_meta: dict, metadata_text: str, prov: RecordProvenance,
) -> None:
    """Create ``<pre.sample_id>``: datasets ct/dose/flux, then the v2 six then v3 attrs."""
    group = h5.create_group(pre.sample_id)
    for name, data in zip(DATASET_NAMES, (pre.ct, pre.dose, pre.flux)):
        group.create_dataset(name, data=data, **DATASET_KWARGS)
    v2 = v2_attrs(raw_meta, pre.initial_energy_norm)
    for key in V2_ATTRS:
        group.attrs[key] = v2[key]
    v3 = v3_attrs(raw_meta, metadata_text, pre, prov)
    for key in V3_ATTRS:
        group.attrs[key] = v3[key]


def is_complete_group(group: h5py.Group) -> bool:
    """``True`` when ``group`` has the three datasets and the ``schema_version`` attr."""
    return all(name in group for name in DATASET_NAMES) and "schema_version" in group.attrs


def write_file_attrs(h5: h5py.File, attrs: Mapping[str, object]) -> None:
    """Write file-level attrs (section 5); a dict value is JSON-serialised first.

    A ``str`` subclass (e.g. ``torch.__version__``'s ``TorchVersion``) is coerced to
    a plain ``str``: h5py's fixed-width-array fallback for such objects raises
    ``TypeError: No conversion path for dtype`` instead of writing a scalar string.
    """
    for key, value in attrs.items():
        if isinstance(value, dict):
            h5.attrs[key] = json.dumps(value)
        elif isinstance(value, str):
            h5.attrs[key] = str(value)
        else:
            h5.attrs[key] = value


def index_row(sample_id: str, attrs: Mapping) -> dict:
    """One index-CSV row: scalars as-is, vectors flattened as ``name_0``, ``name_1``, ...

    ``metadata_json`` is excluded (section 5).
    """
    row: dict = {"sample_id": sample_id}
    for key, value in attrs.items():
        if key == "metadata_json":
            continue
        if isinstance(value, np.ndarray):
            for i, component in enumerate(value.tolist()):
                row[f"{key}_{i}"] = component
        elif isinstance(value, (list, tuple)):
            for i, component in enumerate(value):
                row[f"{key}_{i}"] = component
        else:
            row[key] = value
    return row


def _write_csv(rows: Sequence[dict], path: Path, fieldnames: List[str]) -> None:
    with open(Path(path), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_index_csv(rows: Sequence[dict], path: Path) -> None:
    """``sample_id`` then every attr column (section 5), one row per written record."""
    if not rows:
        _write_csv(rows, path, ["sample_id"])
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    _write_csv(rows, path, fieldnames)


def write_skip_csv(rows: Sequence[dict], path: Path) -> None:
    """Columns ``sample_id``, ``source``, ``reason``, ``detail``; one row per skipped id."""
    _write_csv(rows, path, ["sample_id", "source", "reason", "detail"])


def sha256_of(path: Path) -> str:
    """sha256 hex digest of the file at ``path`` (raises if it does not exist)."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ids_file(path: Path) -> Set[str]:
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def plan_candidates(
    raw_root: Path,
    sources: Sequence[str],
    exclusion_ids: Set[str],
    ids_file: Optional[Path],
    limit: Optional[int],
) -> Tuple[List[Tuple[str, str]], List[dict]]:
    """Candidate ``(source, id)`` pairs, split into retained and ``excluded`` skip rows.

    Per source, in order: :func:`list_record_ids`, restricted to the ids-file set
    when given. Concatenated across sources, then truncated to ``limit``. Only then
    is the exclusion filter applied, so a listed id beyond ``limit`` never appears in
    either list and a listed id within ``limit`` always gets an ``excluded`` row.
    """
    ids_filter = _read_ids_file(ids_file) if ids_file is not None else None
    candidates: List[Tuple[str, str]] = []
    for source in sources:
        ids = list_record_ids(Path(raw_root) / source)
        if ids_filter is not None:
            ids = [i for i in ids if i in ids_filter]
        candidates.extend((source, sample_id) for sample_id in ids)
    if limit is not None:
        candidates = candidates[:limit]

    retained: List[Tuple[str, str]] = []
    skip_rows: List[dict] = []
    for source, sample_id in candidates:
        if sample_id in exclusion_ids:
            skip_rows.append({"sample_id": sample_id, "source": source, "reason": "excluded", "detail": ""})
        else:
            retained.append((source, sample_id))
    return retained, skip_rows
