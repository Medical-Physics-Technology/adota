"""Build a (multi-)dataset from a YAML-style config dict (architecture-level control).

Config shape::

    datasets:
      - name: NSCLC-Radiomics
        anatomy: thoracic
        root: /scratch/mstryja/manifest-1603198545583
        collection: NSCLC-Radiomics      # optional; auto-detected if single
        n_patients: 2                     # optional; null/absent = all
        selection: first                  # first | random
        seed: 0
        patient_ids: [LUNG1-195, ...]     # optional; overrides n_patients
      - name: Prostate-AEC
        anatomy: pelvic
        root: /scratch/mstryja/manifest-1684259732535
        n_patients: 2

One entry -> a single-collection dataset; several entries -> a MultiDataset.
"""
from __future__ import annotations

from typing import List

from src.datasets.base import CTDataset
from src.datasets.multi import MultiDataset
from src.datasets.tcia import TCIADataset

_ALLOWED = {
    "name", "anatomy", "root", "collection", "n_patients", "selection", "seed",
    "patient_ids", "min_slices", "max_slices", "require_ct", "require_monochrome2",
    "qc",
}


def build_tcia_dataset(spec: dict) -> TCIADataset:
    """Construct one TCIADataset from a spec dict."""
    unknown = set(spec) - _ALLOWED
    if unknown:
        raise ValueError(f"unknown dataset keys {sorted(unknown)}; allowed {sorted(_ALLOWED)}")
    if "root" not in spec:
        raise ValueError(f"dataset spec missing 'root': {spec}")
    return TCIADataset(
        root=spec["root"], collection=spec.get("collection"),
        anatomy=spec.get("anatomy", ""), name=spec.get("name"),
        patient_ids=spec.get("patient_ids"), n_patients=spec.get("n_patients"),
        selection=spec.get("selection", "first"), seed=int(spec.get("seed", 0)),
        min_slices=int(spec.get("min_slices", 50)),
        max_slices=int(spec.get("max_slices", 1000)),
        require_ct=bool(spec.get("require_ct", True)),
        require_monochrome2=bool(spec.get("require_monochrome2", True)),
        qc=spec.get("qc"),
    )


def build_dataset_from_config(cfg: dict) -> CTDataset:
    """Build a single dataset (one spec) or a MultiDataset (several) from config."""
    specs = cfg.get("datasets")
    if not specs:
        raise ValueError("config has no 'datasets' list")
    built: List[TCIADataset] = [build_tcia_dataset(s) for s in specs]
    if len(built) == 1:
        return built[0]
    return MultiDataset(built, name=cfg.get("name", "multi"))
