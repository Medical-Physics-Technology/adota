"""TCIA-manifest-backed CT dataset.

Manifest layout: ``<root>/<collection>/<patient>/<study>/<series>/*.dcm``. Each
patient carries several series (CT, RTSTRUCT, SEG, ...); this dataset selects, per
patient, the CT series (Modality==CT, MONOCHROME2) with the most slices, so the
index is the *patient* (what the reviewer counts, and what the AL sampler needs).

Patient subsetting (``n_patients`` / ``patient_ids``) is applied on directory names
*before* any DICOM header is read, so construction stays cheap on 250-patient
collections.
"""
from __future__ import annotations

import logging
import os
import random
from glob import glob
from typing import List, Optional, Sequence

import pydicom

from src.datasets.base import CTDataset, CTRecord
from src.provenance.dicom_qc import check_quality, gates_from_dict, params_from_header

logger = logging.getLogger(__name__)


def _subdirs(path: str) -> List[str]:
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def _auto_collection(root: str) -> str:
    """Single collection dir under a manifest root (ignores metadata.csv etc.)."""
    subs = _subdirs(root)
    if len(subs) != 1:
        raise ValueError(
            f"{root} has {len(subs)} collection dirs {subs}; pass collection= explicitly."
        )
    return subs[0]


class TCIADataset(CTDataset):
    """Patient-indexed CT dataset over one TCIA collection."""

    def __init__(
        self,
        root: str,
        collection: Optional[str] = None,
        anatomy: str = "",
        name: Optional[str] = None,
        patient_ids: Optional[Sequence[str]] = None,
        n_patients: Optional[int] = None,
        selection: str = "first",  # "first" | "last" | "random"
        seed: int = 0,
        min_slices: int = 50,
        max_slices: int = 1000,
        require_ct: bool = True,
        require_monochrome2: bool = True,
        qc: Optional[dict] = None,
        verbose: bool = False,
    ):
        self.root = root
        self.collection = collection or _auto_collection(root)
        self.collection_dir = os.path.join(root, self.collection)
        self.anatomy = anatomy
        self.name = name or self.collection
        # QC gates: legacy scalars form the base; `qc` dict opts into the extended
        # spacing / kVp / tube-current gates (all off by default). Provenance is
        # always recorded regardless of gating.
        self.qc_gates = gates_from_dict(
            qc, min_slices=min_slices, max_slices=max_slices,
            require_ct=require_ct, require_monochrome2=require_monochrome2)
        self.verbose = verbose

        # --- choose patients from directory names only (cheap) ---
        all_patients = _subdirs(self.collection_dir)
        if patient_ids is not None:
            wanted = list(patient_ids)
            missing = [p for p in wanted if p not in set(all_patients)]
            if missing:
                raise ValueError(f"patients not found in {self.collection}: {missing}")
            chosen = wanted
        else:
            chosen = all_patients
            if n_patients is not None and n_patients < len(chosen):
                if selection == "random":
                    chosen = sorted(random.Random(seed).sample(chosen, n_patients))
                elif selection == "last":
                    # take from the back: training-set generation consumed the
                    # first samples, so held-out / expansion patients come last.
                    chosen = chosen[-n_patients:]
                else:
                    chosen = chosen[:n_patients]

        # --- scan only the chosen patients for their CT series ---
        self._records: List[CTRecord] = []
        for pid in chosen:
            rec = self._select_ct_series(pid)
            if rec is not None:
                self._records.append(rec)
            elif verbose:
                logger.warning("no qualifying CT series for patient %s", pid)
        logger.info("TCIADataset[%s]: %d/%d patients -> %d CT series",
                    self.name, len(self._records), len(chosen), len(self._records))

    def _series_dirs(self, patient_id: str):
        pdir = os.path.join(self.collection_dir, patient_id)
        for study in _subdirs(pdir):
            for series in _subdirs(os.path.join(pdir, study)):
                yield os.path.join(pdir, study, series)

    def _select_ct_series(self, patient_id: str) -> Optional[CTRecord]:
        """Return the largest QC-passing CT series for a patient (or None).

        Extracts acquisition provenance from every candidate's header, applies the
        configured QC gates, and keeps the passing series with the most slices. The
        chosen record carries its provenance (+ ``qc_pass``/``qc_reasons``).
        """
        best = None  # (n_slices, series_dir, provenance)
        for series_dir in self._series_dirs(patient_id):
            dcm = glob(os.path.join(series_dir, "*.dcm"))
            if not dcm:
                continue
            try:
                ds = pydicom.dcmread(dcm[0], stop_before_pixels=True)
            except Exception:
                continue
            prov = params_from_header(ds, len(dcm))
            qc_pass, reasons = check_quality(prov, self.qc_gates)
            if not qc_pass:
                if self.verbose:
                    logger.info("  QC drop %s series (%s): %s", patient_id,
                                os.path.basename(series_dir), ", ".join(reasons))
                continue
            if best is None or prov["n_slices"] > best[0]:
                best = (prov["n_slices"], series_dir, prov)
        if best is None:
            return None
        n, series_dir, prov = best
        prov = dict(prov, qc_pass=True, qc_reasons=[])
        return CTRecord(
            dataset_name=self.name, anatomy=self.anatomy, patient_id=patient_id,
            series_uid=prov["series_uid"], series_dir=series_dir, n_slices=n,
            provenance=prov,
        )

    def __len__(self) -> int:
        return len(self._records)

    def record(self, idx: int) -> CTRecord:
        return self._records[idx]
