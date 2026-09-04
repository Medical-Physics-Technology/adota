"""Base types for the CT dataset abstraction: provenance record + dataset ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import SimpleITK as sitk


@dataclass(frozen=True)
class CTRecord:
    """A single CT series with full provenance (loads its image on demand).

    Provenance is first-class so any downstream sample (an MC beamlet, an AL
    acquisition score) can be traced back to the exact CT it came from.
    """

    dataset_name: str          # config/collection name, e.g. "NSCLC-Radiomics"
    anatomy: str               # "thoracic" | "pelvic" | "abdominal" | "head_neck" | ...
    patient_id: str            # e.g. "Prostate-AEC-058"
    series_uid: str            # DICOM SeriesInstanceUID (stable id)
    series_dir: str            # directory holding the series' .dcm files
    n_slices: int
    # Acquisition provenance + QC (pixel spacing, kVp, tube current, kernel, ...);
    # None for synthetic sources. See src/provenance/dicom_qc.py.
    provenance: Optional[Dict] = None

    @property
    def uid(self) -> str:
        """Globally unique, stable identifier for this CT across datasets."""
        return f"{self.dataset_name}/{self.patient_id}/{self.series_uid}"

    def load_image(self) -> sitk.Image:
        """Read the series into a geometrically-ordered SITK image."""
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(self.series_dir, self.series_uid)
        if not files:
            raise RuntimeError(f"No DICOM files for series {self.uid} in {self.series_dir}")
        reader.SetFileNames(files)
        return reader.Execute()


class CTDataset(ABC):
    """Indexable collection of :class:`CTRecord`s (patient-indexed by default)."""

    name: str
    anatomy: str

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def record(self, idx: int) -> CTRecord: ...

    def __getitem__(self, idx: int) -> CTRecord:
        return self.record(idx)

    def __iter__(self) -> Iterator[CTRecord]:
        for i in range(len(self)):
            yield self.record(i)

    def load(self, idx: int) -> Tuple[sitk.Image, CTRecord]:
        """Return ``(image, record)`` for index ``idx``."""
        rec = self.record(idx)
        return rec.load_image(), rec

    def patient_ids(self) -> List[str]:
        return [self.record(i).patient_id for i in range(len(self))]
