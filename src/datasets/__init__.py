"""Reusable CT dataset abstraction (TCIA manifests, composable, provenance-first).

Shared by the MC beamlet-angle generator and the Active Learning beamlet sampler:
both need to enumerate patients across one or several TCIA collections, load a CT
on demand, and trace every sample back to its (dataset, anatomy, patient, series).
"""
from src.datasets.base import CTDataset, CTRecord
from src.datasets.multi import MultiDataset
from src.datasets.tcia import TCIADataset

__all__ = ["CTRecord", "CTDataset", "TCIADataset", "MultiDataset"]
