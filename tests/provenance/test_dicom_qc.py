"""Tests for DICOM QC gating + provenance extraction (src/provenance/dicom_qc.py)."""
from __future__ import annotations

import pytest

from src.provenance.dicom_qc import QCGates, check_quality, gates_from_dict, params_from_header


class _Hdr:
    """Minimal stand-in for a pydicom Dataset (attribute access)."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _good_params(**over):
    p = {"series_uid": "1.2.3", "modality": "CT", "photometric": "MONOCHROME2",
         "n_slices": 200, "pixel_spacing_x": 1.0, "pixel_spacing_y": 1.0,
         "slice_thickness": 1.5, "kvp": 120.0, "tube_current": 200.0,
         "convolution_kernel": "STANDARD"}
    p.update(over)
    return p


def test_default_gates_are_backward_compatible():
    # spacing/acq gates off by default -> only count+modality+photometric checked
    ok, reasons = check_quality(_good_params(pixel_spacing_x=5.0, slice_thickness=9.0), QCGates())
    assert ok and reasons == []


def test_count_modality_photometric_gates():
    g = QCGates()
    assert check_quality(_good_params(n_slices=10), g)[0] is False
    assert check_quality(_good_params(modality="RTSTRUCT"), g)[0] is False
    assert check_quality(_good_params(photometric="RGB"), g)[0] is False


def test_datagenerator_spacing_gates():
    g = QCGates(max_spacing_xy=1.5, max_spacing_z=2.0, min_slices=70)
    assert check_quality(_good_params(), g)[0] is True
    # 5 mm slices fail the z gate (like the abdominal StageII collection)
    ok, reasons = check_quality(_good_params(slice_thickness=5.0), g)
    assert not ok and any("slice_thickness" in r for r in reasons)
    # coarse pixels fail the xy gate
    assert check_quality(_good_params(pixel_spacing_x=1.6), g)[0] is False
    # missing spacing is a failure when the gate is on
    assert check_quality(_good_params(pixel_spacing_x=None), g)[0] is False


def test_abdominal_z_override_accepts_5mm():
    g = QCGates(max_spacing_xy=1.5, max_spacing_z=6.0, min_slices=50)
    assert check_quality(_good_params(slice_thickness=5.0), g)[0] is True


def test_kvp_and_tube_current_gates():
    g = QCGates(kvp_range=(80.0, 140.0), min_tube_current=50.0)
    assert check_quality(_good_params(kvp=120, tube_current=200), g)[0] is True
    assert check_quality(_good_params(kvp=200), g)[0] is False
    assert check_quality(_good_params(tube_current=10), g)[0] is False
    assert check_quality(_good_params(kvp=None), g)[0] is False   # missing gated data fails


def test_exclude_kernels_case_insensitive():
    g = QCGates(exclude_kernels=("bone", "sharp"))
    assert check_quality(_good_params(convolution_kernel="BONEPLUS"), g)[0] is False
    assert check_quality(_good_params(convolution_kernel="B31s soft"), g)[0] is True


def test_params_from_header_extracts_and_coerces():
    ds = _Hdr(SeriesInstanceUID="1.2", Modality="CT", PhotometricInterpretation="MONOCHROME2",
              PixelSpacing=[0.98, 0.98], SliceThickness="2.0", KVP="120",
              XRayTubeCurrent=200, ConvolutionKernel=["B", "30f"], Manufacturer="SIEMENS")
    p = params_from_header(ds, n_slices=180)
    assert p["n_slices"] == 180
    assert p["pixel_spacing_x"] == pytest.approx(0.98)
    assert p["slice_thickness"] == pytest.approx(2.0)   # coerced from str
    assert p["kvp"] == pytest.approx(120.0)
    assert p["modality"] == "CT" and p["manufacturer"] == "SIEMENS"
    assert "30f" in p["convolution_kernel"]


def test_gates_from_dict_merges_over_base():
    g = gates_from_dict({"max_spacing_z": 6.0, "kvp_range": [80, 140]},
                        min_slices=50, max_slices=1000)
    assert g.min_slices == 50 and g.max_spacing_z == 6.0
    assert g.kvp_range == (80.0, 140.0)
    assert g.max_spacing_xy is None       # not set -> stays off
