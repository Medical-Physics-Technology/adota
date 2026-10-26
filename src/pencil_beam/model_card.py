"""README for a commissioned pencil-beam model directory.

A commissioned model is meant to outlive the run that built it: it is *the*
pencil-beam description of our beam (BDL + MCsquare physics), reusable from adota
or from DoseCUDA itself. The card states what it represents, its units and
conventions, how to use it, and (after validation) how accurate it measured.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = ["write_model_card"]


def _accuracy(rows: Sequence[dict], gamma_key: str) -> str:
    lines = ["| Metric | Commissioned nodes | Between nodes |", "|---|---|---|"]
    groups = {True: [r for r in rows if r.get("node")], False: [r for r in rows if not r.get("node")]}

    def cell(fn_rows, key, fn, absolute, fmt):
        vals = np.array([r[key] for r in fn_rows if key in r], dtype=float)
        if not vals.size:
            return "n/a"
        vals = np.abs(vals) if absolute else vals
        return fmt.format(float(fn(vals)))

    spec = [("max abs R80 error (mm)", "dr80_mm", np.nanmax, True, "{:.2f}"),
            ("IDD peak ratio PBA/MC, worst", "idd_peak_ratio", lambda v: v[np.nanargmax(np.abs(v - 1))], False,
             "{:.3f}"),
            ("max abs width error (mm)", "max_abs_dsigma_mm", np.nanmax, True, "{:.2f}"),
            (f"gamma {gamma_key} (%), lowest", f"gpr_{gamma_key}", np.nanmin, False, "{:.1f}")]
    for label, key, fn, absolute, fmt in spec:
        lines.append(f"| {label} | {cell(groups[True], key, fn, absolute, fmt)} "
                     f"| {cell(groups[False], key, fn, absolute, fmt)} |")
    energies = sorted({float(r["energy"]) for r in rows})
    surfaces = sorted({float(r["surface_to_iso_mm"]) for r in rows})
    head = (f"Water, {len(rows)} fresh MCsquare beamlets ({rows[0]['num_primaries']:.0e} primaries) at "
            f"{', '.join(f'{s:g}' for s in surfaces)} mm surface-to-isocenter (not used in the fit), "
            f"energies {', '.join(f'{e:g}' for e in energies)} MeV. "
            "Per-run numbers and IDD/width figures: `validation/`.")
    return head + "\n\n" + "\n".join(lines)


def write_model_card(model_dir: Path, validation_rows: Optional[Sequence[dict]] = None,
                     gamma_key: str = "g3_3_10") -> Path:
    """Write ``README.md`` into ``model_dir`` from its ``commissioning.json``."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "commissioning.json").read_text())
    prov = meta.get("provenance", {})
    com = meta.get("commissioning", {})
    energies = [float(e) for e in meta.get("nodes", {})]
    build = prov.get("dosecuda_build") or {}
    dose_unit = meta.get("dose_unit", "mcsquare_per_weight_1000")
    unit_text = {
        "gy_mm2_per_mu": ("`idd` is Gy mm^2 per MU: a spot of weight `w` MU gives Gy for `w` MU of that "
                          "beamlet, with MCsquare's own protons per MU from the BDL (`ProtonsMU`, interpolated "
                          "and extrapolated linearly as MCsquare does). This is DoseCUDA's convention, so DICOM "
                          "plan MU can be used directly."),
        "mcsquare_per_weight_1000": ("`idd` is on MCsquare's scale: a spot of weight 1000 reproduces MCsquare's "
                                     "eV/g per simulated proton (weights are *not* MU). Convert to Gy per MU "
                                     "with `src.utils.unit_conversions.mcsquare_to_gy_per_mu`."),
    }.get(dose_unit, "unknown")
    weights_text = "Gy for mu MU per spot" if dose_unit == "gy_mm2_per_mu" else "not MU for this model; see Dose unit"
    e_ref = meta.get("rlsp_reference_energy_mev", "?")
    ssds = ", ".join(f"{s:g}" for s in com.get("surface_to_iso_mm", []))
    e_lo, e_hi = min(energies, default=float("nan")), max(energies, default=float("nan"))
    accuracy = (_accuracy(validation_rows, gamma_key) if validation_rows
                else "Not validated yet: run the `validate` stage of `commission_dosecuda.py`.")
    text = f"""# DoseCUDA pencil-beam model: `{prov.get('bdl_file', '?')}` via MCsquare

Proton pencil-beam (PBA) model of **our beam**: the MCsquare beam data library
`{prov.get('bdl_file', '?')}` (sha256 `{prov.get('bdl_sha256', '?')[:16]}...`), with MCsquare's
`{prov.get('scanner', '?')}` scanner and Geant4 stopping powers. Every table was derived from
MCsquare simulations of that beam in water, not from DoseCUDA's shipped (Hitachi) model.
Built by `scripts/pencil_beam/commission_dosecuda.py` in adota; method and limitations in
adota's `docs/pencil_beam.md`.

## Contents (DoseCUDA's lookup-table layout)

| File | Meaning |
|---|---|
| `machine_geometry.csv` | VSADX / VSADY: scanning magnet to isocenter, mm (from the BDL) |
| `HU_RLSP.csv` | HU -> relative linear stopping power, one row per HU, at {e_ref} MeV |
| `rangeshifter_labels.csv` | one entry: no range shifter (`0` -> `RS0`) |
| `RS0/energies.csv` | energy nodes and the in-air spot sigma at the isocenter (mm) |
| `RS0/energy_NNN.csv` | `energy, R80, A, B, C` (sigma_air = A d^2 + B d + C), then `depth, sigma_mcs, idd` |
| `commissioning.json` | provenance, settings, per-node fit diagnostics |
| `validation/` | measured accuracy against MCsquare (after the `validate` stage) |

- **Energies:** {len(energies)} nodes, {e_lo:g}-{e_hi:g} MeV.
- **Depths** are water-equivalent, mm, on a 400-point grid per energy.
- **Dose unit ({dose_unit}):** {unit_text}

## Using it

**From adota** (any energy inside the node span, interpolated in range-scaled depth):

```python
from src.pencil_beam.engine import PencilBeamEngine
eng = PencilBeamEngine("{prov.get('dosecuda_kernels', '<dose_kernels.so>')}", "{model_dir}")
result = eng.dose(ct_hu_image, energy_mev, spots_xy, iso_mc)              # MCsquare scale (eV/g per proton)
result = eng.dose(ct_hu_image, energy_mev, spots_xy, iso_mc, weights=mu)  # {weights_text}
```

`src.pencil_beam.beam_model.PencilBeamModel.read(...)` gives the tables themselves.

**From DoseCUDA directly:** copy this directory to `DoseCUDA/lookuptables/protons/<NAME>/` and
use `IMPTPlan(machine_name="<NAME>")`. DoseCUDA reads DICOM RT Ion plans in patient coordinates.
It only accepts energies equal to a node, so plans with other energies must first be re-binned,
or the model rebuilt at the plan's energies with `PencilBeamModel.at_energy`.

## Measured accuracy

{accuracy}

## Known limitations

- Rotationally symmetric spot: the BDL spot is not (x and y widths differ by up to ~0.5 mm in air);
  the model uses their mean.
- DoseCUDA's nuclear halo is a fixed analytic form; the primary width was fitted given that halo.
- Pencil-beam physics: one central-axis water-equivalent depth per ray, so lateral heterogeneities
  (a beamlet straddling a tissue/air or bone edge) are not resolved the way MC resolves them.

## Provenance

- MCsquare `{prov.get('mcsquare_install', '?')}` (sha256 `{prov.get('mcsquare_sha256', '?')[:16]}...`)
- DoseCUDA kernels: commit `{build.get('dosecuda_commit', '?')}`, built with {build.get('nvcc', '?')}
- adota commit `{prov.get('adota_commit', '?')}`
  (uncommitted changes: {prov.get('adota_uncommitted_changes', '?')})
- Commissioning: {len(com.get('surface_to_iso_mm', []))} source-to-surface distances
  ({ssds} mm), {com.get('num_primaries', 0):.0e} primaries per run;
  MC runs in `{meta.get('runs_dir', '?')}`.
"""
    path = model_dir / "README.md"
    path.write_text(text)
    return path
