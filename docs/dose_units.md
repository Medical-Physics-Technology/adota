# Dose units

One page for the units every beamlet and plan dose in adota is in, and for converting
them. The functions are in `src/utils/unit_conversions.py` (beamlets) and
`src/beamlets/dose_scaling.py` (plans).

## What is on disk

| Array | Unit |
|---|---|
| MCsquare `Dose.mhd`, `Dose_Beamlet_*.mhd`; the generator's `{id}_ds.npy`; the HDF5 `dose` | **eV/g per simulated proton** |
| ADoTA predictions (`{id}_ds_pred.npy`, after `inverse_minmax`) | the same, since ADoTA learns MCsquare's doses |
| Pencil beam at `PencilBeamEngine.mcsquare_spot_weight(E)` (all comparisons; `{id}_ds_pba.npy`) | the same |
| Pencil beam with `weights` in MU, model `dose_unit: gy_mm2_per_mu` (`_v2`) | Gy |

Where this comes from, in MCsquare's source:
- `PostProcess_Scoring` divides the scored energy by the number of simulated
  primaries, with normalisation 1.0, so a plan's spot weights do not scale `Dose.mhd`.
- MCsquare converts to Gy only for its DVHs (`compute_simulation.c`):
  `Gy = dose x 1.602176e-19 x 1000 x delivered_protons x fractions`.
- For a UPenn BDL (ours), `delivered_protons = sum(MU x ProtonsMU(E))`. `ProtonsMU`
  is interpolated linearly in nominal energy and *extrapolated* linearly from the
  outermost two nodes (`data_beam_model.c`).
- Non-UPenn BDLs use an ionisation-chamber formula instead, about 30x larger for ours;
  `mcsquare_to_gy_per_mu` refuses them.

## Converting

- **One beamlet, Gy per MU** (the reporting convention for beamlets):
  `mcsquare_to_gy_per_mu(dose, energy_mev, bdl)` = dose x 1.602176e-16 x
  `mcsquare_protons_per_mu(E)`. A typical 102.6 MeV beamlet peaks at about 8 mGy per MU.
- **One beamlet, Gy per proton:** `mcsquare_to_gy_per_proton(dose)`.
- **A plan:** `src.beamlets.dose_scaling.load_dose_gy` / `dose_to_gy_factor`.
  - These take protons per MU from `BeamDataLibrary.compute_mu_to_protons`, which
    *clamps* outside the BDL table (the OpenTPS convention) where MCsquare
    extrapolates.
  - The two agree for every energy up to the last BDL node (244 MeV) and differ by
    1.2% at 250 MeV.

## The legacy `to_gy` (Gy per 1e6 protons)

Until 2026-10-20, `src.utils.unit_conversions.to_gy` multiplied by 1.60218e-10 and
called its input MeV/g. The input is eV/g per proton, so the result was the dose of
exactly **10^6 protons**, not a physical dose.

The training and evaluation RMSE values reported as `rmse_gy`, and the beamlet
figures labelled Gy, are in that unit. They are kept reproducible under the explicit
name `legacy_to_gy_per_1e6_protons`, which every existing call site now uses. `to_gy`
remains as an alias that warns. To turn a legacy number into Gy per MU, multiply it
by `protons_per_MU(E) / 1e6` (2.2 at 70 MeV, 2.9 at 102.6 MeV, 3.5 at 135 MeV,
4.6 at 200 MeV).

## Energies above 250 MeV

MCsquare's nuclear data ends at 250 MeV, so there is no valid dose above it.

- MC generation and the active-learning candidate space refuse such energies
  (`check_energies_simulable`).
- Training and evaluation record lists drop them (`src.datasets.energy_cut`, config
  key `max_energy_mev`, default 250; `null` reproduces older runs).
- See `docs/pencil_beam.md` for how the limit was found.
