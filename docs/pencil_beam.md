# DoseCUDA pencil beam, commissioned against MCsquare

adota compares three dose engines for the same proton beamlet: the MCsquare Monte
Carlo that produced the training data, ADoTA, and an analytical **pencil-beam
algorithm (PBA)**. The PBA is
[DoseCUDA](https://github.com/jhu-som-radiation-oncology/DoseCUDA) (JHU), a GPU
pencil beam whose proton model is entirely table-driven. This page covers how it is
installed, what "commissioning" means for it here, and the decisions behind it. The
scripts are documented in
[scripts/docs/commission_dosecuda.md](../scripts/docs/commission_dosecuda.md) and
[scripts/docs/compare_engines.md](../scripts/docs/compare_engines.md).

## Source, version, licence and citation

- **Source:** [jhu-som-radiation-oncology/DoseCUDA](https://github.com/jhu-som-radiation-oncology/DoseCUDA), developed at Johns
  Hopkins University School of Medicine, Department of Radiation Oncology.
- **Version:** commit
  [`0d4db1f4f736`](https://github.com/jhu-som-radiation-oncology/DoseCUDA/commit/0d4db1f4f736361b1c93ada457148c6d8fe9e896). It is recorded in
  every build's `BUILD_INFO.json` and in every commissioned model's model card, so a
  result can always be traced to the DoseCUDA code that produced it.
- **What adota uses:** only DoseCUDA's CUDA kernels (the compiled `dose_kernels`
  extension), built **unmodified** from a `git archive` of that commit by
  `scripts/pencil_beam/provision_dosecuda.sh`. DoseCUDA's Python package is never
  imported; the beam model, HU -> RLSP and geometry are adota's own
  (`src/pencil_beam/`).
- **Licence:** DoseCUDA is distributed under the GNU General Public License,
  version 2. adota does not include or redistribute any of its source code; the
  kernels are compiled locally from the user's own clone.
- **Citation:** the DoseCUDA authors ask that work using it cite their paper. Any
  result, figure or publication that uses the pencil beam must cite:

  > Bhattacharya M, Reamy C, Li H, Lee J, Hrinivich WT. A Python package for fast
  > GPU-based proton pencil beam dose calculation. *Journal of Applied Clinical
  > Medical Physics.* 2025:e70093. [doi:10.1002/acm2.70093](https://doi.org/10.1002/acm2.70093)

  ```bibtex
  @article{bhattacharya2025dosecuda,
    author  = {Bhattacharya, Mahasweta and Reamy, Clayton and Li, Heng and Lee, Junghoon and Hrinivich, William T.},
    title   = {A Python package for fast GPU-based proton pencil beam dose calculation},
    journal = {Journal of Applied Clinical Medical Physics},
    year    = {2025},
    pages   = {e70093},
    doi     = {10.1002/acm2.70093},
    note    = {Code: https://github.com/jhu-som-radiation-oncology/DoseCUDA}
  }
  ```

## Install: only the CUDA kernels, built for adota

```
scripts/pencil_beam/provision_dosecuda.sh [SOURCE_REPO] [DEST_DIR] [GIT_REF]
```

DoseCUDA is an external, uncommitted install, like MCsquare. The script builds only
the compiled `dose_kernels` extension from a DoseCUDA clone
(default `/home/mstryja/projects/DoseCUDA`, `git archive` of `HEAD`, so local edits in
the clone are ignored) into `/home/mstryja/tools/dosecuda/<commit>/`. It points
`current` at that build and writes `BUILD_INFO.json` (commit, Python, numpy, nvcc,
CUDA architecture). A smoke test ray-traces a water cube on the GPU. YAML configs
reference `.../current/dose_kernels.so` through the key `dosecuda_kernels`.

Why not `pip install DoseCUDA` into the venv:

- DoseCUDA pins `numpy < 2` while adota runs numpy 2, and an extension compiled
  against numpy 1 cannot be imported under numpy 2.
- A package outside the lockfile would be removed by the next exact `uv sync`.
- DoseCUDA's Python layer reads its tables through `pkg_resources` from its own
  install directory, so a model could not live anywhere else.

adota therefore loads the extension by path (`src/pencil_beam/kernels.py`) and does
everything else itself.

Build requirements: nvcc 12.3 (NVIDIA HPC SDK) with `g++-12`, because nvcc 12.3
rejects gcc 13, and cmake through `uv tool run`. The static CUDA runtime and native
`sm_86` code run on the 535.x driver.

## What the kernel needs, and where each piece comes from

DoseCUDA's proton dose for a spot of weight `w`, at a voxel with water-equivalent
depth `z_w` and squared distance `r^2` from the spot axis, is

```
D = w * IDD(z_w) * [ (1 - h) G(r; s_t) + h G(r; hypot(s_t, s_h)) ]
s_t = sigma_air(d) + sigma_mcs(z_w),   d = distance_to_source - z_w + 0.7 R80
sigma_air(d) = A d^2 + B d + C
```

`G` is a normalised 2D Gaussian. The nuclear halo weight `h` and width `s_h` are
hard-coded in the kernel as functions of `z_w` and R80.

| Kernel input | Source in adota |
|---|---|
| `VSADX`, `VSADY` | the BDL's `SMX/SMY to Isocenter distance` (2014.9 / 2584.1 mm) |
| HU -> RLSP | MCsquare's own scanner and materials (`src/pencil_beam/rlsp.py`) |
| `IDD`, `sigma_mcs`, `A,B,C`, `R80` per energy | MCsquare water-phantom beamlets (`src/pencil_beam/commissioning.py`) |
| IDD unit | `dose_unit`: `gy_mm2_per_mu` (model `_v2`; weight in MU gives Gy, with MCsquare's protons per MU) or `mcsquare_per_weight_1000` (`_v1`: weight 1000 = MCsquare's eV/g per proton). See [dose_units.md](dose_units.md). |
| beam geometry | the MC beamlet frame, mapped once (`src/pencil_beam/engine.py`) |

### HU -> RLSP

MCsquare assigns density by linear interpolation of `HU_Density_Conversion.txt` and
material by a step lookup in `HU_Material_Conversion.txt`: the last row whose HU is
strictly below the voxel's. Stopping powers come from the Geant4 tables
(`DB_STOP_POW = SP_GEANT4`). RLSP is density times the material's mass stopping
power relative to water at 100 MeV. It is written at every integer HU, so the
kernel's linear interpolation reproduces the material steps.

HU 0 is `Schneider_AT_AG_SI4` at 1.000 g/cm3, with **RLSP 1.017**, not water. A
"water" phantom in MCsquare therefore ranges 1.7% short of water, and commissioning
depths are converted with this same curve. The ratio varies by 0.3% for soft tissue
and 0.6% for bone between 70 and 230 MeV.

### Geometry

MC beamlets are simulated at the canonical gantry 90, with the beam along +x from
the CT's `x = 0` face and the isocenter in grid-corner coordinates. DoseCUDA works in
DICOM coordinates with `origin` at voxel 0's centre and adds 180 degrees to the
gantry internally. The mapping is therefore isocenter `origin - spacing/2 + iso_mc`,
DoseCUDA gantry 270, and the **spot passed through unchanged**: MC spot `(X, Y)`
lands at image `(y, z)` offset `(-X, +Y)` in both engines, with `X` diverging about
`d_smx` and `Y` about `d_smy`. A GPU test checks this to 0.1 mm with off-axis spots.

### Commissioning recipe (DoseCUDA's, with MCsquare as the reference)

1. One central spot per energy node into a 120 x 120 mm water box with 20 mm of air
   in front only, at four source-to-surface distances. The surface sits 100, 175,
   250 and 325 mm upstream of the isocenter.
2. Per depth slice: IDD (the lateral sum, per unit spot weight, so the PBA lands on
   the MC dose scale) and the lateral width.
3. R80 from the IDD. At `z_ref = 0.7 R80`, the per-SSD width is fitted with a
   quadratic in the kernel's `d`, which gives `sigma_air`.
4. `sigma_mcs(z) = sigma(z) - sigma(z_ref)` and IDD are averaged over the SSDs on a
   400-point depth grid per energy.

Two deviations from DoseCUDA's own script, both measured:

- **Width in the kernel's own form.** A single-Gaussian width fed to the kernel
  comes out about 0.1 mm too wide, with a 3-8% low central-axis dose at 120 MeV,
  because the kernel then adds its halo on top. The lateral profile is therefore
  fitted as `(1-h) G(s) + h G(hypot(s, s_h))`, with the kernel's `h, s_h`. After
  that, the width at a held-out SSD agrees with MC to 0.01 mm.
- **A per-energy depth grid.** DoseCUDA compiles a table length of 400, but looks
  depths up by value. Each energy therefore carries its own grid from 0 to 1.3 R80.
  That covers the highest energies (R80 ~ 37 cm at 250 MeV) without patching the kernel.

**Continuous energies.** ADoTA's beamlets span 70-270 MeV continuously (the pencil beam
covers 70-250 MeV; see the next section), while
DoseCUDA only knows tabulated energies. Nodes are therefore commissioned every
5 MeV. `PencilBeamModel.at_energy` builds any energy in between by blending its two
neighbours in range-scaled depth (`z / R80`), with R80 interpolated log-log.

### MCsquare above 250 MeV

MCsquare's ICRU nuclear cross-section tables (`Materials/<element>/ICRU_Nuclear_elastic.dat`
and `_inelastic.dat`) end at **250 MeV**. `compute_nuclear_interaction.c` interpolates
between table entries `index` and `index + 1`; above 250 MeV `index` is the last entry,
so it reads past the end of the arrays.

- **~250-258 MeV:** it runs, but the nuclear cross sections are interpolated with whatever
  memory follows the table, so the dose is undefined.
- **From ~259 MeV:** it crashes with a segmentation fault (exit -11) right after
  "Simulation started". This was measured on 2026-10-20 at 260, 265 and 270 MeV, in every
  geometry tried, at every seed and primary count. It runs normally with nuclear
  interactions switched off.

What counts is the mean proton energy, which the BDL maps from the nominal energy:
nominal 250 MeV is 249.1 MeV (safe), nominal 255 MeV is 254.1 MeV (not). The datagenerator
install is byte-identical, binary and tables both.

Consequences:

- The pencil beam is commissioned from 70 to 250 MeV. Above that there is no valid MC
  reference, and both scripts refuse such energies.
- `MCSquareRunner` logs a warning for any beamlet above 250 MeV. The limit is
  `src.mc_generation.mcsquare_runner.MCSQUARE_NUCLEAR_DATA_MAX_MEV`.
- **Training data:** the v3 HDF5 holds 853 beamlets above 250 MeV, 422 of them above
  258.4 MeV, which this engine cannot produce at all with nuclear interactions on.
  Since 2026-10-20, training, active-learning splits and evaluation drop every beamlet
  above 250 MeV (`src.datasets.energy_cut`, config `max_energy_mev`, default 250), and
  MC generation and the candidate space refuse such energies. See
  [dose_units.md](dose_units.md).

### MCsquare nozzle transport: a geometry trap

MCsquare starts each proton at the nozzle exit (420 mm upstream of the isocenter)
and moves it through vacuum to the CT. When the nozzle exit lies more than about
340 mm upstream of the CT face, the beam enters late by the excess, and the skipped
material is simply missing. This was measured at 120 MeV: 60, 35 and 10 mm late at
400, 375 and 350 mm. Generated datasets are unaffected, because their isocenter is
the grid centre and the nozzle sits 160-220 mm upstream of the face. Commissioning
phantoms are guarded: `air_front_mm + surface_to_iso_mm` must stay in [120, 420] mm.
**Any new MC geometry with the isocenter near the CT entrance hits this.**

## Known limitations of the kernel (not of the commissioning)

- **Rotationally symmetric spot.** The BDL spot is not symmetric: MC's x and y
  widths differ by up to ~0.5 mm in air and 0.1-0.2 mm at mid-range. The kernel
  uses their mean.
- **Fixed nuclear halo.** Its weight and width are DoseCUDA's analytic
  parameterisation, not MCsquare's. Commissioning matches the primary profile
  given that halo.
- **Central-axis WET.** The kernel smooths WET over a 10 mm disk, but cannot split a
  beamlet that straddles a lateral density interface. MC then shows a distal
  shoulder or a double peak that the PBA smears into one deeper, broader peak. This
  is the effect the engine comparison is designed to expose. Global 3%/3mm gamma can
  hide it, so read it together with the range metrics.
