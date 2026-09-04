# MCsquare engine (independent install)

The MCsquare Monte Carlo proton dose engine used by adota's MC-generation code
(`src/mc_generation/`, `scripts/mc/`) lives in an **independent, persistent home**,
decoupled from the obsolete `datagenerator` repo:

```
/home/mstryja/tools/mcsquare
```

It is a self-contained, statically-linked binary plus data (~53 MB): `MCsquare_linux`,
`BDL/`, `Materials/`, `Scanners/` (HU conversion tables + phantoms), the `config.txt`
template, `Sample_input_data/`, and `src/`. It is **never committed to git** and
**never placed inside the adota repo**; adota references it by absolute path from the
YAML config (`mcsquare_install`).

## Provisioning

```
scripts/mc/provision_mcsquare.sh [SOURCE_DIR] [DEST_DIR]
```

Idempotent. Copies the engine to the independent home (default
`/home/mstryja/tools/mcsquare`) and runs the bundled sample at low primaries from a
scratch working dir as a smoke test. The original vendor install it was copied from is
`/home/mstryja/projects/datagenerator/MCsquare` (kept intact during the migration).

## Run contract (the working-dir strategy)

MCsquare resolves `Materials/`, `Scanners/`, and `BDL/` relative to the **current working
directory**, and writes into `Output_Directory`. The runner therefore uses a
**per-run working directory on `/scratch`** (never in the repo, never in `/home/tools`):

```
/scratch/<user>/mc_work/<run-id>/
    Materials -> /home/mstryja/tools/mcsquare/Materials   (symlink)
    Scanners  -> /home/mstryja/tools/mcsquare/Scanners    (symlink)
    BDL       -> /home/mstryja/tools/mcsquare/BDL         (symlink)
    CT.mhd, CT.raw            (input, written per run)
    PlanPencil.txt            (single-beamlet plan, written per run)
    config.txt                (written per run)
    Outputs/Dose.mhd, ...     (MCsquare output)
```

The engine binary/data stay in `/home/tools` (symlinked in, not copied per run); inputs
and outputs live on `/scratch`.

## Key config fields (written per run)

`CT_File`, `BDL_Plan_File` (the PlanPencil), `BDL_Machine_Parameter_File`
(`BDL/hptc_beam_model_rsnone.txt` — the beam model the DoTA training set used),
`HU_Density_Conversion_File` / `HU_Material_Conversion_File`
(`Scanners/default/...`), `Output_Directory`, `Dose_MHD_Output True`,
`Num_Primaries`, `Num_Threads`, `RNG_Seed`, `E_Cut_Pro`, `D_Max`, `Epsilon_Max`,
`Te_Min`, and `Compute_stat_uncertainty` for the per-voxel uncertainty map.

## Verified

Relocated engine ran the bundled sample (1e5 primaries, 8 threads) from a `/scratch`
working dir in ~4.4 s, producing `Outputs/Dose.mhd`. See
`scripts/mc/provision_mcsquare.sh` for the reproducible check.
