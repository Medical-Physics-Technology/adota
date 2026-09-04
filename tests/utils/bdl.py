"""Synthetic MCsquare beam-data-library text for the beamlet tests.

Eight beamlet test modules need a small BDL file. They differ only in their
geometry distances and energy rows, so the shared shape lives here rather
than being copied into each module (TESTS.md: shared test code belongs in an
importable module, not duplicated across test files).
"""

from __future__ import annotations

# The BDL column header is a fixed-format line from the MCsquare beam data
# library; it must not be reflowed, so it is written once here rather than
# duplicated (at over 120 characters) across eight test modules.
BDL_TABLE_HEADER = (
    "NominalEnergy MeanEnergy EnergySpread ProtonsMU Weight1 "
    "SpotSize1x Divergence1x Correlation1x SpotSize1y Divergence1y Correlation1y"
)


def build_bdl_text(
    *,
    nozzle_isocenter: float = 400.0,
    smx: float = 2000.0,
    smy: float = 2500.0,
    energy_rows: tuple[str, ...] = (
        "100.0 100.0 1.0 1000.0 1.0 4.0 0.003 0.5 3.0 0.004 0.6",
        "200.0 200.0 0.5 2000.0 1.0 3.0 0.002 0.3 2.0 0.003 0.4",
    ),
    preamble: str = "",
) -> str:
    """Render a synthetic MCsquare beam-data-library file.

    ``preamble`` inserts the optional ``--synthetic beam model--`` / energy-count
    block that the parser tolerates; ``energy_rows`` are whitespace-separated
    rows matching :data:`BDL_TABLE_HEADER`.
    """
    lines = []
    if preamble:
        lines.extend([preamble, ""])
    lines += [
        "Nozzle exit to Isocenter distance",
        f"{nozzle_isocenter}",
        "SMX to Isocenter distance",
        f"{smx}",
        "SMY to Isocenter distance",
        f"{smy}",
        BDL_TABLE_HEADER,
        *energy_rows,
    ]
    return "\n".join(lines) + "\n"
