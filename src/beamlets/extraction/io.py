"""Output-tree preparation and per-spot writes.

:func:`_prepare_output_dir` creates (or, with ``overwrite``, clears) the
``adota_beamlets/`` tree; :func:`_save_spot` writes one spot's ``_ct.npy``,
``_flux.npy`` and ``_sim_res.json``. Keeping both here gives the extraction one
JSON encoding policy and one place to change the on-disk layout.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

from src.utils.serialization import NumpyEncoder

logger = logging.getLogger(__name__)

def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create the output dir; on overwrite WIPE it first so no stale files survive.

    Accumulation reads every ``*_sim_res.json`` in the directory, so per-spot
    files left over from a previous run (e.g. a different grid / spot subset)
    must be removed -- not just written over -- or they would be mixed into the
    accumulated dose.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} is not empty; pass overwrite=True "
                "to extract into it anyway."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _save_spot(
    output_dir: Path,
    spot_id_str: str,
    cropped_ct: np.ndarray,
    flux: np.ndarray,
    sim_res: dict,
) -> None:
    """Write the CT crop, flux projection and metadata for one spot."""
    np.save(output_dir / f"{spot_id_str}_ct.npy", cropped_ct)
    np.save(output_dir / f"{spot_id_str}_flux.npy", flux)
    (output_dir / f"{spot_id_str}_sim_res.json").write_text(
        json.dumps(sim_res, indent=4, cls=NumpyEncoder)
    )
