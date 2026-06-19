"""Stage 2: in-process batched ADoTA inference over extracted beamlets.

Reuses the trusted per-spot loader/saver (``get_single_record_no_gt`` /
``save_prediction`` in :mod:`src.loaders.dir_based`), so the trilinear
down/up-sampling and de-normalization are exactly the training/evaluation path.
Each spot's input is the **two-channel** ``(CT, flux)`` volume: the loader reads
both ``{id}_ct.npy`` and ``{id}_flux.npy``, normalizes them, permutes to
``(D, H, W)``, trilinear-resizes ``(320, 60, 60) -> (160, 30, 30)``, and
concatenates them on the channel axis. The pipeline then:

* batches the records and runs ``model(x, e)`` on the device once per batch,
* per spot, ``save_prediction`` upsamples ``(160, 30, 30) -> (320, 60, 60)``,
  de-normalizes with the dose scale, and writes ``{id}_ds_pred.npy``.

Batching (default 56, as in the original script) is the speed lever; the model
runs in ``eval``/``no_grad`` so a sample's output is independent of its batch.

**Scale:** ``DoTA_v3_grid_search_v11`` is the wide-range (Lungs+Pelvis) model, so
inference defaults to :data:`src.adota.config.DEFAULT_SCALE` (max_energy=270),
not the low-range default baked into ``dir_based``. Pass an explicit scale to
override.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import torch

from src.adota.config import DEFAULT_SCALE
from src.loaders.dir_based import get_single_record_no_gt, save_prediction

logger = logging.getLogger(__name__)

__all__ = ["InferenceConfig", "discover_spot_ids", "run_inference"]


@dataclass
class InferenceConfig:
    """Configuration for :func:`run_inference`.

    Attributes:
        batch_size: Number of spots per forward pass.
        normalize_flux: Min-max normalize the flux channel per crop (training path).
        downsampling_method: ``"interpolation"`` (trilinear) or ``"avg_pooling"``.
        scale: Normalization scale; defaults to the wide-range
            :data:`src.adota.config.DEFAULT_SCALE`.
    """

    batch_size: int = 56
    normalize_flux: bool = True
    downsampling_method: str = "interpolation"
    scale: dict = field(default_factory=lambda: dict(DEFAULT_SCALE))
    grid_factor: int = 1
    """Field-level resampling factor; must match the extraction. 1 = the crop is
    1mm ``(60,60,320)`` and is resized to the model grid, then the prediction is
    up-sampled to the ROI for deposit (byte-identical). 2 = the crop is already the
    2mm model grid, so the input resize and the prediction up-sample are skipped
    (the prediction is saved at ``(160,30,30)`` for the 2mm deposit/de-rotate)."""


def discover_spot_ids(beamlets_dir: Path) -> List[str]:
    """Return ids of spots with a **complete** input set, sorted.

    A spot is usable for inference only if all three per-spot files exist:
    ``{id}_ct.npy``, ``{id}_flux.npy`` (both are model input channels) and
    ``{id}_sim_res.json`` (energy / metadata). Ids missing any of these are
    skipped with a warning, so a partial extraction cannot silently feed the
    model an incomplete input.

    Args:
        beamlets_dir: Directory of extracted per-spot files.

    Returns:
        Sorted list of complete spot ids.
    """
    directory = Path(beamlets_dir)
    ct_ids = {p.name[: -len("_ct.npy")] for p in directory.glob("*_ct.npy")}
    flux_ids = {p.name[: -len("_flux.npy")] for p in directory.glob("*_flux.npy")}
    meta_ids = {p.name[: -len("_sim_res.json")] for p in directory.glob("*_sim_res.json")}

    complete = ct_ids & flux_ids & meta_ids
    incomplete = (ct_ids | flux_ids | meta_ids) - complete
    if incomplete:
        logger.warning(
            "Skipping %d spot(s) missing ct/flux/sim_res (e.g. %s)",
            len(incomplete),
            sorted(incomplete)[:5],
        )
    return sorted(complete)


def run_inference(
    beamlets_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    config: Optional[InferenceConfig] = None,
) -> dict:
    """Run batched ADoTA inference over all complete beamlets.

    Writes ``{id}_ds_pred.npy`` (de-normalized dose, shape ``(1, 1, 320, 60, 60)``)
    next to the inputs in ``beamlets_dir``.

    Args:
        beamlets_dir: Directory of extracted ``{id}_ct.npy`` / ``{id}_flux.npy`` /
            ``{id}_sim_res.json``.
        model: The loaded ADoTA model (already on ``device``, in eval mode).
        device: Target device.
        config: Inference options.

    Returns:
        A summary dict (spot count, batch count, forward-pass timing).

    Raises:
        FileNotFoundError: If no complete beamlet inputs are found.
    """
    config = config or InferenceConfig()
    beamlets_dir = Path(beamlets_dir)
    spot_ids = discover_spot_ids(beamlets_dir)
    if not spot_ids:
        raise FileNotFoundError(
            f"No complete beamlet inputs (ct+flux+sim_res) found in {beamlets_dir}"
        )

    batches = [
        spot_ids[i : i + config.batch_size]
        for i in range(0, len(spot_ids), config.batch_size)
    ]
    logger.info(
        "Inference: %d spots in %d batch(es) of %d on %s",
        len(spot_ids),
        len(batches),
        config.batch_size,
        device,
    )

    model.eval()
    resize = config.grid_factor == 1  # gf=2 crops are already the model grid
    storage = str(beamlets_dir)
    read_s = 0.0
    downsample_s = 0.0
    forward_s = 0.0
    upsample_s = 0.0
    save_write_s = 0.0
    started = perf_counter()

    for batch in batches:
        # Each record is the 2-channel (CT, flux) input + normalized energy. The
        # loader splits its own time into file read vs down-sample to the ADoTA grid.
        load_timing: dict = {}
        records = [
            get_single_record_no_gt(
                spot_id,
                storage,
                scale=config.scale,
                normalize_flux=config.normalize_flux,
                downsampling_method=config.downsampling_method,
                timing=load_timing,
                device=device,  # CT/flux moved to the device -> resize runs there
                resize=resize,
            )
            for spot_id in batch
        ]
        x = torch.stack([r[0] for r in records]).to(device)
        energy = torch.stack([r[1] for r in records]).to(device)
        read_s += load_timing.get("read", 0.0)
        downsample_s += load_timing.get("downsample", 0.0)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_t = perf_counter()
        with torch.no_grad():
            pred = model(x, energy)[0]  # model returns (dose, attention)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_s += perf_counter() - forward_t

        # The saver splits its time into the up-sample back to the beamlet ROI
        # grid (320, 60, 60) vs the de-normalize + .npy write.
        save_timing: dict = {}
        for i, spot_id in enumerate(batch):
            save_prediction(
                pred[i].unsqueeze(0),
                spot_id,
                storage,
                scale=config.scale,
                timing=save_timing,
                upsample=resize,  # gf=2 keeps (160,30,30) for the 2mm deposit
            )
        upsample_s += save_timing.get("upsample", 0.0)
        save_write_s += save_timing.get("write", 0.0)

    elapsed = perf_counter() - started
    n = len(spot_ids)
    load_s = read_s + downsample_s
    save_s = upsample_s + save_write_s
    summary = {
        "n_spots": n,
        "n_batches": len(batches),
        "batch_size": config.batch_size,
        "load_s": load_s,
        "read_s": read_s,
        "downsample_s": downsample_s,
        "forward_s": forward_s,
        "save_s": save_s,
        "upsample_s": upsample_s,
        "save_write_s": save_write_s,
        "ms_per_spot_load": load_s / n * 1000.0,
        "ms_per_spot_read": read_s / n * 1000.0,
        "ms_per_spot_downsample": downsample_s / n * 1000.0,
        "ms_per_spot_forward": forward_s / n * 1000.0,
        "ms_per_spot_save": save_s / n * 1000.0,
        "ms_per_spot_upsample": upsample_s / n * 1000.0,
        "ms_per_spot_save_write": save_write_s / n * 1000.0,
        "elapsed_s": elapsed,
        "device": str(device),
    }
    logger.info(
        "Inference complete: %d spots | read %.1fs, downsample %.1fs, forward %.1fs "
        "(%.2f ms/spot), upsample %.1fs, write %.1fs | total %.1fs",
        n,
        read_s,
        downsample_s,
        forward_s,
        summary["ms_per_spot_forward"],
        upsample_s,
        save_write_s,
        elapsed,
    )
    return summary
