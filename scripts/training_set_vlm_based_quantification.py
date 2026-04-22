"""
Training Set Analysis – VLM-based Difficulty Quantification

Uses a locally-hosted Vision-Language Model to score beamlet difficulty
from pre-inference input data (CT, flux, energy).  Ground-truth dose
is used only for:
  1. Surrogate Bragg Peak range estimation (to be replaced later).
  2. GPR computation as the evaluation label.

The VLM receives a rendered multi-panel image (CT slices, segmentation,
Sobel edges) and text metadata, and returns a structured difficulty
score in [1, 10].

Uses HuggingFace Transformers to load and run the VLM locally on GPU.
"""

import csv
import json
import logging
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
import yaml
from scipy.ndimage import sobel as ndimage_sobel
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.config import (
    DEFAULT_GAMMA_PARAMS,
    DEFAULT_SCALE,
    denormalize_energy,
    get_device,
    load_yaml_config,
    setup_logging,
    setup_run_directory,
)
from src.adota.models import DoTA3D_v3
from src.adota.utils import load_model
from src.figures.ct_visualizations import (
    plot_ct_with_segmentation,
    segment_hu,
    smooth_ct,
)
from src.loaders.generator import H5PYGenerator
from src.loaders.utils import validate_inputs
from src.metrics.gamma_pass_rate import gamma_index_torch
from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import to_gy

logger = logging.getLogger(__name__)
app = typer.Typer(help="VLM-based difficulty quantification of training set beamlets")


# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_VLM_PROMPT = """\
You are an expert in proton therapy physics.  You are shown a \
multi-panel image of a single proton beamlet passing through a CT volume.

The panels show (from top to bottom):
- Row 1: Raw CT (Hounsfield Units), axial and sagittal views
- Row 2: Gaussian-smoothed CT
- Row 3: 7-class tissue segmentation (Air, Lung, Fat, Soft tissue, \
Contrast/blood, Cancellous bone, Cortical bone)
- Row 4: Sobel gradient magnitude on raw CT (edge map)
- Row 5: Sobel gradient magnitude on smoothed CT

Beam metadata:
- Initial energy: {energy_mev:.1f} MeV
- BP depth range: [{bp_min_mm:.1f}, {bp_max_mm:.1f}] mm
- Number of density regions in BP zone: {n_density_regions}
- Voxel spacing: {voxel_spacing:.1f} mm (isotropic)

Rate how DIFFICULT it would be for a deep-learning dose prediction \
model to accurately predict the dose distribution for this beamlet.

Difficulty drivers include: tissue heterogeneity along the beam path, \
bone-tissue or air-tissue interfaces near the Bragg peak, complex \
anatomy, and lateral density variations.

Respond ONLY with valid JSON:
{{"difficulty": <int 1-10>, "confidence": <float 0-1>, "reason": "<brief explanation>"}}
"""


from src.schemas.configs import VLMConfig
from src.schemas.results import VLMResult

# ── Helpers (shared with advanced metrics script) ───────────────────────────


def estimate_bp_range(
    ct_grid: np.ndarray,
    dose_grid: np.ndarray,
    proximal_fraction: float = 0.50,
    fall_fraction: float = 0.10,
) -> tuple[float, float]:
    """Estimate BP range from GT IDD (surrogate, to be replaced)."""
    idd = dose_grid.sum(axis=(1, 2))
    idd_max = idd.max()
    if idd_max < 1e-9:
        return 0.0, 0.0

    bp_idx = int(np.argmax(idd))

    # Proximal: walk backward
    z_min = 0.0
    for k in range(bp_idx, -1, -1):
        if idd[k] < proximal_fraction * idd_max:
            z_min = float(k)
            break

    # Distal: walk forward
    n_slices = len(idd)
    z_max = float(n_slices - 1)
    for k in range(bp_idx, n_slices):
        if idd[k] < fall_fraction * idd_max:
            z_max = float(k)
            break

    return z_min, z_max


def analyse_density_regions(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    z_min: float,
    z_max: float,
    flux_threshold_frac: float = 0.10,
) -> int:
    """Count density regions along beam path in BP zone. Returns n_regions."""
    k_start = int(np.ceil(z_min))
    k_end = int(np.floor(z_max))
    if k_end <= k_start:
        return 0

    mean_hu = np.zeros(k_end - k_start + 1)
    for i, k in enumerate(range(k_start, k_end + 1)):
        flux_slice = np.abs(flux[k])
        ct_slice = ct_hu[k]
        f_max = flux_slice.max()
        if f_max < 1e-12:
            mean_hu[i] = ct_slice.mean()
            continue
        mask = flux_slice >= flux_threshold_frac * f_max
        if mask.sum() == 0:
            mean_hu[i] = ct_slice.mean()
            continue
        mean_hu[i] = np.average(ct_slice[mask], weights=flux_slice[mask])

    classes = segment_hu(mean_hu)
    # Count contiguous class regions
    n_regions = 1
    for j in range(1, len(classes)):
        if classes[j] != classes[j - 1]:
            n_regions += 1
    return n_regions


# ── Panel rendering ─────────────────────────────────────────────────────────


def render_review_panel(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    energy_mev: float,
    sample_id: str,
    output_path: Path,
    config: VLMConfig,
    gt_dose: Optional[np.ndarray] = None,
    bp_range_slices: Optional[tuple] = None,
) -> Path:
    """Render a multi-panel review image for VLM consumption.

    Reuses plot_ct_with_segmentation which renders:
    - Raw CT, smoothed CT, segmentation, raw Sobel, smoothed Sobel
    - IDD row (only if gt_dose provided)

    Args:
        ct_hu: Raw CT volume in HU (D, H, W).
        flux: Flux volume (D, H, W).
        energy_mev: Beam energy in MeV.
        sample_id: Unique identifier.
        output_path: Where to save the PNG.
        config: VLM configuration.
        gt_dose: Optional GT dose for IDD rendering.
        bp_range_slices: Optional (z_min, z_max) for IDD annotation.

    Returns:
        Path to the saved panel image.
    """
    # Smooth CT
    ct_smooth = smooth_ct(
        ct_hu, method=config.smoothing_method, sigma=config.smoothing_sigma
    )

    # Sobel on raw CT
    sobel_raw = np.sqrt(
        ndimage_sobel(ct_hu, axis=0) ** 2
        + ndimage_sobel(ct_hu, axis=1) ** 2
        + ndimage_sobel(ct_hu, axis=2) ** 2
    )

    # Sobel on smoothed CT
    sobel_smooth = np.sqrt(
        ndimage_sobel(ct_smooth, axis=0) ** 2
        + ndimage_sobel(ct_smooth, axis=1) ** 2
        + ndimage_sobel(ct_smooth, axis=2) ** 2
    )

    plot_ct_with_segmentation(
        ct_hu=ct_smooth,
        sample_id=sample_id,
        output_path=output_path,
        ct_hu_unsmoothed=ct_hu,
        gt_dose=gt_dose,
        bp_range_slices=bp_range_slices,
        voxel_spacing_mm=config.resolution[0],
        sobel_magnitude=sobel_smooth,
        sobel_magnitude_raw=sobel_raw,
    )

    plt.close("all")
    return output_path


# ── VLM query ───────────────────────────────────────────────────────────────


def _load_vlm(config: VLMConfig) -> tuple:
    """Load the VLM processor and model from HuggingFace.

    Returns:
        (processor, model) ready for inference.
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    vlm_device = (
        torch.device(f"cuda:{config.vlm_device_index}")
        if torch.cuda.is_available() and config.vlm_device_index >= 0
        else torch.device("cpu")
    )

    logger.info(f"Loading VLM: {config.vlm_model} on {vlm_device}")
    processor = AutoProcessor.from_pretrained(config.vlm_model)
    model = LlavaForConditionalGeneration.from_pretrained(
        config.vlm_model,
        torch_dtype=torch.float16 if vlm_device.type == "cuda" else torch.float32,
        device_map=vlm_device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info("VLM loaded successfully")
    return processor, model, vlm_device


def _parse_vlm_response(text: str) -> dict:
    """Extract JSON from VLM response text, handling markdown fences."""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning(f"Could not parse VLM response as JSON: {text[:200]}")
                return {
                    "difficulty": -1,
                    "confidence": 0.0,
                    "reason": f"PARSE_ERROR: {text[:200]}",
                }
        else:
            return {
                "difficulty": -1,
                "confidence": 0.0,
                "reason": f"PARSE_ERROR: {text[:200]}",
            }

    # Validate and clamp
    diff = result.get("difficulty", -1)
    if isinstance(diff, (int, float)):
        result["difficulty"] = max(1, min(10, int(diff)))
    else:
        result["difficulty"] = -1

    conf = result.get("confidence", 0.0)
    if isinstance(conf, (int, float)):
        result["confidence"] = max(0.0, min(1.0, float(conf)))
    else:
        result["confidence"] = 0.0

    result.setdefault("reason", "")
    return result


def query_vlm(
    image_path: Path,
    prompt: str,
    config: VLMConfig,
    processor,
    vlm_model,
    vlm_device: torch.device,
) -> list[dict]:
    """Query a locally-loaded HuggingFace VLM with an image and prompt.

    Args:
        image_path: Path to the review panel PNG.
        prompt: Formatted prompt string.
        config: VLM configuration.
        processor: HuggingFace processor.
        vlm_model: Loaded VLM model.
        vlm_device: Device the VLM is on.

    Returns:
        List of K parsed result dicts, each with keys
        'difficulty', 'confidence', 'reason'.
    """
    from PIL import Image as PILImage

    image = PILImage.open(image_path).convert("RGB")

    # Build conversation in the LLaVA chat format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_input = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(text=text_input, images=image, return_tensors="pt").to(
        vlm_device
    )

    votes: list[dict] = []
    for vote_idx in range(config.vlm_n_votes):
        try:
            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": config.vlm_max_new_tokens,
                    "do_sample": config.vlm_temperature > 0,
                }
                if config.vlm_temperature > 0:
                    generate_kwargs["temperature"] = config.vlm_temperature
                    generate_kwargs["top_p"] = 0.9
                    # Different seed per vote
                    torch.manual_seed(vote_idx * 42 + 1)

                output_ids = vlm_model.generate(**inputs, **generate_kwargs)

            # Decode only newly generated tokens
            generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
            raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]
            parsed = _parse_vlm_response(raw_text)
        except Exception as e:
            logger.error(f"VLM inference failed (vote {vote_idx}): {e}")
            parsed = {"difficulty": -1, "confidence": 0.0, "reason": f"ERROR: {e}"}

        votes.append(parsed)

    return votes


def aggregate_votes(votes: list[dict]) -> dict:
    """Aggregate K votes into a single result using median.

    Args:
        votes: List of parsed VLM responses.

    Returns:
        Dict with 'difficulty', 'confidence', 'reason', 'all_votes'.
    """
    valid_diffs = [v["difficulty"] for v in votes if v["difficulty"] > 0]
    valid_confs = [v["confidence"] for v in votes if v["difficulty"] > 0]

    if not valid_diffs:
        return {
            "difficulty": -1,
            "confidence": 0.0,
            "reason": "ALL_VOTES_FAILED",
            "all_votes": votes,
        }

    median_diff = statistics.median(valid_diffs)
    median_conf = statistics.median(valid_confs)

    # Pick the reason from the vote closest to the median difficulty
    closest_vote = min(votes, key=lambda v: abs(v.get("difficulty", -1) - median_diff))

    return {
        "difficulty": median_diff,
        "confidence": median_conf,
        "reason": closest_vote.get("reason", ""),
        "all_votes": votes,
    }


# ── Per-sample extraction (simplified from advanced metrics) ────────────────


def extract_sample(
    sample_idx: int,
    record_id: str,
    dataset: H5PYGenerator,
    config: VLMConfig,
) -> Optional[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    """Extract CT, flux, energy, and GT dose for one beamlet.

    Returns:
        (ct_hu, flux, energy_mev, gt_dose) or None if skipped.
    """
    scale = config.scale
    x, energy, y = dataset[sample_idx]

    energy_mev = denormalize_energy(energy.item(), scale)

    # Zero-flux guard
    if torch.abs(x[1]).max().item() < 1e-9:
        return None

    # Energy guard
    if energy_mev > config.max_energy_mev:
        return None

    ct_hu = inverse_minmax(x[0].cpu().numpy(), scale["min_ct"], scale["max_ct"])
    flux = x[1].cpu().numpy()
    gt_dose_norm = y.cpu().numpy()
    gt_dose = inverse_minmax(
        gt_dose_norm if gt_dose_norm.ndim == 4 else gt_dose_norm[np.newaxis],
        scale["min_ds"],
        scale["max_ds"],
    ).squeeze()

    return ct_hu, flux, energy_mev, gt_dose


# ── Results I/O ─────────────────────────────────────────────────────────────


RESULT_FIELDNAMES = [
    "sample_id",
    "energy_mev",
    "bp_range_min_mm",
    "bp_range_max_mm",
    "n_density_regions",
    "vlm_difficulty",
    "vlm_confidence",
    "vlm_reason",
    "vlm_votes",
    "gpr_pct",
    "panel_path",
    "query_time_s",
]


def save_results_csv(results: list[VLMResult], output_path: Path) -> None:
    """Save results to CSV."""
    sorted_results = sorted(results, key=lambda r: r.energy_mev)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for r in sorted_results:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "energy_mev": f"{r.energy_mev:.2f}",
                    "bp_range_min_mm": f"{r.bp_range_min_mm:.1f}",
                    "bp_range_max_mm": f"{r.bp_range_max_mm:.1f}",
                    "n_density_regions": r.n_density_regions,
                    "vlm_difficulty": f"{r.vlm_difficulty:.1f}",
                    "vlm_confidence": f"{r.vlm_confidence:.3f}",
                    "vlm_reason": r.vlm_reason,
                    "vlm_votes": r.vlm_votes,
                    "gpr_pct": f"{r.gpr:.2f}",
                    "panel_path": r.panel_path,
                    "query_time_s": f"{r.query_time_s:.2f}",
                }
            )
    logger.info(f"Results saved to {output_path}")


# ── Correlation analysis ────────────────────────────────────────────────────


def generate_correlation_analysis(
    results: list[VLMResult],
    output_dir: Path,
    config: VLMConfig,
) -> None:
    """Generate scatter plots and correlation statistics."""
    valid = [r for r in results if r.vlm_difficulty > 0]
    if len(valid) < 10:
        logger.warning(f"Too few valid results ({len(valid)}) for correlation analysis")
        return

    difficulties = [r.vlm_difficulty for r in valid]
    gprs = [r.gpr for r in valid]
    energies = [r.energy_mev for r in valid]

    # Overall correlation
    sp_rho, sp_p = spearmanr(difficulties, gprs)
    pe_rho, pe_p = pearsonr(difficulties, gprs)
    logger.info(
        f"VLM difficulty vs GPR: Spearman ρ={sp_rho:.4f} (p={sp_p:.2e}), "
        f"Pearson r={pe_rho:.4f} (p={pe_p:.2e})"
    )

    # ── Scatter: VLM difficulty vs GPR ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(difficulties, gprs, c=energies, cmap="viridis", alpha=0.5, s=10)
    plt.colorbar(sc, ax=ax, label="Energy [MeV]")
    ax.set_xlabel("VLM Difficulty Score")
    ax.set_ylabel("Gamma Pass Rate [%]")
    ax.set_title(
        f"VLM Difficulty vs GPR (n={len(valid)})\n"
        f"Spearman ρ={sp_rho:.3f}, Pearson r={pe_rho:.3f}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "vlm_difficulty_vs_gpr.png", dpi=200)
    plt.close(fig)

    # ── Scatter: VLM difficulty vs GPR (no energy colour) ───────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(difficulties, gprs, alpha=0.3, s=10, c="steelblue")
    ax.set_xlabel("VLM Difficulty Score")
    ax.set_ylabel("Gamma Pass Rate [%]")
    ax.set_title(f"VLM Difficulty vs GPR (n={len(valid)})\n" f"Spearman ρ={sp_rho:.3f}")
    fig.tight_layout()
    fig.savefig(output_dir / "vlm_difficulty_vs_gpr_plain.png", dpi=200)
    plt.close(fig)

    # ── Difficulty distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(difficulties, bins=range(1, 12), edgecolor="black", alpha=0.7)
    ax.set_xlabel("VLM Difficulty Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of VLM Difficulty Scores (n={len(valid)})")
    fig.tight_layout()
    fig.savefig(output_dir / "vlm_difficulty_distribution.png", dpi=200)
    plt.close(fig)

    # ── Energy-stratified correlations ──────────────────────────────────
    energy_bins = config.energy_bins
    bin_labels = [
        f"{energy_bins[i]}–{energy_bins[i+1]}" for i in range(len(energy_bins) - 1)
    ]

    df = pd.DataFrame(
        {
            "difficulty": difficulties,
            "gpr": gprs,
            "energy": energies,
        }
    )
    df["energy_bin"] = pd.cut(
        df["energy"], bins=energy_bins, labels=bin_labels, right=False
    )

    strat_results = []
    for label in bin_labels:
        subset = df[df["energy_bin"] == label]
        if len(subset) >= 10:
            rho, p = spearmanr(subset["difficulty"], subset["gpr"])
            strat_results.append({"bin": label, "n": len(subset), "rho": rho, "p": p})
            logger.info(f"  {label} MeV (n={len(subset)}): ρ={rho:.3f}, p={p:.2e}")

    if strat_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins_x = [s["bin"] for s in strat_results]
        rhos = [s["rho"] for s in strat_results]
        ax.bar(bins_x, rhos, color="steelblue", edgecolor="black")
        ax.set_xlabel("Energy Bin [MeV]")
        ax.set_ylabel("Spearman ρ (Difficulty vs GPR)")
        ax.set_title("Energy-Stratified VLM Difficulty Correlation")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(output_dir / "vlm_energy_stratified_correlation.png", dpi=200)
        plt.close(fig)

    # ── Failure detection AUC-ROC ───────────────────────────────────────
    try:
        from sklearn.metrics import roc_auc_score, roc_curve

        failure_threshold = 95.0
        labels = [1 if g < failure_threshold else 0 for g in gprs]
        if sum(labels) > 0 and sum(labels) < len(labels):
            auc = roc_auc_score(labels, difficulties)
            logger.info(
                f"Failure detection AUC-ROC (GPR < {failure_threshold}%): {auc:.4f}"
            )

            fpr, tpr, _ = roc_curve(labels, difficulties)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"Failure Detection ROC (GPR < {failure_threshold}%)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "vlm_failure_roc.png", dpi=200)
            plt.close(fig)
        else:
            logger.info("Cannot compute AUC-ROC: all samples in same class")
    except ImportError:
        logger.info("sklearn not available — skipping AUC-ROC analysis")


# ── Main processing loop ───────────────────────────────────────────────────


def process_all_samples(
    model: DoTA3D_v3,
    record_ids: list[str],
    dataset: H5PYGenerator,
    config: VLMConfig,
    device: torch.device,
    panels_dir: Path,
    show_progress: bool = True,
) -> list[VLMResult]:
    """Process all beamlets: extract → render → query VLM → compute GPR.

    Args:
        model: Loaded ADoTA model (for GPR evaluation label).
        record_ids: List of record IDs.
        dataset: H5PYGenerator dataset.
        config: VLM configuration.
        device: Torch device.
        panels_dir: Directory for saving review panel PNGs.
        show_progress: Whether to show progress bar.

    Returns:
        List of VLMResult objects.
    """
    results: list[VLMResult] = []
    n_skipped = 0
    n_samples = len(dataset)

    # ── Load VLM once ───────────────────────────────────────────
    vlm_processor, vlm_model_hf, vlm_device = _load_vlm(config)

    iterator = (
        tqdm(range(n_samples), desc="Processing beamlets")
        if show_progress
        else range(n_samples)
    )

    for i in iterator:
        record_id = record_ids[i]

        # ── Extract data ────────────────────────────────────────────
        out = extract_sample(i, record_id, dataset, config)
        if out is None:
            n_skipped += 1
            continue

        ct_hu, flux, energy_mev, gt_dose = out

        # ── BP range estimation (surrogate: from GT) ────────────────
        z_min, z_max = estimate_bp_range(
            ct_hu,
            gt_dose,
            proximal_fraction=config.proximal_fraction,
            fall_fraction=config.fall_fraction,
        )
        voxel_spacing = config.resolution[0]
        bp_min_mm = z_min * voxel_spacing
        bp_max_mm = z_max * voxel_spacing

        # ── Density region count ────────────────────────────────────
        n_regions = analyse_density_regions(
            ct_hu, flux, z_min, z_max, config.flux_threshold_frac
        )

        # ── Render review panel ─────────────────────────────────────
        panel_path = panels_dir / f"{record_id}_panel.png"
        render_review_panel(
            ct_hu=ct_hu,
            flux=flux,
            energy_mev=energy_mev,
            sample_id=record_id,
            output_path=panel_path,
            config=config,
            gt_dose=gt_dose,
            bp_range_slices=(z_min, z_max),
        )

        # ── Format prompt ───────────────────────────────────────────
        prompt = config.vlm_prompt.format(
            energy_mev=energy_mev,
            bp_min_mm=bp_min_mm,
            bp_max_mm=bp_max_mm,
            n_density_regions=n_regions,
            voxel_spacing=voxel_spacing,
        )

        # ── Query VLM ──────────────────────────────────────────────
        t0 = perf_counter()
        votes = query_vlm(
            panel_path,
            prompt,
            config,
            vlm_processor,
            vlm_model_hf,
            vlm_device,
        )
        agg = aggregate_votes(votes)
        query_time = perf_counter() - t0

        # ── Compute GPR (evaluation label) ──────────────────────────
        x_tensor, energy_tensor, y_tensor = dataset[i]
        x_tensor = x_tensor.to(device)
        energy_tensor = energy_tensor.to(device)
        y_tensor = y_tensor.to(device)

        with torch.no_grad():
            y_pred = model(x_tensor.unsqueeze(0), energy_tensor.unsqueeze(0))[0]

        scale = config.scale
        scale_gpr = {"y_min": scale["min_ds"], "y_max": scale["max_ds"]}
        gpr_result = gamma_index_torch(
            y_tensor.unsqueeze(0),
            y_pred,
            scale=scale_gpr,
            gamma_params=config.gamma_params,
            resolution=config.resolution,
        )
        gpr = gpr_result[1][0] * 100

        # ── Store result ────────────────────────────────────────────
        result = VLMResult(
            sample_id=record_id,
            energy_mev=energy_mev,
            bp_range_min_mm=bp_min_mm,
            bp_range_max_mm=bp_max_mm,
            n_density_regions=n_regions,
            vlm_difficulty=agg["difficulty"],
            vlm_confidence=agg["confidence"],
            vlm_reason=agg["reason"],
            vlm_votes=json.dumps(agg["all_votes"]),
            gpr=gpr,
            panel_path=str(panel_path.name),
            query_time_s=query_time,
        )
        results.append(result)

        logger.info(
            f"Sample {record_id}: E={energy_mev:.0f} MeV, "
            f"VLM={agg['difficulty']:.0f}/10 (conf={agg['confidence']:.2f}), "
            f"GPR={gpr:.2f}%, t={query_time:.1f}s"
        )

        if show_progress:
            iterator.set_postfix(
                E=f"{energy_mev:.0f}",
                VLM=f"{agg['difficulty']:.0f}",
                GPR=f"{gpr:.1f}",
            )

        # Clean up panel if not saving
        if not config.save_panels and panel_path.exists():
            panel_path.unlink()

    if n_skipped:
        logger.info(
            f"Skipped {n_skipped}/{n_samples} samples "
            f"(zero flux or E > {config.max_energy_mev:.0f} MeV)"
        )

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────


@app.command()
def main(
    model_name: Annotated[
        Optional[str],
        typer.Argument(help="Name of the model directory under models/"),
    ] = None,
    h5_path: Annotated[
        Optional[Path],
        typer.Argument(help="Path to the HDF5 dataset file"),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(help="Path to YAML configuration file"),
    ] = None,
    excluded_indexes_file: Annotated[
        Optional[Path],
        typer.Option(help="Path to excluded indexes file"),
    ] = None,
    model_fname: Annotated[Optional[str], typer.Option(help="Model filename")] = None,
    device_index: Annotated[
        Optional[int], typer.Option(help="CUDA device index (-1 for CPU)")
    ] = None,
    n_samples: Annotated[
        Optional[int],
        typer.Option(help="Limit to first N samples"),
    ] = None,
    no_progress: Annotated[
        Optional[bool],
        typer.Option(help="Disable progress bar"),
    ] = None,
    verbose: Annotated[
        Optional[bool],
        typer.Option(help="Enable verbose output"),
    ] = None,
) -> None:
    """VLM-based difficulty quantification of training set beamlets.

    Scores each beamlet's predicted difficulty using a local VLM,
    then evaluates against GPR as the ground-truth label.
    """
    # ── Load & merge config ─────────────────────────────────────────
    yaml_config: dict = {}
    config_path: Optional[Path] = None
    if config is not None:
        config_path = config if config.is_absolute() else PROJECT_ROOT / config
        yaml_config = load_yaml_config(config_path)

    model_name = model_name or yaml_config.get("model_name")
    h5_path = h5_path or (
        Path(yaml_config["h5_path"]) if "h5_path" in yaml_config else None
    )
    excluded_indexes_file = excluded_indexes_file or (
        Path(yaml_config["excluded_indexes_file"])
        if "excluded_indexes_file" in yaml_config
        else None
    )
    model_fname = model_fname or yaml_config.get("model_fname", "best_model.pth")
    device_index = (
        device_index if device_index is not None else yaml_config.get("device_index", 0)
    )
    n_samples = n_samples if n_samples is not None else yaml_config.get("n_samples")
    no_progress = (
        no_progress
        if no_progress is not None
        else yaml_config.get("no_progress", False)
    )
    verbose = verbose if verbose is not None else yaml_config.get("verbose", False)

    # Validate required
    if model_name is None:
        raise typer.BadParameter("MODEL_NAME is required (CLI or YAML)")
    if h5_path is None:
        raise typer.BadParameter("H5_PATH is required (CLI or YAML)")

    # ── Run directory ───────────────────────────────────────────────
    runs_dir = PROJECT_ROOT / "runs"
    run_dir = setup_run_directory(runs_dir, subdirs=("figures", "panels"))
    log_file = setup_logging(
        run_dir, verbose=verbose, log_filename="vlm_evaluation.log"
    )

    if config_path is not None:
        shutil.copy2(config_path, run_dir / config_path.name)

    logger.info(f"Run directory: {run_dir}")

    # ── Build VLM config ────────────────────────────────────────────
    gamma_params = DEFAULT_GAMMA_PARAMS.copy()
    gamma_params.update(yaml_config.get("gamma_params", {}))

    # VLM prompt: from YAML or default
    vlm_prompt = yaml_config.get("vlm_prompt", DEFAULT_VLM_PROMPT)

    vlm_config = VLMConfig(
        resolution=tuple(yaml_config.get("resolution", [2.0, 2.0, 2.0])),
        max_energy_mev=yaml_config.get("max_energy_mev", 250.0),
        smoothing_sigma=yaml_config.get("smoothing_sigma", 1.0),
        smoothing_method=yaml_config.get("smoothing_method", "gaussian"),
        proximal_fraction=yaml_config.get("proximal_fraction", 0.50),
        fall_fraction=yaml_config.get("fall_fraction", 0.10),
        flux_threshold_frac=yaml_config.get("flux_threshold_frac", 0.10),
        gamma_params=gamma_params,
        vlm_model=yaml_config.get("vlm_model", "llava-hf/llava-1.5-7b-hf"),
        vlm_device_index=yaml_config.get("vlm_device_index", 0),
        vlm_temperature=yaml_config.get("vlm_temperature", 0.0),
        vlm_n_votes=yaml_config.get("vlm_n_votes", 3),
        vlm_prompt=vlm_prompt,
        vlm_max_new_tokens=yaml_config.get("vlm_max_new_tokens", 256),
        save_panels=yaml_config.get("save_panels", True),
        energy_bins=yaml_config.get("energy_bins", [70, 100, 130, 160, 190, 220, 250]),
    )

    # ── Log configuration ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("VLM DIFFICULTY QUANTIFICATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name} / {model_fname}")
    logger.info(f"HDF5: {h5_path}")
    logger.info(f"Device: {device_index}")
    logger.info(f"VLM model: {vlm_config.vlm_model}")
    logger.info(f"VLM device index: {vlm_config.vlm_device_index}")
    logger.info(f"VLM temperature: {vlm_config.vlm_temperature}")
    logger.info(f"VLM votes (K): {vlm_config.vlm_n_votes}")
    logger.info(f"Save panels: {vlm_config.save_panels}")
    logger.info(f"N samples: {n_samples if n_samples else 'all'}")
    logger.info("=" * 60)

    # ── Resolve paths ───────────────────────────────────────────────
    model_hub = PROJECT_ROOT / "models"
    model_path = model_hub / model_name / model_fname
    hyperparams_path = model_hub / model_name / "hyperparams.json"
    if not h5_path.is_absolute():
        h5_path = PROJECT_ROOT / h5_path
    validate_inputs(h5_path, model_path, hyperparams_path)

    # ── Load excluded indexes ───────────────────────────────────────
    excluded_indexes: list[str] = []
    if excluded_indexes_file is not None:
        exc_path = (
            excluded_indexes_file
            if excluded_indexes_file.is_absolute()
            else PROJECT_ROOT / excluded_indexes_file
        )
        if exc_path.exists():
            with open(exc_path, "r") as f:
                excluded_indexes = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(excluded_indexes)} excluded indexes")

    # ── Discover samples ────────────────────────────────────────────
    with h5py.File(h5_path, "r") as ds:
        all_record_ids = list(ds.keys())
    logger.info(f"Total records: {len(all_record_ids)}")

    record_ids = [r for r in all_record_ids if r not in excluded_indexes]
    logger.info(f"After exclusion: {len(record_ids)}")

    if n_samples is not None:
        record_ids = record_ids[:n_samples]
        logger.info(f"Limited to {len(record_ids)} samples")

    if not record_ids:
        logger.error("No samples remaining")
        raise typer.Exit(code=1)

    # ── Build dataset ───────────────────────────────────────────────
    dataset = H5PYGenerator(
        file_path=str(h5_path),
        indexes=record_ids,
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=True,
    )

    # ── Load model ──────────────────────────────────────────────────
    device = get_device(device_index)
    model = load_model(model_path, hyperparams_path, device)
    logger.info(f"Model loaded on {device}")

    # ── Process all samples ─────────────────────────────────────────
    panels_dir = run_dir / "panels"
    start_time = perf_counter()

    results = process_all_samples(
        model=model,
        record_ids=record_ids,
        dataset=dataset,
        config=vlm_config,
        device=device,
        panels_dir=panels_dir,
        show_progress=not no_progress,
    )

    total_time = perf_counter() - start_time
    logger.info(f"Processing complete: {len(results)} samples in {total_time:.1f}s")

    # ── Save results ────────────────────────────────────────────────
    save_results_csv(results, run_dir / "vlm_results.csv")

    # ── Correlation analysis ────────────────────────────────────────
    figures_dir = run_dir / "figures"
    generate_correlation_analysis(results, figures_dir, vlm_config)

    # ── Summary ─────────────────────────────────────────────────────
    valid = [r for r in results if r.vlm_difficulty > 0]
    if valid:
        diffs = [r.vlm_difficulty for r in valid]
        gprs = [r.gpr for r in valid]
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Valid results: {len(valid)}/{len(results)}")
        logger.info(
            f"VLM difficulty: mean={np.mean(diffs):.2f}, "
            f"std={np.std(diffs):.2f}, range=[{min(diffs):.0f}, {max(diffs):.0f}]"
        )
        logger.info(f"GPR: mean={np.mean(gprs):.2f}%, std={np.std(gprs):.2f}%")
        if len(valid) >= 10:
            rho, _ = spearmanr(diffs, gprs)
            logger.info(f"Spearman ρ (difficulty vs GPR): {rho:.4f}")
        logger.info(
            f"Total time: {total_time:.1f}s "
            f"({total_time / len(results):.1f}s/sample)"
        )
    else:
        logger.warning("No valid VLM results obtained")

    logger.info(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    app()
