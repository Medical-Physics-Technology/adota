from __future__ import annotations

import csv
import importlib
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

os.environ.setdefault("MPLBACKEND", "Agg")


def _preload_torch_cuda_libs() -> None:
    """CuPy needs libnvrtc.so.11.2 (and friends) at runtime. Torch's pip wheel
    ships them under site-packages/nvidia/*/lib, but the dynamic linker won't
    find them by name. Preload via ctypes so cupy's later dlopen-by-name
    resolves to the already-loaded handle. Mutating LD_LIBRARY_PATH from inside
    Python is too late (linker caches paths).
    """
    import ctypes
    import site
    seen: set[str] = set()
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_dir = Path(sp) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for sub in sorted(nvidia_dir.iterdir()):
            lib_dir = sub / "lib"
            if not lib_dir.is_dir():
                continue
            for so in lib_dir.glob("*.so.*"):
                if so.name in seen:
                    continue
                seen.add(so.name)
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


_preload_torch_cuda_libs()


import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import typer
from scipy import ndimage

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.loaders.plan_parser import parse_plan


app = typer.Typer(
    help="Multi-framework benchmark: SciPy vs CuPy vs PyTorch for 3D rotation of CT, dose, and target around an arbitrary pivot."
)


FRAMEWORK_ORDER = ("scipy", "cupy", "torch")
VOLUME_ORDER = ("ct", "dose", "target")


@dataclass
class VolumeRotation:
    rotated: Optional[np.ndarray]
    pure_seconds_all: list[float] = field(default_factory=list)
    practical_seconds_all: list[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def pure_median(self) -> float:
        return float(np.median(self.pure_seconds_all)) if self.pure_seconds_all else float("nan")

    @property
    def practical_median(self) -> float:
        return float(np.median(self.practical_seconds_all)) if self.practical_seconds_all else float("nan")


@dataclass
class FrameworkResult:
    name: str
    device: str
    ct: Optional[VolumeRotation] = None
    dose: Optional[VolumeRotation] = None
    target: Optional[VolumeRotation] = None
    error: Optional[str] = None

    def by_volume(self, vol_name: str) -> Optional[VolumeRotation]:
        return getattr(self, vol_name, None)


@dataclass
class TextTable:
    field_names: list[str]
    rows: list[list[str]] = field(default_factory=list)

    def add_row(self, row: list[object]) -> None:
        self.rows.append([str(value) for value in row])

    def __str__(self) -> str:
        widths = [len(name) for name in self.field_names]
        for row in self.rows:
            widths = [max(width, len(value)) for width, value in zip(widths, row)]

        def fmt_row(values: list[str]) -> str:
            return "| " + " | ".join(
                value.ljust(width) for value, width in zip(values, widths)
            ) + " |"

        separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
        lines = [separator, fmt_row(self.field_names), separator]
        lines.extend(fmt_row(row) for row in self.rows)
        lines.append(separator)
        return "\n".join(lines)


def build_pivot_rotation_matrix_3d(
    angle_deg: float, pivot_xy: tuple[float, float]
) -> np.ndarray:
    """4x4 homogeneous forward rotation around the D axis about a pivot in
    (d, h, w) pixel coords. pivot_xy = (x, y) = (w, h) following the existing
    script's convention (rotation_point[0]=x=column, [1]=y=row).

    The matrix represents the FORWARD rotation R(angle). SciPy/CuPy's
    affine_transform and Torch's grid_sample both interpret the matrix as
    output->input (inverse), so the displayed image rotates by -angle for a
    positive `angle_deg` (forward-matrix convention).
    """
    pivot_w = float(pivot_xy[0])
    pivot_h = float(pivot_xy[1])
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    off_h = pivot_h * (1.0 - c) + pivot_w * s
    off_w = pivot_w * (1.0 - c) - pivot_h * s
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, off_h],
            [0.0, s, c, off_w],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _scipy_rotate_one(
    vol: np.ndarray, matrix4: np.ndarray, repeats: int
) -> VolumeRotation:
    matrix3 = matrix4[:3, :3]
    offset = matrix4[:3, 3]
    cval = float(vol.min())
    vol_f = vol.astype(np.float32, copy=False)

    _ = ndimage.affine_transform(
        vol_f, matrix=matrix3, offset=offset, order=1, mode="constant", cval=cval
    )

    pure_times: list[float] = []
    rotated: Optional[np.ndarray] = None
    for _ in range(repeats):
        t0 = perf_counter()
        rotated = ndimage.affine_transform(
            vol_f, matrix=matrix3, offset=offset, order=1, mode="constant", cval=cval
        )
        t1 = perf_counter()
        pure_times.append(t1 - t0)
    return VolumeRotation(
        rotated=np.asarray(rotated),
        pure_seconds_all=pure_times,
        practical_seconds_all=list(pure_times),
    )


def rotate_volumes_scipy(
    volumes: dict[str, Optional[np.ndarray]],
    angle_deg: float,
    pivot_xy: tuple[float, float],
    repeats: int,
    device: str = "cpu",
) -> tuple[str, dict[str, Optional[VolumeRotation]]]:
    matrix4 = build_pivot_rotation_matrix_3d(angle_deg, pivot_xy)
    out: dict[str, Optional[VolumeRotation]] = {}
    for name, vol in volumes.items():
        if vol is None:
            out[name] = None
            continue
        out[name] = _scipy_rotate_one(vol, matrix4, repeats)
    return "cpu", out


def _cupy_rotate_one(vol, matrix4, repeats, cp, cundi) -> VolumeRotation:
    cval = float(vol.min())
    vol_f = vol.astype(np.float32, copy=False)
    vol_dev = cp.asarray(vol_f)
    matrix3_dev = cp.asarray(matrix4[:3, :3])
    offset_dev = cp.asarray(matrix4[:3, 3])

    _ = cundi.affine_transform(
        vol_dev,
        matrix=matrix3_dev,
        offset=offset_dev,
        order=1,
        mode="constant",
        cval=cval,
    )
    cp.cuda.Stream.null.synchronize()

    pure_times: list[float] = []
    practical_times: list[float] = []
    rotated_np: Optional[np.ndarray] = None
    rotated_dev = None

    for _ in range(repeats):
        cp.cuda.Stream.null.synchronize()
        t0 = perf_counter()
        rotated_dev = cundi.affine_transform(
            vol_dev,
            matrix=matrix3_dev,
            offset=offset_dev,
            order=1,
            mode="constant",
            cval=cval,
        )
        cp.cuda.Stream.null.synchronize()
        t1 = perf_counter()
        pure_times.append(t1 - t0)

        cp.cuda.Stream.null.synchronize()
        t2 = perf_counter()
        vol_dev_p = cp.asarray(vol_f)
        m4_p = matrix4
        m3_p = cp.asarray(m4_p[:3, :3])
        off_p = cp.asarray(m4_p[:3, 3])
        rotated_p = cundi.affine_transform(
            vol_dev_p,
            matrix=m3_p,
            offset=off_p,
            order=1,
            mode="constant",
            cval=cval,
        )
        rotated_np = cp.asnumpy(rotated_p)
        cp.cuda.Stream.null.synchronize()
        t3 = perf_counter()
        practical_times.append(t3 - t2)

    del vol_dev, matrix3_dev, offset_dev
    if rotated_dev is not None:
        del rotated_dev
    cp.get_default_memory_pool().free_all_blocks()

    return VolumeRotation(
        rotated=rotated_np,
        pure_seconds_all=pure_times,
        practical_seconds_all=practical_times,
    )


def rotate_volumes_cupy(
    volumes: dict[str, Optional[np.ndarray]],
    angle_deg: float,
    pivot_xy: tuple[float, float],
    repeats: int,
    device: str = "cuda:0",
) -> tuple[str, dict[str, Optional[VolumeRotation]]]:
    cp = importlib.import_module("cupy")
    cundi = importlib.import_module("cupyx.scipy.ndimage")

    device_id = int(device.split(":")[1]) if ":" in device else 0
    cp.cuda.Device(device_id).use()
    matrix4 = build_pivot_rotation_matrix_3d(angle_deg, pivot_xy)
    out: dict[str, Optional[VolumeRotation]] = {}
    for name, vol in volumes.items():
        if vol is None:
            out[name] = None
            continue
        try:
            out[name] = _cupy_rotate_one(vol, matrix4, repeats, cp, cundi)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  [cupy:{name}] FAILED: {exc}\n{tb}", flush=True)
            out[name] = VolumeRotation(rotated=None, error=str(exc) + "\n" + tb)
    return f"cuda:{device_id}", out


def _torch_norm_theta(
    matrix4_pix: np.ndarray, shape_dhw: tuple[int, int, int]
) -> torch.Tensor:
    """Convert a 4x4 (d, h, w) pixel-space matrix to a 3x4 normalized theta for
    F.affine_grid with align_corners=True on a 5D input. Torch grid coords are
    (x, y, z) -> (W, H, D); reorder (d, h, w) -> (w, h, d), then rescale.
    """
    D, H, W = shape_dhw
    P = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    M_xyz = P @ matrix4_pix @ P
    S_norm_to_pix = np.array(
        [
            [(W - 1) / 2.0, 0.0, 0.0, (W - 1) / 2.0],
            [0.0, (H - 1) / 2.0, 0.0, (H - 1) / 2.0],
            [0.0, 0.0, (D - 1) / 2.0, (D - 1) / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    S_pix_to_norm = np.array(
        [
            [2.0 / (W - 1), 0.0, 0.0, -1.0],
            [0.0, 2.0 / (H - 1), 0.0, -1.0],
            [0.0, 0.0, 2.0 / (D - 1), -1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    M_norm = S_pix_to_norm @ M_xyz @ S_norm_to_pix
    return torch.from_numpy(M_norm[:3, :]).float()


def _torch_rotate_one(
    vol: np.ndarray, matrix4: np.ndarray, repeats: int, dev: torch.device
) -> VolumeRotation:
    cval = float(vol.min())
    vol_f = vol.astype(np.float32, copy=False)
    D, H, W = vol_f.shape
    vol_t = torch.from_numpy(vol_f).to(dev).unsqueeze(0).unsqueeze(0)
    theta = _torch_norm_theta(matrix4, (D, H, W)).to(dev).unsqueeze(0)

    def _make_grid(th: torch.Tensor, size: torch.Size) -> torch.Tensor:
        n_batches, _, depth, height, width = size
        z = torch.linspace(-1.0, 1.0, depth, device=th.device, dtype=th.dtype)
        y = torch.linspace(-1.0, 1.0, height, device=th.device, dtype=th.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=th.device, dtype=th.dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        xx = xx.expand(n_batches, -1, -1, -1)
        yy = yy.expand(n_batches, -1, -1, -1)
        zz = zz.expand(n_batches, -1, -1, -1)
        coeff = th.view(n_batches, 3, 4, 1, 1, 1)
        grid_x = coeff[:, 0, 0] * xx + coeff[:, 0, 1] * yy + coeff[:, 0, 2] * zz + coeff[:, 0, 3]
        grid_y = coeff[:, 1, 0] * xx + coeff[:, 1, 1] * yy + coeff[:, 1, 2] * zz + coeff[:, 1, 3]
        grid_z = coeff[:, 2, 0] * xx + coeff[:, 2, 1] * yy + coeff[:, 2, 2] * zz + coeff[:, 2, 3]
        return torch.stack((grid_x, grid_y, grid_z), dim=-1)

    def _rotate(vol_5d: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        grid = _make_grid(th, vol_5d.shape)
        shifted = vol_5d - cval
        rotated = F.grid_sample(
            shifted, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        outside = torch.any((grid < -1.0) | (grid > 1.0), dim=-1, keepdim=True)
        rotated = rotated.masked_fill(outside.permute(0, 4, 1, 2, 3), 0.0)
        return rotated + cval

    _ = _rotate(vol_t, theta)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    pure_times: list[float] = []
    practical_times: list[float] = []
    rotated_np: Optional[np.ndarray] = None

    for _ in range(repeats):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = perf_counter()
        _ = _rotate(vol_t, theta)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t1 = perf_counter()
        pure_times.append(t1 - t0)

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t2 = perf_counter()
        vol_t_p = torch.from_numpy(vol_f).to(dev).unsqueeze(0).unsqueeze(0)
        theta_p = _torch_norm_theta(matrix4, vol_f.shape).to(dev).unsqueeze(0)
        rotated_p = _rotate(vol_t_p, theta_p)
        rotated_np = rotated_p.squeeze(0).squeeze(0).detach().cpu().numpy().copy()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t3 = perf_counter()
        practical_times.append(t3 - t2)

    return VolumeRotation(
        rotated=rotated_np,
        pure_seconds_all=pure_times,
        practical_seconds_all=practical_times,
    )


def rotate_volumes_torch(
    volumes: dict[str, Optional[np.ndarray]],
    angle_deg: float,
    pivot_xy: tuple[float, float],
    repeats: int,
    device: str = "cuda:0",
) -> tuple[str, dict[str, Optional[VolumeRotation]]]:
    want_cuda = device.startswith("cuda")
    dev = torch.device(device if (want_cuda and torch.cuda.is_available()) else "cpu")
    matrix4 = build_pivot_rotation_matrix_3d(angle_deg, pivot_xy)
    out: dict[str, Optional[VolumeRotation]] = {}
    for name, vol in volumes.items():
        if vol is None:
            out[name] = None
            continue
        try:
            out[name] = _torch_rotate_one(vol, matrix4, repeats, dev)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  [torch:{name}] FAILED: {exc}\n{tb}", flush=True)
            out[name] = VolumeRotation(rotated=None, error=str(exc) + "\n" + tb)
    return str(dev), out


FRAMEWORK_FNS: dict[str, Callable[..., tuple[str, dict[str, Optional[VolumeRotation]]]]] = {
    "scipy": rotate_volumes_scipy,
    "cupy": rotate_volumes_cupy,
    "torch": rotate_volumes_torch,
}


def orient_structure_to_ct_array(
    structure: sitk.Image, ct_arr_shape: tuple[int, int, int]
) -> np.ndarray:
    """Return the structure as a numpy array shaped like the CT array (D, H, W).

    If sitk's array already matches the CT shape we use it as-is. Otherwise we
    fall back to the legacy reorient (swapaxes(0,2) + flip axis=1) and verify
    the result. Plans differ in how structure masks are stored on disk.
    """
    arr = sitk.GetArrayFromImage(structure)
    if arr.shape == ct_arr_shape:
        return np.ascontiguousarray(arr)
    swapped = np.swapaxes(arr, 0, 2)
    flipped = np.copy(np.flip(swapped, axis=1))
    if flipped.shape != ct_arr_shape:
        print(
            f"WARNING: structure shape {flipped.shape} does not match CT shape "
            f"{ct_arr_shape} even after legacy reorient. Using as-is.",
            flush=True,
        )
    return flipped


def load_plan_data(
    plans_root: Path, plan_name: str, load_target: bool
) -> tuple[
    sitk.Image,
    Optional[sitk.Image],
    Optional[np.ndarray],
    Path,
    tuple[float, float, float],
]:
    plan_path = plans_root / plan_name
    if not plan_path.is_dir():
        available_plans = (
            sorted(p.name for p in plans_root.iterdir() if p.is_dir())
            if plans_root.exists()
            else []
        )
        raise FileNotFoundError(
            f"Plan {plan_name!r} not found in {plans_root}. Available plans: {available_plans}"
        )

    plan_pencil_path = plan_path / "PlanPencil.txt"
    ct_grid_path = plan_path / "CT.mhd"
    mc_grid_path = plan_path / "Dose.mhd"
    target_path = plan_path / "target.mhd"

    parsed_plan = parse_plan(str(plan_pencil_path))
    plan_isocenter = parsed_plan.fractions[0].fields[0].isocenter

    ct_grid = sitk.ReadImage(str(ct_grid_path))
    ct_arr_shape = sitk.GetArrayFromImage(ct_grid).shape
    mc_grid = sitk.ReadImage(str(mc_grid_path)) if mc_grid_path.exists() else None
    target_arr = (
        orient_structure_to_ct_array(sitk.ReadImage(str(target_path)), ct_arr_shape)
        if (load_target and target_path.exists())
        else None
    )
    return ct_grid, mc_grid, target_arr, plan_pencil_path, plan_isocenter


def save_or_show_plot(output_path: Path | None, show: bool) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}", flush=True)
    if show:
        plt.show()
    else:
        plt.close()


def _draw_subplot(
    ax,
    title: str,
    ct_slice: np.ndarray,
    dose_slice: Optional[np.ndarray],
    target_slice: Optional[np.ndarray],
    pivot_xy: tuple[float, float],
    error_msg: Optional[str] = None,
) -> None:
    ax.set_title(title, fontsize=10)
    if error_msg is not None and ct_slice is None:
        ax.text(
            0.5,
            0.5,
            f"ERROR\n{error_msg.splitlines()[-1] if error_msg else ''}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="red",
        )
        ax.axis("off")
        return
    ax.imshow(ct_slice, cmap="gray", origin="lower")
    if dose_slice is not None:
        ax.imshow(dose_slice, cmap="hot", origin="lower", alpha=0.5)
    if target_slice is not None and target_slice.any():
        ax.contour(
            target_slice,
            colors="red",
            linewidths=0.5,
            origin="upper", # I think the target mask is stored in the opposite orientation to the CT, so contour needs origin=upper to align. Plans differ in how masks are stored.
            levels=[0.5],
        )
    ax.scatter([pivot_xy[0]], [pivot_xy[1]], color="red", s=20, label="pivot")
    H, W = ct_slice.shape
    ax.scatter([W // 2], [H // 2], color="blue", s=15, label="image center")
    ax.grid(linestyle="--", linewidth=0.5, color="white")
    ax.legend(loc="upper right", fontsize=8)


def plot_comparison_grid(
    results: list[FrameworkResult],
    original_ct: np.ndarray,
    original_dose: Optional[np.ndarray],
    original_target: Optional[np.ndarray],
    pivot_xy: tuple[float, float],
    plot_z: int,
    output_path: Path | None,
    show: bool,
) -> None:
    """2x2 grid. Bottom-left is the original (unrotated) CT with original
    overlays. The 3 frameworks fill TL/TR/BR. With fewer than 3 frameworks the
    remaining cells are left blank.
    """
    framework_slots = ["TL", "TR", "BR"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    slot_to_ax = {
        "TL": axes[0, 0],
        "TR": axes[0, 1],
        "BL": axes[1, 0],
        "BR": axes[1, 1],
    }

    _draw_subplot(
        slot_to_ax["BL"],
        f"original (unrotated) | slice z={plot_z}",
        original_ct[plot_z, :, :],
        original_dose[plot_z, :, :] if original_dose is not None else None,
        original_target[plot_z, :, :] if original_target is not None else None,
        pivot_xy,
    )

    for slot in framework_slots:
        ax = slot_to_ax[slot]
        ax.axis("off")

    for slot, res in zip(framework_slots, results):
        ax = slot_to_ax[slot]
        ax.axis("on")
        ct_rot = res.ct.rotated if res.ct is not None else None
        dose_rot = res.dose.rotated if res.dose is not None else None
        target_rot = res.target.rotated if res.target is not None else None
        ct_pure = res.ct.pure_median if res.ct is not None else float("nan")
        ct_practical = res.ct.practical_median if res.ct is not None else float("nan")
        title = (
            f"{res.name} | ct pure={ct_pure:.4f}s practical={ct_practical:.4f}s | device={res.device}"
        )
        error_msg = None
        if ct_rot is None:
            error_msg = (res.ct.error if res.ct is not None else None) or res.error
        _draw_subplot(
            ax,
            title,
            ct_rot[plot_z, :, :] if ct_rot is not None else None,
            dose_rot[plot_z, :, :] if dose_rot is not None else None,
            target_rot[plot_z, :, :] if target_rot is not None else None,
            pivot_xy,
            error_msg=error_msg,
        )

    fig.tight_layout()
    save_or_show_plot(output_path, show)


def build_results_table(
    results: list[FrameworkResult],
    grid_shape: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    angle_deg: float,
) -> TextTable:
    table = TextTable(
        field_names=[
        "framework",
        "volume",
        "grid_shape",
        "spacing_mm",
        "angle_deg",
        "pure_rotate_s",
        "practical_s",
        "device",
        ]
    )
    spacing_str = f"({spacing_zyx[0]:.3f}, {spacing_zyx[1]:.3f}, {spacing_zyx[2]:.3f})"
    for res in results:
        for vol_name in VOLUME_ORDER:
            vol_rot = res.by_volume(vol_name)
            if vol_rot is None:
                continue
            if vol_rot.rotated is None:
                pure_s = "ERROR"
                prac_s = "ERROR"
            else:
                pure_s = f"{vol_rot.pure_median:.4f}"
                prac_s = f"{vol_rot.practical_median:.4f}"
            table.add_row(
                [
                    res.name,
                    vol_name,
                    str(tuple(grid_shape)),
                    spacing_str,
                    f"{angle_deg:.3f}",
                    pure_s,
                    prac_s,
                    res.device,
                ]
            )
    return table


def save_results_csv(
    csv_path: Path,
    results: list[FrameworkResult],
    grid_shape: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    angle_deg: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "framework",
                "volume",
                "grid_shape",
                "spacing_mm_z",
                "spacing_mm_y",
                "spacing_mm_x",
                "angle_deg",
                "pure_rotate_s",
                "practical_s",
                "device",
                "error",
            ]
        )
        for res in results:
            for vol_name in VOLUME_ORDER:
                vol_rot = res.by_volume(vol_name)
                if vol_rot is None:
                    continue
                err = vol_rot.error or res.error or ""
                writer.writerow(
                    [
                        res.name,
                        vol_name,
                        str(tuple(grid_shape)),
                        f"{spacing_zyx[0]:.6f}",
                        f"{spacing_zyx[1]:.6f}",
                        f"{spacing_zyx[2]:.6f}",
                        f"{angle_deg:.6f}",
                        "" if vol_rot.rotated is None else f"{vol_rot.pure_median:.6f}",
                        "" if vol_rot.rotated is None else f"{vol_rot.practical_median:.6f}",
                        res.device,
                        err.replace("\n", " | "),
                    ]
                )
    print(f"Saved CSV: {csv_path}", flush=True)


def run_one_framework(
    name: str,
    volumes: dict[str, Optional[np.ndarray]],
    angle_deg: float,
    pivot_xy: tuple[float, float],
    repeats: int,
    device: str,
) -> FrameworkResult:
    fn = FRAMEWORK_FNS[name]
    try:
        print(f"[{name}] starting | repeats={repeats} | device={device}", flush=True)
        resolved_device, rotations = fn(volumes, angle_deg, pivot_xy, repeats, device=device)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[{name}] FRAMEWORK FAILED:\n{tb}", flush=True)
        return FrameworkResult(
            name=name,
            device=device if name != "scipy" else "cpu",
            error=str(exc) + "\n" + tb,
        )

    result = FrameworkResult(name=name, device=resolved_device)
    for vol_name, vol_rot in rotations.items():
        if vol_rot is None:
            continue
        setattr(result, vol_name, vol_rot)
        if vol_rot.rotated is not None:
            print(
                f"  [{name}:{vol_name}] pure={vol_rot.pure_median:.4f}s practical={vol_rot.practical_median:.4f}s",
                flush=True,
            )
    return result


def run_all_frameworks(
    volumes: dict[str, Optional[np.ndarray]],
    angle_deg: float,
    pivot_xy: tuple[float, float],
    repeats: int,
    device: str,
    skip: set[str],
) -> list[FrameworkResult]:
    results: list[FrameworkResult] = []
    for name in FRAMEWORK_ORDER:
        if name in skip:
            print(f"[{name}] skipped via --skip-framework", flush=True)
            continue
        results.append(run_one_framework(name, volumes, angle_deg, pivot_xy, repeats, device))
    return results


def correctness_vs_scipy(
    results: list[FrameworkResult], volumes: dict[str, Optional[np.ndarray]]
) -> None:
    scipy_res = next((r for r in results if r.name == "scipy"), None)
    if scipy_res is None:
        print("No SciPy result to compare against; skipping correctness check.", flush=True)
        return
    print("Correctness check vs SciPy (per volume, atol = 1e-2 * intensity range):", flush=True)
    for res in results:
        if res.name == "scipy":
            continue
        for vol_name in VOLUME_ORDER:
            scipy_vol = scipy_res.by_volume(vol_name)
            other_vol = res.by_volume(vol_name)
            if scipy_vol is None or other_vol is None:
                continue
            if scipy_vol.rotated is None or other_vol.rotated is None:
                continue
            src = volumes.get(vol_name)
            if src is None:
                continue
            intensity_range = float(src.max()) - float(src.min())
            atol = max(1e-2 * intensity_range, 1e-6)
            ok = np.allclose(scipy_vol.rotated, other_vol.rotated, atol=atol)
            diff = np.abs(scipy_vol.rotated - other_vol.rotated)
            print(
                f"  {res.name}:{vol_name}: allclose={ok}  max_abs_diff={diff.max():.4g}  mean_abs_diff={diff.mean():.4g}  (atol={atol:.4g})",
                flush=True,
            )


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_smoke_volumes(
    shape_dhw: tuple[int, int, int] = (32, 64, 64),
) -> dict[str, Optional[np.ndarray]]:
    rng = np.random.RandomState(0)
    depth, height, width = shape_dhw
    ct = rng.rand(depth, height, width).astype(np.float32)
    dose = rng.rand(depth, height, width).astype(np.float32) * 50.0
    zz, yy, xx = np.meshgrid(
        np.arange(depth), np.arange(height), np.arange(width), indexing="ij"
    )
    radius = max(1, min(depth, height, width) // 6)
    target = (
        ((zz - depth // 2) ** 2 + (yy - height // 2) ** 2 + (xx - width // 2) ** 2)
        <= radius**2
    ).astype(np.float32)
    return {"ct": ct, "dose": dose, "target": target}


@app.command()
def main(
    plans_root: Path = typer.Option(
        Path("/scratch/mstryja/opentps_plans/"),
        "--plans-root",
        help="Root directory containing OpenTPS plan folders.",
    ),
    plan_name: str = typer.Option(
        "Prostate-AEC-120_100M_bilateral_test_1_review",
        "--plan-name",
        help="Plan folder name inside --plans-root.",
    ),
    angle: float = typer.Option(
        30.0,
        "--angle",
        help="Rotation angle in degrees (forward R(angle), same value for all frameworks).",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="GPU device for CuPy/Torch, e.g. cuda, cuda:0, or cpu. Defaults to CUDA when available.",
    ),
    repeats: int = typer.Option(
        3,
        "--repeats",
        min=1,
        help="Number of timed runs per (framework, volume). Median reported.",
    ),
    skip_framework: list[str] = typer.Option(
        [],
        "--skip-framework",
        help="Frameworks to skip. Repeatable. Choices: scipy, cupy, torch.",
    ),
    no_target: bool = typer.Option(
        False,
        "--no-target",
        help="Do not load/rotate the target mask.",
    ),
    no_dose: bool = typer.Option(
        False,
        "--no-dose",
        help="Do not load/rotate the dose grid.",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="Display plots interactively in addition to saving them when --output-dir is set.",
    ),
    output_dir: Optional[Path] = typer.Option(
        ROOT_DIR / "results" / "rotation_performance_analysis",
        "--output-dir",
        help="Directory where outputs are saved.",
    ),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Skip plan loading and run on small synthetic CT+dose+target volumes (32, 64, 64).",
    ),
    smoke_depth: int = typer.Option(
        32,
        "--smoke-depth",
        min=1,
        help="Depth dimension for synthetic smoke volumes.",
    ),
    smoke_height: int = typer.Option(
        64,
        "--smoke-height",
        min=1,
        help="Height dimension for synthetic smoke volumes.",
    ),
    smoke_width: int = typer.Option(
        64,
        "--smoke-width",
        min=1,
        help="Width dimension for synthetic smoke volumes.",
    ),
) -> None:
    device_resolved = _resolve_device(device)
    skip_set = {s.strip().lower() for s in skip_framework}

    if not torch.cuda.is_available() and not device_resolved.startswith("cpu"):
        print(
            "WARNING: CUDA not available; Torch will run on CPU. CuPy will likely fail.",
            flush=True,
        )

    if smoke:
        volumes = _make_smoke_volumes((smoke_depth, smoke_height, smoke_width))
        spacing_zyx = (1.0, 1.0, 1.0)
        pivot_xy = (0.375 * smoke_width, 0.625 * smoke_height)
        plot_z = volumes["ct"].shape[0] // 2
        print(
            f"Smoke mode: synthetic volumes shape={volumes['ct'].shape} pivot_xy={pivot_xy} plot_z={plot_z}",
            flush=True,
        )
    else:
        ct_grid, mc_grid, target_arr_ct_oriented, plan_pencil_path, plan_isocenter = (
            load_plan_data(plans_root, plan_name, load_target=not no_target)
        )
        ct_arr = sitk.GetArrayFromImage(ct_grid)
        dose_arr: Optional[np.ndarray] = None
        if mc_grid is not None and not no_dose:
            dose_arr = sitk.GetArrayFromImage(mc_grid)
            if dose_arr.shape != ct_arr.shape:
                print(
                    f"WARNING: dose shape {dose_arr.shape} != CT shape {ct_arr.shape}; "
                    "rotation will still time, but overlay may not align.",
                    flush=True,
                )
        spacing_xyz = ct_grid.GetSpacing()
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        rotation_point = np.array(
            ct_grid.TransformPhysicalPointToIndex(
                np.asarray(ct_grid.GetOrigin()) + plan_isocenter
            )
        )
        pivot_xy = (float(rotation_point[0]), float(rotation_point[1]))
        plot_z = int(rotation_point[-1])
        volumes = {"ct": ct_arr, "dose": dose_arr, "target": target_arr_ct_oriented}

        print(f"Using plan from: {plan_pencil_path.parent}", flush=True)
        print(
            f"CT size: {ct_grid.GetSize()} spacing: {ct_grid.GetSpacing()} array shape: {ct_arr.shape}",
            flush=True,
        )
        if dose_arr is not None:
            print(f"Dose array shape: {dose_arr.shape}", flush=True)
        if target_arr_ct_oriented is not None:
            print(f"Target array shape: {target_arr_ct_oriented.shape}", flush=True)
        print(f"Plan isocenter: {plan_isocenter}", flush=True)
        print(f"Voxel coordinates of the rotation point: {rotation_point}", flush=True)
        print(f"Pivot xy (W, H) voxel indices: {pivot_xy} | plot_z={plot_z}", flush=True)

    print(
        f"Running frameworks {[f for f in FRAMEWORK_ORDER if f not in skip_set]} "
        f"on volumes {[v for v, a in volumes.items() if a is not None]} "
        f"| angle={angle}deg | repeats={repeats} | device={device_resolved}",
        flush=True,
    )

    results = run_all_frameworks(volumes, angle, pivot_xy, repeats, device_resolved, skip_set)

    correctness_vs_scipy(results, volumes)

    grid_shape = volumes["ct"].shape
    table = build_results_table(results, grid_shape, spacing_zyx, angle)
    print("\nRotation timing summary (per-volume)", flush=True)
    print(table, flush=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison_grid(
            results,
            volumes["ct"],
            volumes.get("dose"),
            volumes.get("target"),
            pivot_xy,
            plot_z,
            output_dir / "rotation_comparison.png",
            show,
        )
        save_results_csv(
            output_dir / "results_table.csv",
            results,
            grid_shape,
            spacing_zyx,
            angle,
        )


if __name__ == "__main__":
    app()
