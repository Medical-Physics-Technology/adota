"""Loader for an OpenTPS plan directory.

An OpenTPS plan directory bundles every input the ADoTA plan-level pipeline
needs for a single plan:

* ``CT.mhd`` / ``CT.raw``     -- the CT grid MCsquare actually simulated on,
* ``PlanPencil.txt``         -- the pencil-beam scanning plan (parsed here),
* ``bdl.txt``                -- the beam data library (nozzle/SMX/SMY + energy
                                table); fine-tuned and request-only, so it is
                                NOT committed to the repo,
* ``config.txt``             -- the MCsquare run configuration,
* ``target.mhd`` / ``OAR_*.mhd`` -- structure masks (loaded as contours),
* ``Dose.mhd`` / ``Dose.raw``    -- the MCsquare reference dose (path only,
                                treated as READ-ONLY by the pipeline).

This module loads those files into a single :class:`PlanDirectory` and offers a
human-readable :meth:`PlanDirectory.summary`. CT and contour grids are read with
SimpleITK so their physical metadata (origin, spacing, direction) is preserved.

Full BDL geometry parsing is intentionally out of scope here -- the BDL is read
as raw text and its path recorded; the typed ``BeamDataLibrary`` lands in
``src/beamlets/bdl.py`` later in the integration plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import SimpleITK as sitk

from src.loaders.plan_parser import Plan, parse_plan

logger = logging.getLogger(__name__)

__all__ = [
    "PlanDirectory",
    "load_plan_directory",
    "parse_opentps_config",
    "CT_FILENAME",
    "PLAN_FILENAME",
    "CONFIG_FILENAME",
    "DEFAULT_BDL_FILENAME",
    "MC_DOSE_FILENAME",
]

CT_FILENAME = "CT.mhd"
PLAN_FILENAME = "PlanPencil.txt"
CONFIG_FILENAME = "config.txt"
DEFAULT_BDL_FILENAME = "bdl.txt"
MC_DOSE_FILENAME = "Dose.mhd"
ADOTA_DOSE_FILENAME = "Dose_ADoTA.mhd"

# .mhd files that are NOT structure contours (the CT and the dose grids -- the
# MC reference and our accumulated ADoTA output written into the same dir).
_NON_CONTOUR_MHD = {CT_FILENAME, MC_DOSE_FILENAME, ADOTA_DOSE_FILENAME}


def parse_opentps_config(config_path: Path) -> Dict[str, Optional[str]]:
    """Parse an MCsquare ``config.txt`` into a ``{key: value}`` dict.

    Lines are ``Key value`` pairs; ``#`` starts a comment (whole-line or
    inline). A key with no value maps to ``None``.

    Args:
        config_path: Path to the ``config.txt`` file.

    Returns:
        Mapping from configuration key to its (string) value or ``None``.
    """
    config: Dict[str, Optional[str]] = {}
    for raw_line in Path(config_path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split(None, 1)
        if len(tokens) == 2:
            config[tokens[0]] = tokens[1].strip()
        else:
            config[tokens[0]] = None
    return config


@dataclass
class PlanDirectory:
    """Everything loaded from a single OpenTPS plan directory.

    Attributes:
        plan_dir: The source directory.
        plan: The parsed :class:`~src.loaders.plan_parser.Plan`.
        ct: The CT grid as a SimpleITK image (physical metadata preserved).
        contours: Structure masks keyed by file stem (e.g. ``"target"``,
            ``"OAR_1"``).
        config: Parsed ``config.txt`` key/value pairs.
        bdl_path: Path to the beam data library used.
        bdl_text: Raw text of the beam data library.
        mc_dose_path: Path to the MCsquare reference ``Dose.mhd`` if present
            (the file itself is left untouched), otherwise ``None``.
    """

    plan_dir: Path
    plan: Plan
    ct: sitk.Image
    contours: Dict[str, sitk.Image] = field(default_factory=dict)
    config: Dict[str, Optional[str]] = field(default_factory=dict)
    bdl_path: Optional[Path] = None
    bdl_text: str = ""
    mc_dose_path: Optional[Path] = None

    def summary(self, max_fields: int = 2, max_control_points: int = 3) -> str:
        """Return a human-readable preview of the loaded plan directory.

        Args:
            max_fields: Maximum number of fields to expand per fraction.
            max_control_points: Maximum number of control points to expand
                per field.

        Returns:
            A multi-line preview string (directory, CT/contours geometry,
            BDL/config, and a truncated plan tree).
        """
        lines: List[str] = []
        lines.append(f"Plan directory : {self.plan_dir}")
        lines.append("")

        lines.append("CT grid (SimpleITK):")
        lines.extend(_indent(_image_summary(self.ct)))
        lines.append("")

        lines.append(f"Contours ({len(self.contours)}):")
        for name, image in self.contours.items():
            lines.append(f"  {name}: size={image.GetSize()}")
        if not self.contours:
            lines.append("  (none found)")
        lines.append("")

        lines.append("Beam data library:")
        lines.append(f"  path : {self.bdl_path}")
        for preview_line in _bdl_preview(self.bdl_text):
            lines.append(f"  | {preview_line}")
        lines.append("")

        lines.append(f"MCsquare config : {len(self.config)} keys")
        lines.append(f"MC reference dose : {self.mc_dose_path or '(not found)'}")
        lines.append("")

        lines.append("Parsed plan:")
        lines.extend(_indent(_plan_preview(self.plan, max_fields, max_control_points)))
        return "\n".join(lines)


def load_plan_directory(
    plan_dir: Path,
    bdl_path: Optional[Path] = None,
) -> PlanDirectory:
    """Load every ADoTA input from an OpenTPS plan directory.

    Args:
        plan_dir: Directory containing ``CT.mhd``, ``PlanPencil.txt``,
            ``config.txt``, the beam data library, structure masks, and
            (optionally) the MCsquare ``Dose.mhd``.
        bdl_path: Explicit beam data library path. When ``None`` the
            plan-local ``bdl.txt`` is used.

    Returns:
        The populated :class:`PlanDirectory`.

    Raises:
        FileNotFoundError: If the directory or any required file is missing.
    """
    plan_dir = Path(plan_dir)
    if not plan_dir.is_dir():
        raise FileNotFoundError(f"Plan directory not found: {plan_dir}")

    ct_path = _require(plan_dir / CT_FILENAME)
    plan_path = _require(plan_dir / PLAN_FILENAME)
    config_path = _require(plan_dir / CONFIG_FILENAME)

    resolved_bdl = Path(bdl_path) if bdl_path is not None else plan_dir / DEFAULT_BDL_FILENAME
    _require(resolved_bdl)

    logger.info("Loading CT grid from %s", ct_path)
    ct = sitk.ReadImage(str(ct_path))

    logger.info("Parsing plan from %s", plan_path)
    plan = parse_plan(str(plan_path))

    contours = _load_contours(plan_dir)
    logger.info("Loaded %d contour(s): %s", len(contours), ", ".join(contours) or "-")

    config = parse_opentps_config(config_path)
    logger.info("Parsed config with %d keys from %s", len(config), config_path)

    bdl_text = resolved_bdl.read_text()
    logger.info("Read beam data library from %s", resolved_bdl)

    mc_dose = plan_dir / MC_DOSE_FILENAME
    mc_dose_path = mc_dose if mc_dose.is_file() else None

    return PlanDirectory(
        plan_dir=plan_dir,
        plan=plan,
        ct=ct,
        contours=contours,
        config=config,
        bdl_path=resolved_bdl,
        bdl_text=bdl_text,
        mc_dose_path=mc_dose_path,
    )


def _load_contours(plan_dir: Path) -> Dict[str, sitk.Image]:
    """Load every structure-mask ``.mhd`` (all but CT and the reference dose)."""
    contours: Dict[str, sitk.Image] = {}
    for mhd in sorted(plan_dir.glob("*.mhd")):
        if mhd.name in _NON_CONTOUR_MHD:
            continue
        contours[mhd.stem] = sitk.ReadImage(str(mhd))
    return contours


def _require(path: Path) -> Path:
    """Return *path* if it exists, else raise a clear FileNotFoundError."""
    if not path.is_file():
        raise FileNotFoundError(f"Required plan file not found: {path}")
    return path


def _indent(lines: List[str], prefix: str = "  ") -> List[str]:
    return [f"{prefix}{line}" for line in lines]


def _image_summary(image: sitk.Image) -> List[str]:
    """Summarize a SimpleITK image's geometry and pixel type."""
    return [
        f"size    : {image.GetSize()}",
        f"spacing : {tuple(round(s, 4) for s in image.GetSpacing())}",
        f"origin  : {tuple(round(o, 4) for o in image.GetOrigin())}",
        f"pixel   : {image.GetPixelIDTypeAsString()}",
    ]


def _bdl_preview(bdl_text: str, n_lines: int = 6) -> List[str]:
    """Return the first *n_lines* non-empty lines of the BDL for a preview."""
    preview = [ln.strip() for ln in bdl_text.splitlines() if ln.strip()]
    return preview[:n_lines]


def _plan_preview(plan: Plan, max_fields: int, max_control_points: int) -> List[str]:
    """Build a truncated, readable tree of the parsed plan."""
    lines: List[str] = [
        f"name : {plan.name}",
        f"treatment fractions : {plan.n_treatment_fractions}",
        f"fraction blocks     : {len(plan.fractions)}",
        f"total meterset weight : {plan.total_msw}",
    ]
    total_fields = sum(len(fr.fields) for fr in plan.fractions)
    total_spots = sum(
        len(cp.spots)
        for fr in plan.fractions
        for fld in fr.fields
        for cp in fld.control_points
    )
    lines.append(f"fields : {total_fields}, total spots : {total_spots}")

    for fr in plan.fractions:
        lines.append(f"fraction {fr.id} (field ids {fr.field_ids}):")
        for fld in fr.fields[:max_fields]:
            field_spots = sum(len(cp.spots) for cp in fld.control_points)
            energies = [cp.energy_mev for cp in fld.control_points]
            energy_range = (
                f"{min(energies):.1f}-{max(energies):.1f} MeV" if energies else "n/a"
            )
            lines.append(
                f"  field {fld.id}: gantry={fld.gantry_angle} couch={fld.couch_angle} "
                f"iso={fld.isocenter}"
            )
            lines.append(
                f"    control points={len(fld.control_points)} spots={field_spots} "
                f"energy={energy_range}"
            )
            for cp in fld.control_points[:max_control_points]:
                lines.append(
                    f"    cp {cp.index}: E={cp.energy_mev} MeV, "
                    f"msw={cp.cumulative_msw}, spots={len(cp.spots)}"
                )
            if len(fld.control_points) > max_control_points:
                lines.append(
                    f"    ... ({len(fld.control_points) - max_control_points} more "
                    "control points)"
                )
        if len(fr.fields) > max_fields:
            lines.append(f"  ... ({len(fr.fields) - max_fields} more fields)")
    return lines
