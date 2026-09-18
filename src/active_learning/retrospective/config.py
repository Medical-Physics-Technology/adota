"""The retrospective benchmark's run configuration.

Split out of :mod:`src.active_learning.retrospective.loop`, which re-exports
:class:`RetroConfig` (``from src.active_learning.retrospective.loop import
RetroConfig`` still works), purely to keep that module under the 500-line
ratchet; nothing here is specific to any one stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

from src.active_learning.retrospective.dataset import DEFAULT_EXCLUDE_PATH
from src.active_learning.retrospective.lr_schedule import ALL_LR_SCHEDULES
from src.active_learning.retrospective.validation import MetricSettings
from src.adota.config import DEFAULT_GAMMA_PARAMS, DEFAULT_SCALE
from src.schemas.configs import TrainingConfig

PROVENANCE_CSV = ("/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition/"
                  "uuid_provenance_map.csv")


@dataclass
class RetroConfig:
    """Everything one run of the retrospective benchmark needs."""

    experiment: str = "EXP-0009"
    dataset_path: str = ""
    exclude_indexes_path: str = DEFAULT_EXCLUDE_PATH
    record_provenance_csv: Optional[str] = PROVENANCE_CSV
    splits_dir: str = "/scratch/mstryja/adota_runs/al_retro/splits"
    runs_dir: str = "/scratch/mstryja/adota_runs/al_retro"
    data_fraction: float = 1.0
    """Share of ``D`` the experiment uses, drawn uniformly at random with
    ``data_fraction_seed`` before any split; V, T, the cycle-0 set and the pool
    all come from that subset. 0.3 is the fast pilot; 0.4, 0.5, 0.6 and 1.0 are
    the planned scale-ups. Every value gets its own ``splits_dir``."""
    data_fraction_seed: int = 20260911
    max_records: Optional[int] = None
    """Cap on ``D`` after ``data_fraction``; the smoke test's knob, never the real run's."""
    val_fraction: float = 0.15
    initial_fraction: float = 0.20
    split_seed: int = 42
    initial_seed: int = 20260910
    batch_fraction: float = 0.10
    n_cycles: int = 5
    epochs_per_cycle: int = 50
    eval_every_n_epochs: int = 5
    eval_subsample_size: int = 1000
    eval_subsample_seed: int = 20260910
    strategy: str = "random"
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    """Keyword parameters forwarded to the strategy (for example
    ``top_fraction`` for ``score_topk_mixed``)."""
    seed: int = 1234
    device_index: Optional[int] = None
    checkpoint_every_n_epochs: int = 10
    scorer: Dict[str, Any] = field(default_factory=lambda: {"name": "difficulty"})
    gamma_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_GAMMA_PARAMS))
    gamma_resolution_mm: List[float] = field(default_factory=lambda: [2.0, 2.0, 2.0])
    gamma_cutoff_percent: float = 10.0
    gamma_backend: str = "torch"
    lr_schedule: str = "plateau"
    """How the learning rate evolves within a cycle. ``"plateau"`` is today's
    ``ReduceLROnPlateau`` on the validation loss, its state carried across
    cycles (the default, so old configs and EXP-0009 reproduce byte for byte).
    ``"constant"`` holds ``training.learning_rate`` fixed at every epoch of
    every cycle. ``"cosine_per_cycle"`` decays from ``training.learning_rate``
    to ``lr_min`` over each cycle and warm-restarts at the start of the next
    one. The fixed modes decouple the learning rate from the strategy, so only
    the training-set growth differs between runs: EXP-0009 found the plateau
    scheduler otherwise amplifies whatever difference the data made to the
    validation loss."""
    lr_min: float = 0.0
    """The floor of the ``"cosine_per_cycle"`` schedule: an absolute learning
    rate, not a fraction of ``training.learning_rate``. Unused otherwise."""
    training: Dict[str, Any] = field(default_factory=dict)
    """The :class:`TrainingConfig` block: model, optimizer, loader, scale."""

    def __post_init__(self) -> None:
        if self.lr_schedule not in ALL_LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {ALL_LR_SCHEDULES}, got {self.lr_schedule!r}")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RetroConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown retrospective config keys {unknown}")
        return cls(**raw)

    def training_config(self) -> TrainingConfig:
        valid = {f.name for f in fields(TrainingConfig)}
        block = {k: v for k, v in self.training.items() if k in valid}
        block.update(dataset_path=self.dataset_path,
                     excluded_indexes_file=self.exclude_indexes_path,
                     seed=self.seed, runs_dir=self.runs_dir,
                     device_index=self.device_index if self.device_index is not None else 0,
                     gamma_params=dict(self.gamma_params),
                     checkpoint_every_n_epochs=self.checkpoint_every_n_epochs,
                     gpr_resolution_mm=tuple(self.gamma_resolution_mm))
        block.setdefault("scale", dict(DEFAULT_SCALE))
        if "input_shape" in block:
            block["input_shape"] = tuple(block["input_shape"])
        return TrainingConfig(**block)

    def metric_settings(self) -> MetricSettings:
        cfg = self.training_config()
        return MetricSettings(scale=dict(cfg.scale), gamma_params=dict(self.gamma_params),
                              resolution_mm=tuple(self.gamma_resolution_mm),
                              gamma_cutoff_percent=self.gamma_cutoff_percent,
                              gamma_backend=self.gamma_backend,
                              lps_dx_mm=cfg.lps_dx_mm, lps_dy_mm=cfg.lps_dy_mm)
