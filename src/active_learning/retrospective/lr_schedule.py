"""Fixed (non-plateau) learning-rate schedules for the retrospective loop.

``ReduceLROnPlateau`` couples the learning rate to whatever the validation
loss happens to do, which couples it to the active-learning strategy: a run
whose loss stalls gets its LR cut and trains slower thereafter, so the
scheduler amplifies whatever difference the data made (EXP-0009). The fixed
schedules here are pure functions of the epoch and the cycle length, so a
fixed-schedule run carries no scheduler object and therefore no scheduler
state to checkpoint: mid-cycle resume is exact because the epoch index alone
determines the learning rate.

The optional linear warmup (EXP-0012) is part of that pure function too. It
exists because the warm restart of ``"cosine_per_cycle"`` was itself a
divergence in EXP-0011: jumping from ``lr_min`` back to ``lr0`` at epoch 0 of
every cycle multiplied the training loss 40 to 67-fold.
"""
from __future__ import annotations

import math

FIXED_LR_SCHEDULES = ("constant", "cosine_per_cycle")
ALL_LR_SCHEDULES = ("plateau",) + FIXED_LR_SCHEDULES


def validate_warmup(warmup_epochs: int, epochs_in_cycle: int) -> None:
    """Raise ``ValueError`` unless ``0 <= warmup_epochs < epochs_in_cycle``
    (a warmup of zero epochs is always valid)."""
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    if warmup_epochs > 0 and warmup_epochs >= epochs_in_cycle:
        raise ValueError(f"warmup_epochs ({warmup_epochs}) must be smaller than the "
                         f"{epochs_in_cycle} epochs of a cycle")


def fixed_lr(schedule: str, lr0: float, lr_min: float, epoch_in_cycle: int,
             epochs_in_cycle: int, warmup_epochs: int = 0) -> float:
    """The learning rate at ``epoch_in_cycle`` (0-based) of a cycle with
    ``epochs_in_cycle`` epochs, under a fixed (non-plateau) schedule.

    ``"constant"``: always ``lr0``.

    ``"cosine_per_cycle"``: a cosine decay from ``lr0`` at epoch 0 to
    ``lr_min`` at the last epoch of the cycle, so every cycle is a warm
    restart; a cycle of one epoch returns ``lr0``.

    ``warmup_epochs = W > 0`` prepends a linear warmup to either schedule:
    epochs ``0 .. W-1`` rise linearly from ``lr_min`` (exactly ``lr_min`` at
    epoch 0, the value a cosine cycle ended at, so there is no jump) towards
    ``lr0``, reached at epoch ``W``. ``"constant"`` then stays at ``lr0``;
    ``"cosine_per_cycle"`` decays from ``lr0`` at epoch ``W`` to ``lr_min`` at
    the last epoch (when ``W`` is the last epoch there is no room to decay and
    it returns ``lr0``, as a one-epoch cycle does). ``W = 0`` is the formula
    above, unchanged.

    Raises:
        ValueError: ``schedule`` is not one of :data:`FIXED_LR_SCHEDULES`
            (this includes ``"plateau"``, which has no fixed formula: it is
            driven by the ``ReduceLROnPlateau`` scheduler object instead), or
            ``warmup_epochs`` fails :func:`validate_warmup`.
    """
    if schedule not in FIXED_LR_SCHEDULES:
        raise ValueError(f"unknown fixed lr_schedule {schedule!r}; expected one of {FIXED_LR_SCHEDULES}")
    validate_warmup(warmup_epochs, epochs_in_cycle)
    if warmup_epochs > 0:
        if epoch_in_cycle < warmup_epochs:
            return float(lr_min + (lr0 - lr_min) * epoch_in_cycle / warmup_epochs)
        if schedule == "constant":
            return float(lr0)
        decay_epochs = epochs_in_cycle - 1 - warmup_epochs
        if decay_epochs <= 0:
            return float(lr0)
        progress = (epoch_in_cycle - warmup_epochs) / decay_epochs
        return float(lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress)))
    if schedule == "constant":
        return float(lr0)
    if epochs_in_cycle <= 1:
        return float(lr0)
    progress = epoch_in_cycle / (epochs_in_cycle - 1)
    return float(lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress)))
