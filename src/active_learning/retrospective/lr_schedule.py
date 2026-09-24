"""Fixed (non-plateau) learning-rate schedules for the retrospective loop.

``ReduceLROnPlateau`` couples the learning rate to whatever the validation
loss happens to do, which couples it to the active-learning strategy: a run
whose loss stalls gets its LR cut and trains slower thereafter, so the
scheduler amplifies whatever difference the data made (EXP-0009). The fixed
schedules here are pure functions of the epoch and the cycle length, so a
fixed-schedule run carries no scheduler object and therefore no scheduler
state to checkpoint: mid-cycle resume is exact because the epoch index alone
determines the learning rate.
"""
from __future__ import annotations

import math

FIXED_LR_SCHEDULES = ("constant", "cosine_per_cycle")
ALL_LR_SCHEDULES = ("plateau",) + FIXED_LR_SCHEDULES


def fixed_lr(schedule: str, lr0: float, lr_min: float, epoch_in_cycle: int,
            epochs_in_cycle: int) -> float:
    """The learning rate at ``epoch_in_cycle`` (0-based) of a cycle with
    ``epochs_in_cycle`` epochs, under a fixed (non-plateau) schedule.

    ``"constant"``: always ``lr0``.

    ``"cosine_per_cycle"``: a cosine decay from ``lr0`` at epoch 0 to
    ``lr_min`` at the last epoch of the cycle, so every cycle is a warm
    restart; a cycle of one epoch returns ``lr0``.

    Raises:
        ValueError: ``schedule`` is not one of :data:`FIXED_LR_SCHEDULES`
            (this includes ``"plateau"``, which has no fixed formula: it is
            driven by the ``ReduceLROnPlateau`` scheduler object instead).
    """
    if schedule == "constant":
        return float(lr0)
    if schedule == "cosine_per_cycle":
        if epochs_in_cycle <= 1:
            return float(lr0)
        progress = epoch_in_cycle / (epochs_in_cycle - 1)
        return float(lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress)))
    raise ValueError(f"unknown fixed lr_schedule {schedule!r}; expected one of {FIXED_LR_SCHEDULES}")
