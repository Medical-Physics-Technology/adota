"""The search space of the per-mixer hyperparameter optimisation, and the trials
drawn from it.

A space is a mapping from a parameter name to one of

- ``{"choices": [a, b, ...]}``: categorical, the unit interval cut into equal bins;
- ``{"loguniform": [lo, hi]}``: continuous, uniform in ``log``;
- ``{"uniform": [lo, hi]}``: continuous, uniform.

Trials are drawn from a scrambled Sobol sequence over the shared parameters
plus the mixer's own, so they cover the space evenly instead of clustering.
That is the reason for quasi-random search here rather than Bayesian
optimisation: the trials double as the pool from which a second, distant
configuration of the same mixer is picked (EXP-0014's C1), and BO trials
cluster around one optimum. A drawn configuration over the parameter cap is
skipped and the next Sobol point taken, so the cap applies to every mixer
alike and the order of the points fixes the result.

Every trial keeps its point in the unit cube; :func:`unit_distance` compares two
trials of one space there, which is how "far from C0" is measured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import qmc

KINDS = ("choices", "loguniform", "uniform")


def _kind(spec: Mapping[str, Any]) -> str:
    found = [k for k in KINDS if k in spec]
    if len(found) != 1:
        raise ValueError(f"a parameter needs exactly one of {KINDS}, got {sorted(spec)}")
    return found[0]


def validate_space(space: Mapping[str, Mapping[str, Any]]) -> None:
    for name, spec in space.items():
        kind = _kind(spec)
        values = spec[kind]
        if kind == "choices" and not values:
            raise ValueError(f"{name}: empty choices")
        if kind != "choices":
            lo, hi = values
            if not lo < hi or (kind == "loguniform" and lo <= 0):
                raise ValueError(f"{name}: invalid range {values}")


def decode(u: Sequence[float], space: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Map a point of the unit cube to parameter values, one coordinate per
    parameter in the order of ``space``."""
    if len(u) != len(space):
        raise ValueError(f"point has {len(u)} coordinates for {len(space)} parameters")
    out: Dict[str, Any] = {}
    for x, (name, spec) in zip(u, space.items()):
        kind = _kind(spec)
        x = float(min(max(x, 0.0), 1.0 - 1e-12))
        if kind == "choices":
            out[name] = spec["choices"][int(x * len(spec["choices"]))]
        elif kind == "loguniform":
            lo, hi = spec["loguniform"]
            out[name] = float(math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))))
        else:
            lo, hi = spec["uniform"]
            out[name] = float(lo + x * (hi - lo))
    return out


def encode(values: Mapping[str, Any], space: Mapping[str, Mapping[str, Any]]) -> List[float]:
    """The inverse of :func:`decode`: a categorical value maps to its bin centre.
    Used to place a hand-picked configuration (v11) in the same cube."""
    u = []
    for name, spec in space.items():
        kind, value = _kind(spec), values[name]
        if kind == "choices":
            choices = list(spec["choices"])
            if value not in choices:
                raise ValueError(f"{name}={value!r} is not among the choices {choices}")
            u.append((choices.index(value) + 0.5) / len(choices))
        elif kind == "loguniform":
            lo, hi = spec["loguniform"]
            u.append((math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo)))
        else:
            lo, hi = spec["uniform"]
            u.append((value - lo) / (hi - lo))
    return u


def unit_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance of two points of one space, scaled by sqrt(dimensions)
    so it lies in [0, 1]."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / math.sqrt(max(len(a), 1)))


@dataclass
class Trial:
    """One configuration of one mixer."""

    trial_id: str
    mixer: str
    params: Dict[str, Any]
    unit_point: List[float]
    n_params: int
    sobol_index: Optional[int] = None
    """Position in the Sobol sequence; ``None`` for a fixed (hand-picked) trial."""
    extra: Dict[str, Any] = field(default_factory=dict)


def draw_trials(mixer: str, space: Mapping[str, Mapping[str, Any]], n_trials: int, seed: int,
                count_params: Callable[[Dict[str, Any]], int], max_params: int,
                fixed: Sequence[Dict[str, Any]] = (), max_draws: int = 4096) -> List[Trial]:
    """``n_trials`` trials of ``mixer``: the ``fixed`` configurations first (each
    must lie in the space and under the cap), then scrambled-Sobol points in
    sequence order, skipping any over ``max_params``."""
    validate_space(space)
    trials: List[Trial] = []
    for k, values in enumerate(fixed):
        params = {name: values[name] for name in space}
        n = int(count_params(params))
        if n > max_params:
            raise ValueError(f"fixed trial {k} of {mixer} has {n} parameters, over the cap {max_params}")
        trials.append(Trial(f"{mixer}_fixed{k}", mixer, params, encode(params, space), n, None,
                            {"fixed": True}))
    if len(trials) > n_trials:
        raise ValueError(f"{len(fixed)} fixed trials exceed n_trials={n_trials}")
    sampler = qmc.Sobol(d=max(len(space), 1), scramble=True, seed=seed)
    points = sampler.random_base2(int(math.ceil(math.log2(max_draws))))
    skipped = 0
    for index, point in enumerate(points):
        if len(trials) == n_trials:
            break
        u = [float(x) for x in point[:len(space)]]
        params = decode(u, space)
        n = int(count_params(params))
        if n > max_params:
            skipped += 1
            continue
        trials.append(Trial(f"{mixer}_{index:03d}", mixer, params, u, n, index, {"fixed": False}))
    if len(trials) < n_trials:
        raise RuntimeError(f"only {len(trials)} of {n_trials} {mixer} trials fit under the cap "
                           f"{max_params} in {len(points)} Sobol points")
    for t in trials:
        t.extra["skipped_over_cap_before_draw"] = skipped
    return trials
