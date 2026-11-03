"""The search space of a model HPO: a YAML mapping, turned into Optuna suggestions
and then into a :class:`~src.schemas.configs.TrainingConfig`.

A space maps a parameter name to exactly one of

- ``{choices: [a, b, ...]}``: categorical;
- ``{int: [lo, hi]}``: integer, both ends included;
- ``{uniform: [lo, hi]}``: float;
- ``{loguniform: [lo, hi]}``: float, uniform in ``log``;

optionally with ``when: {other: value, ...}``: the parameter is sampled only when
every named parameter (listed earlier in the space) took that value, and is
otherwise ``otherwise`` (or left at the base configuration's value).

Names are :class:`TrainingConfig` fields, with two exceptions:

- ``warmup_epochs`` belongs to the run, not to the model (the linear warmup
  before the cosine decay, see
  :func:`src.active_learning.retrospective.lr_schedule.fixed_lr`);
- ``dim_feedforward_factor`` sets ``dim_feedforward`` to that multiple of the
  model's token size, which depends on the other parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional

import torch

from src.schemas.configs import TrainingConfig
from src.training.factory import build_adota_model

KINDS = ("choices", "int", "uniform", "loguniform")
RUN_PARAMS = ("warmup_epochs",)
FEEDFORWARD_FACTOR = "dim_feedforward_factor"


def _kind(name: str, spec: Mapping[str, Any]) -> str:
    found = [k for k in KINDS if k in spec]
    if len(found) != 1:
        raise ValueError(f"{name}: a parameter needs exactly one of {KINDS}, got {sorted(spec)}")
    return found[0]


def validate_space(space: Mapping[str, Mapping[str, Any]]) -> None:
    """Raise ``ValueError`` for a malformed space or an unknown parameter name."""
    known = {f.name for f in fields(TrainingConfig)} | set(RUN_PARAMS) | {FEEDFORWARD_FACTOR}
    seen = []
    for name, spec in space.items():
        if name not in known:
            raise ValueError(f"{name}: not a TrainingConfig field, {RUN_PARAMS} or {FEEDFORWARD_FACTOR}")
        kind = _kind(name, spec)
        values = spec[kind]
        if kind == "choices" and not values:
            raise ValueError(f"{name}: empty choices")
        if kind != "choices":
            lo, hi = values
            if not lo < hi or (kind == "loguniform" and lo <= 0):
                raise ValueError(f"{name}: invalid range {values}")
        for other in spec.get("when", {}):
            if other not in seen:
                raise ValueError(f"{name}: condition on {other!r}, which must be listed before it")
        seen.append(name)


def suggest(trial, space: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """One point of ``space`` from an Optuna trial, in the order of ``space``."""
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        condition = spec.get("when", {})
        if any(params.get(k) != v for k, v in condition.items()):
            if "otherwise" in spec:
                params[name] = spec["otherwise"]
            continue
        kind = _kind(name, spec)
        if kind == "choices":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif kind == "int":
            lo, hi = spec["int"]
            params[name] = trial.suggest_int(name, int(lo), int(hi))
        else:
            lo, hi = spec[kind]
            params[name] = trial.suggest_float(name, float(lo), float(hi), log=kind == "loguniform")
    return params


@dataclass
class Candidate:
    """A configuration ready to train, with what the search needs to know about it."""

    config: TrainingConfig
    warmup_epochs: int
    params: Dict[str, Any]
    n_params: int
    token_size: int


def build_candidate(base: TrainingConfig, params: Mapping[str, Any], default_warmup: int = 0) -> Candidate:
    """``base`` with ``params`` applied. The model is built on the ``meta`` device
    to read its token size and count its trainable parameters, so nothing is
    allocated.

    Raises:
        ValueError: ``num_heads`` does not divide the token size (the attention
            layers cannot be built).
    """
    model_fields = {k: v for k, v in params.items() if k not in RUN_PARAMS and k != FEEDFORWARD_FACTOR}
    config = replace(base, **model_fields)
    meta = torch.device("meta")
    token = int(build_adota_model(replace(config, num_transformers=0, dim_feedforward=None), meta).token_size)
    if config.num_transformers and token % int(config.num_heads):
        raise ValueError(f"num_heads={config.num_heads} does not divide the token size {token}")
    if FEEDFORWARD_FACTOR in params:
        config = replace(config, dim_feedforward=int(params[FEEDFORWARD_FACTOR]) * token)
    model = build_adota_model(config, meta)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return Candidate(config=config, warmup_epochs=int(params.get("warmup_epochs", default_warmup)),
                     params=dict(params), n_params=n_params, token_size=token)


def rejection(candidate: Candidate, max_params: Optional[int]) -> Optional[str]:
    """Why ``candidate`` may not be trained, or ``None``."""
    if max_params is not None and candidate.n_params > max_params:
        return f"{candidate.n_params:,} parameters over the cap of {max_params:,}"
    return None
