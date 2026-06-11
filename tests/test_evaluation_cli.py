"""Unit tests for src/evaluation/cli.py (config merge + device resolution)."""

from __future__ import annotations

import torch

from src.evaluation.cli import merge_config, resolve_device


# ── merge_config ─────────────────────────────────────────────────────────────


def test_cli_overrides_yaml_and_default():
    merged = merge_config(
        cli_overrides={"device_index": 2},
        yaml_config={"device_index": 1},
        defaults={"device_index": 0},
    )
    assert merged["device_index"] == 2


def test_none_cli_falls_through_to_yaml():
    merged = merge_config(
        cli_overrides={"model_name": None},
        yaml_config={"model_name": "from_yaml"},
        defaults={"model_name": "from_default"},
    )
    assert merged["model_name"] == "from_yaml"


def test_missing_yaml_falls_through_to_default():
    merged = merge_config(
        cli_overrides={"downsampling_method": None},
        yaml_config={},
        defaults={"downsampling_method": "interpolation"},
    )
    assert merged["downsampling_method"] == "interpolation"


def test_key_only_in_cli_is_kept():
    merged = merge_config(
        cli_overrides={"verbose": True},
        yaml_config={},
        defaults=None,
    )
    assert merged["verbose"] is True


def test_unset_everywhere_is_none():
    merged = merge_config(
        cli_overrides={"foo": None},
        yaml_config={"bar": 1},
        defaults={"baz": 2},
    )
    # foo was None in CLI and absent from yaml/defaults -> not a key
    assert "foo" not in merged
    assert merged == {"bar": 1, "baz": 2}


def test_falsy_but_not_none_cli_wins():
    # 0 / "" / False are valid overrides; only None means "unset".
    merged = merge_config(
        cli_overrides={"device_index": 0},
        yaml_config={"device_index": 1},
        defaults={},
    )
    assert merged["device_index"] == 0


# ── resolve_device ───────────────────────────────────────────────────────────


def test_negative_index_forces_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert resolve_device(-1) == torch.device("cpu")


def test_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None) == torch.device("cpu")
    assert resolve_device(0) == torch.device("cpu")


def test_auto_picks_cuda0_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device(None) == torch.device("cuda:0")


def test_valid_index_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert resolve_device(2) == torch.device("cuda:2")


def test_out_of_range_index_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_device(3) == torch.device("cpu")
