"""Unit tests for src/evaluation/cli.py (config merge, --set overrides, device resolution)."""

from __future__ import annotations

import pytest
import torch

from src.evaluation.cli import apply_set_overrides, merge_config, resolve_device

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


# ── apply_set_overrides ──────────────────────────────────────────────────────


def test_set_top_level_key_with_yaml_scalar_parsing():
    out = apply_set_overrides({"n_cycles": 5, "data_fraction": 0.3},
                              ["n_cycles=2", "data_fraction=1.0", "max_records=400"])
    assert out == {"n_cycles": 2, "data_fraction": 1.0, "max_records": 400}
    assert isinstance(out["n_cycles"], int) and isinstance(out["data_fraction"], float)


def test_set_dotted_key_reaches_nested_mapping():
    cfg = {"training": {"compile": True, "allow_tf32": True}, "scorer": {"n_workers": 12}}
    out = apply_set_overrides(cfg, ["training.compile=false", "scorer.n_workers=8"])
    assert out["training"] == {"compile": False, "allow_tf32": True}
    assert out["scorer"]["n_workers"] == 8


def test_set_creates_missing_intermediate_mappings():
    out = apply_set_overrides({"loop": {"n_cycles": 2}},
                              ["loop.train_overrides.compile=false",
                               "loop.train_overrides.num_epochs=1"])
    assert out["loop"] == {"n_cycles": 2,
                           "train_overrides": {"compile": False, "num_epochs": 1}}


def test_set_parses_null_lists_and_quoted_strings():
    out = apply_set_overrides({}, ["arm=null", "energies=[80.0, 105.0]",
                                   "variant='full (ridge)'", "path=/scratch/x/y", "empty="])
    assert out["arm"] is None
    assert out["energies"] == [80.0, 105.0]
    assert out["variant"] == "full (ridge)"
    assert out["path"] == "/scratch/x/y"
    assert out["empty"] is None


def test_set_does_not_mutate_input_and_later_entries_win():
    cfg = {"training": {"compile": True}}
    out = apply_set_overrides(cfg, ["training.compile=false", "training.compile=true"])
    assert cfg == {"training": {"compile": True}}
    assert out["training"]["compile"] is True


def test_set_no_overrides_is_a_copy():
    cfg = {"a": {"b": 1}}
    out = apply_set_overrides(cfg, [])
    assert out == cfg and out is not cfg and out["a"] is not cfg["a"]


@pytest.mark.parametrize("entry", ["n_cycles", "=5", "  =5", "a..b=1", ".a=1"])
def test_set_refuses_malformed_entries(entry):
    with pytest.raises(ValueError, match="--set"):
        apply_set_overrides({}, [entry])


def test_set_refuses_walking_through_a_non_mapping():
    with pytest.raises(ValueError, match="not a mapping"):
        apply_set_overrides({"n_cycles": 5}, ["n_cycles.x=1"])


def test_set_value_may_contain_equals_sign():
    out = apply_set_overrides({}, ["comment=a=b"])
    assert out["comment"] == "a=b"


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
