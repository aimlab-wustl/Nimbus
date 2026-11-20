"""nimbus.experiments.config: validation, overrides, hashing, round-trip (owned by experiments-cli)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.quick


from nimbus.core.errors import ConfigError
from nimbus.experiments.config import RunConfig, apply_overrides, config_digest, load_config, parse_set_value

MINIMAL = {
    "schema": "nimbus.run/1",
    "problem": {"kind": "toy_cardinality"},
    "transitions": {"library": "pair_exchanges"},
    "annealing": {"schedule": "constant", "params": {"tau": 0.5}, "attempts": 10},
}


def test_minimal_config_loads(tmp_path):
    cfg = load_config(MINIMAL, base_dir=tmp_path)
    assert cfg.data["schema"] == "nimbus.run/1"
    assert cfg.data["backend"]["name"] == "auto"
    assert cfg.data["run"]["replicas"] == 1
    assert cfg.data["run"]["seed"] == 0


def test_unknown_field_rejected(tmp_path):
    bad = dict(MINIMAL, bogus=1)
    with pytest.raises(ConfigError, match="unknown"):
        load_config(bad, base_dir=tmp_path)


def test_missing_required_section(tmp_path):
    bad = {k: v for k, v in MINIMAL.items() if k != "problem"}
    with pytest.raises(ConfigError):
        load_config(bad, base_dir=tmp_path)


def test_schedule_needs_params_or_auto(tmp_path):
    bad = dict(MINIMAL, annealing={"schedule": "constant", "attempts": 10})
    with pytest.raises(ConfigError):
        load_config(bad, base_dir=tmp_path)


def test_auto_schedule_needs_calibration_eta(tmp_path):
    bad = dict(MINIMAL, annealing={"schedule": "fn_endpoints", "params": "auto", "attempts": 10})
    with pytest.raises(ConfigError, match="eta"):
        load_config(bad, base_dir=tmp_path)


def test_stop_on_target_needs_targets(tmp_path):
    bad = dict(MINIMAL, run={"stop_on_target": True})
    with pytest.raises(ConfigError, match="targets"):
        load_config(bad, base_dir=tmp_path)


def test_oracle_rejects_batching(tmp_path):
    bad = dict(MINIMAL, backend={"name": "oracle"},
              annealing={"schedule": "constant", "params": {"tau": 0.1}, "attempts": 10,
                        "batching": {"capacity": 4}})
    with pytest.raises(ConfigError, match="oracle"):
        load_config(bad, base_dir=tmp_path)


@pytest.mark.parametrize("name", ["emulation", "cpu", "gpu", "auto"])
def test_backend_precision_float32_rejected(tmp_path, name):
    bad = dict(MINIMAL, backend={"name": name, "precision": "float32"})
    with pytest.raises(ConfigError, match="float32 is not supported"):
        load_config(bad, base_dir=tmp_path)


# ------------------------------------------------------------------------------------------------
# overrides
# ------------------------------------------------------------------------------------------------
def test_overrides_short_keys(tmp_path):
    cfg = load_config(MINIMAL, {"seed": 7, "replicas": 3}, base_dir=tmp_path)
    assert cfg.data["run"]["seed"] == 7
    assert cfg.data["run"]["replicas"] == 3
    assert any(a["key"] == "seed" for a in cfg.overrides)


def test_overrides_dotted_path(tmp_path):
    cfg = load_config(MINIMAL, {"run.seed": 9}, base_dir=tmp_path)
    assert cfg.data["run"]["seed"] == 9


def test_overrides_unknown_section_rejected():
    with pytest.raises(ConfigError):
        apply_overrides(MINIMAL, {"nope.x": 1})


def test_with_overrides_revalidates(tmp_path):
    cfg = load_config(MINIMAL, base_dir=tmp_path)
    cfg2 = cfg.with_overrides({"attempts": 999})
    assert cfg2.data["annealing"]["attempts"] == 999
    assert cfg.data["annealing"]["attempts"] == 10  # original untouched


def test_parse_set_value_types():
    assert parse_set_value("3") == 3
    assert parse_set_value("3.5") == 3.5
    assert parse_set_value("true") is True
    assert parse_set_value("[1, 2]") == [1, 2]
    assert parse_set_value("null") is None


# ------------------------------------------------------------------------------------------------
# hashing and round-trip
# ------------------------------------------------------------------------------------------------
def test_trajectory_hash_ignores_non_trajectory_fields(tmp_path):
    cfg = load_config(MINIMAL, base_dir=tmp_path)
    cfg2 = cfg.with_overrides({"output": "somewhere"})
    assert cfg.trajectory_hash() == cfg2.trajectory_hash()
    assert cfg.full_hash() != cfg2.full_hash()


def test_trajectory_hash_changes_with_seed(tmp_path):
    cfg = load_config(MINIMAL, base_dir=tmp_path)
    cfg2 = cfg.with_overrides({"seed": 123})
    assert cfg.trajectory_hash() != cfg2.trajectory_hash()


def test_config_digest_deterministic():
    assert config_digest({"a": 1, "b": [1, 2]}) == config_digest({"b": [1, 2], "a": 1})


def test_write_and_reload_resolved(tmp_path):
    cfg = load_config(MINIMAL, base_dir=tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    p = cfg.write_resolved(run_dir)
    assert p.exists()
    reloaded = RunConfig.from_resolved(p)
    assert reloaded.trajectory_hash() == cfg.trajectory_hash()


def test_duplicate_yaml_key_rejected(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text("schema: nimbus.run/1\nschema: nimbus.run/1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(p)


def test_relative_path_resolves_against_config_dir(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    p = sub / "cfg.yaml"
    import yaml

    p.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.base_dir == str(sub)
