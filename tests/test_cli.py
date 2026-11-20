"""nimbus.cli: subcommands against the toy example (owned by experiments-cli)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.quick

import json


from nimbus.cli import main

TOY_YAML = """
schema: nimbus.run/1
name: toy_cli
problem: {kind: toy_cardinality}
transitions: {library: pair_exchanges}
annealing: {schedule: geometric, params: {tau_start: 2.0, tau_end: 0.05}, attempts: 300}
run: {seed: 1, replicas: 1}
backend: {name: emulation}
"""


@pytest.fixture
def toy_config(tmp_path):
    p = tmp_path / "toy.yaml"
    p.write_text(TOY_YAML, encoding="utf-8")
    return p


def test_doctor_json(capsys):
    assert main(["--json", "doctor"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "emulation" in out["backends"]
    assert out["backends"]["emulation"] == "available"


def test_problems_json(capsys):
    assert main(["--json", "problems"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["toy_cardinality"]["available"] is True


def test_validate_ok(capsys, toy_config):
    assert main(["--json", "validate", str(toy_config)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert "trajectory_hash" in out


def test_validate_invalid_reports_error_and_exit_1(capsys, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: nimbus.run/1\nproblem: {kind: toy_cardinality}\n", encoding="utf-8")
    code = main(["--json", "validate", str(bad)])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["ok"] is False


def test_inspect(capsys, toy_config):
    assert main(["--json", "inspect", str(toy_config)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["problem"]["n_spins"] == 6
    assert out["compile"]["n_spins"] == 6


def test_solve_and_analyze(capsys, toy_config, tmp_path):
    out_dir = tmp_path / "run"
    code = main(["--json", "--quiet", "solve", str(toy_config), "--output", str(out_dir)])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["status"] == "ok"
    assert out["rows"] == 1

    code2 = main(["--json", "analyze", str(out_dir)])
    report = json.loads(capsys.readouterr().out)
    assert code2 == 0
    assert report["n_runs"] == 1
    assert report["runs"][0]["n_feasible"] == 1


def test_solve_then_resume(capsys, toy_config, tmp_path):
    out_dir = tmp_path / "run"
    main(["--json", "--quiet", "solve", str(toy_config), "--output", str(out_dir),
         "--set", "output.keep_checkpoint=true"])
    capsys.readouterr()
    code = main(["--json", "resume", str(out_dir)])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["status"] == "ok"


def test_unknown_config_path_reports_error(capsys):
    code = main(["--json", "validate", "no/such/file.yaml"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["ok"] is False


def test_solve_text_mode_prints_status(capsys, toy_config, tmp_path):
    code = main(["--quiet", "solve", str(toy_config), "--output", str(tmp_path / "run")])
    text = capsys.readouterr().out
    assert code == 0
    assert "status: ok" in text
