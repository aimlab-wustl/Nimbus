"""Penalty machines, the calibrated schedule and the shortcut libraries reproduce the reference outputs of the code
that produced the paper's graph-matching results, bit for bit at equal seeds.

The reference (``fixtures/matching/dev_reference.json``) was recorded with the original implementation on the
fixture tensors ``five.npz`` (L = 5) and ``seven.npz`` (L = 7).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from nimbus.annealing.calibration import CalibrationWarning
from nimbus.annealing.protocol import calibrate_temperatures, scaled_temperatures
from nimbus.baselines.penalty_machines import PenaltyMachine
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights
from nimbus.problems.formats.matching_tensor import load_tensor
from nimbus.problems.matching import MatchingProblem, describe_library, shortcut_library

FIX = Path(__file__).parent / "fixtures" / "matching"
REF = json.loads((FIX / "dev_reference.json").read_text())
FILES = sorted(REF)


def _prob(fname):
    t = load_tensor(FIX / fname)
    return t, MatchingProblem(t)


def _nan(x):
    return [np.nan if v is None else v for v in x]


@pytest.mark.parametrize("fname", FILES)
def test_penalty_machines_match_reference(fname):
    t, prob = _prob(fname)
    for ref in REF[fname]["penalty"]:
        pm = PenaltyMachine(prob, ref["kind"], ref["lam"], ref["mult"])
        assert (pm.n_aux, pm.ntot) == (ref["n_aux"], ref["ntot"])
        r = pm.run(ref["budget"], ref["T_start"], ref["T_end"], ref["replicas"], ref["seed"], threads=2)
        np.testing.assert_array_equal(r["bestF"], np.array(_nan(ref["bestF"]), dtype=float))
        assert r["best_event"].tolist() == ref["best_event"]
        assert r["final_feasible"].tolist() == ref["final_feasible"]
        assert r["accepted"].tolist() == ref["accepted"]
        assert (r["K"], r["tests"]) == (ref["K"], ref["tests"])
        assert r["drift"] < 1e-9


@pytest.mark.parametrize("fname", FILES)
def test_shortcut_libraries_match_reference(fname):
    t, prob = _prob(fname)
    for ref in REF[fname]["libraries"]:
        lib, info = shortcut_library(prob, ref["spec"], t)
        assert describe_library(ref["spec"]) == ref["describe"]
        assert int(lib.n_groups) == ref["n_groups"]
        assert list(lib.labels) == ref["labels"]
        assert (info["copies"], info["R_nodup"]) == (ref["copies"], ref["R_nodup"])
        assert [list(p) for p in info.get("pairs", [])] == ref["pairs"]


@pytest.mark.parametrize("fname", FILES)
def test_calibration_matches_reference(fname):
    t, prob = _prob(fname)
    models = {}
    for ref in REF[fname]["calibration"]:
        key = json.dumps(ref["spec"], sort_keys=True)
        if key not in models:
            lib, _ = shortcut_library(prob, ref["spec"], t)
            model = compile_model(prob.structure(), lib, identities=tuple(prob.identities()))
            models[key] = (model, bind_weights(model, prob.term_weights(0), prob.const(0)))
        model, binding = models[key]
        kw = dict(ref["kwargs"])
        args = dict(min_ratio=kw.pop("kappa", None), min_ratio_mode=kw.pop("kappa_mode", "lower_end"),
                    late=kw.pop("late", True), raw=kw.pop("raw", False))
        assert not kw
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CalibrationWarning)
            ts, te = calibrate_temperatures(model, binding, prob, ref["seed"], **args)
        assert (ts, te) == (ref["tau_start"], ref["tau_end"])


@pytest.mark.parametrize("fname", FILES)
def test_scaled_temperatures_with_m1_equal_lower_end_rule(fname):
    """m = 1 applied to the raw calibration equals the calibration option min_ratio = kappa, lower_end."""
    raw = {json.dumps(r["spec"], sort_keys=True): r for r in REF[fname]["calibration"] if r["kwargs"].get("raw")}
    k2 = {json.dumps(r["spec"], sort_keys=True): r for r in REF[fname]["calibration"]
          if r["kwargs"].get("kappa") == 2.0 and "kappa_mode" not in r["kwargs"]}
    for key, r in raw.items():
        ts, te = scaled_temperatures(r["tau_start"], r["tau_end"], 1.0, 2.0)
        assert (ts, min(te, ts)) == (k2[key]["tau_start"], k2[key]["tau_end"])


def test_path_library_name_and_tree_alias():
    t, prob = _prob("five.npz")
    path = prob.transition_library("path")
    assert path.n_groups == 4
    with pytest.warns(DeprecationWarning):
        tree = prob.transition_library("tree")
    assert list(tree.labels) == list(path.labels)
    assert prob.library_names() == ["complete", "path"]


def test_config_accepts_min_ratio_fields():
    from nimbus.annealing.policy import CalibrationSpec

    spec = CalibrationSpec.from_dict({"min_ratio": 2, "min_ratio_mode": "lower_end"})
    kw = spec.to_kwargs()
    assert kw["min_ratio"] == 2.0 and kw["min_ratio_mode"] == "lower_end"
    from nimbus.core.errors import ConfigError

    with pytest.raises(ConfigError):
        CalibrationSpec.from_dict({"min_ratio": 0.5})
    with pytest.raises(ConfigError):
        CalibrationSpec.from_dict({"min_ratio_mode": "other"})
