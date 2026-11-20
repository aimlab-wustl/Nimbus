"""Event-driven experiment scripts: a tiny run, its window analysis and the size counts."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ED = Path(__file__).resolve().parents[1] / "experiments" / "event_driven"
sys.path.insert(0, str(ED))
sys.path.insert(0, str(ED.parent / "graph_matching"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tiny_event_run_and_analysis(tmp_path):
    run_event = _load("run_event", ED / "run_event.py")
    windows = _load("event_windows", ED / "analysis" / "event_windows.py")
    run_event.run_task("pair2d__0", 12, 4, 1, tmp_path)
    data = windows.load([tmp_path])
    summ, win = windows.analyse("pair2d", data["pair2d"])
    assert summ["identity_max_rel_dev"] < 1e-9           # ratio = a * rho in every window
    assert summ["rho_min"] == pytest.approx(2.0)         # single flips, order-2 terms
    assert np.all(np.diff(win.win_hi.values) > 0)


def test_program_size_counts():
    ps = _load("program_size", ED / "program_size.py")
    _, _, _, _, model = ps.build("surface:3:stabilizers", None)
    rec = ps.program_bits(model)
    assert rec["R"] == 8 and rec["routes"] == 48 and rec["M_obj"] == 27   # as recorded in the paper's counts
