"""ObjectiveStructure, CompiledModel validation/immutability/persistence, WeightBinding and bind_weights."""

import dataclasses
import itertools
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from helpers_core import onehot_identities, permutation_states, random_distances, cyclic_assignment_objective
from nimbus.compiler import compile_model
from nimbus.core.errors import FormatError
from nimbus.core.model import CompiledModel, ObjectiveStructure, WeightBinding, bind_weights, load_model
from nimbus.transitions import builders as B


def _model():
    P = 4
    structure, w, c = cyclic_assignment_objective(random_distances(P, 5))
    lib = B.row_exchanges(P, "complete")
    return compile_model(structure, lib, identities=onehot_identities(P), readout="all"), structure, w, c, lib


def test_objective_structure_canonical():
    s = ObjectiveStructure(n_spins=5, terms=[(3, 1, 3), (2, 2), (), (4,)], labels=["a", "b", "c", "d"])
    assert s.terms == ((1,), (), (), (4,))
    assert s.unique_supports() == [(1,), (4,)]
    assert s.max_degree == 1 and s.n_terms == 4
    assert ObjectiveStructure.from_dict(s.to_dict()).hash == s.hash
    with pytest.raises(ValueError):
        ObjectiveStructure(n_spins=3, terms=[(3,)])
    with pytest.raises(ValueError):
        ObjectiveStructure(n_spins=3, terms=[(0,)], labels=["x", "y"])


def test_model_arrays_readonly_and_dtypes():
    model, *_ = _model()
    for name, dt in (("clause_ptr", np.int64), ("clause_spins", np.int32), ("clause_role", np.uint8),
                     ("route_clause", np.int32), ("route_ctrl", np.int32), ("route_pol", np.int8),
                     ("act_kind", np.int8), ("clause_wroute", np.int64), ("term_to_clause", np.int32)):
        a = getattr(model, name)
        assert a.dtype == dt and a.flags.c_contiguous and not a.flags.writeable
    with pytest.raises(ValueError):
        model.route_clause[0] = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.n_spins = 3


@pytest.mark.parametrize("mutation", ["clause_range", "unsorted_support", "route_order", "dependency", "pol",
                                      "readout_map", "act_bijection", "ttc_role"])
def test_model_validation_rejects_corruption(mutation):
    model, *_ = _model()
    kw = {f.name: getattr(model, f.name) for f in dataclasses.fields(model)}
    kw["topology_hash"] = ""
    if mutation == "clause_range":
        cs = model.clause_spins.copy()
        cs[0] = model.n_spins
        kw["clause_spins"] = cs
    elif mutation == "unsorted_support":
        k = int(np.flatnonzero(np.diff(model.clause_ptr) >= 2)[0])
        cs = model.clause_spins.copy()
        a = int(model.clause_ptr[k])
        cs[a], cs[a + 1] = cs[a + 1], cs[a]
        kw["clause_spins"] = cs
    elif mutation == "route_order":
        rc = model.route_clause.copy()
        rc[0], rc[1] = rc[1], rc[0]
        kw["route_clause"] = rc
    elif mutation == "dependency":
        cw = model.clause_wroute.copy()
        cw[0] += 1
        kw["clause_wroute"] = cw
    elif mutation == "pol":
        pol = model.route_pol.copy()
        pol[np.flatnonzero(model.route_ctrl >= 0)[0]] = 0
        kw["route_pol"] = pol
    elif mutation == "readout_map":
        m = model.readout_clause_of_spin.copy()
        m[0] = m[5]
        kw["readout_clause_of_spin"] = m
    elif mutation == "act_bijection":
        src = model.act_src.copy()
        src[0] = src[1]
        kw["act_src"] = src
    elif mutation == "ttc_role":
        t = model.term_to_clause.copy()
        gate = int(np.flatnonzero((model.clause_role & 1) == 0)[0])
        t[0] = gate
        kw["term_to_clause"] = t
    with pytest.raises(ValueError):
        CompiledModel(**kw)


def test_save_load_roundtrip_and_tamper(tmp_path):
    model, structure, w, c, lib = _model()
    npz, js = model.save(tmp_path / "m")
    assert os.path.exists(npz) and os.path.exists(js)
    back = load_model(tmp_path / "m.npz")
    assert back.topology_hash == model.topology_hash
    for name, arr in model.arrays().items():
        assert np.array_equal(getattr(back, name), arr)
    assert back.group_label == model.group_label and back.identities == model.identities
    assert back.metadata["resources"]["materialized"] == model.metadata["resources"]["materialized"]
    S, _ = permutation_states(4)
    b = bind_weights(model, w, c)
    assert np.allclose(back.energy(S, bind_weights(back, w, c)), model.energy(S, b))
    # tamper with the recorded hash
    meta = json.loads(open(js, encoding="utf-8").read())
    meta["topology_hash"] = "0" * 64
    with open(js, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with pytest.raises(FormatError):
        load_model(tmp_path / "m")
    # tamper with an array (re-save arrays with one changed route polarity)
    model.save(tmp_path / "m2")
    with np.load(tmp_path / "m2.npz") as z:
        arrays = {k: z[k] for k in z.files}
    pos = int(np.flatnonzero(arrays["route_ctrl"] >= 0)[0])
    arrays["route_pol"][pos] *= -1
    np.savez_compressed(tmp_path / "m2.npz", **arrays)
    with pytest.raises(FormatError, match="hash"):
        load_model(tmp_path / "m2")
    with pytest.raises(FormatError):
        load_model(tmp_path / "missing")


def test_hashes_stable_across_processes():
    """Topology and weight hashes do not depend on the interpreter session (e.g. hash randomization)."""
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, r"%s")
        from helpers_core import onehot_identities, random_distances, cyclic_assignment_objective
        from nimbus.compiler import compile_model
        from nimbus.core.model import bind_weights
        from nimbus.transitions import builders as B
        s, w, c = cyclic_assignment_objective(random_distances(5, 9))
        m = compile_model(s, B.row_exchanges(5, "random_tree", seed=4) + B.segment_reversals(5),
                          identities=onehot_identities(5), readout="all")
        print(m.topology_hash, bind_weights(m, w, c).weight_hash)
    """) % os.path.dirname(os.path.abspath(__file__))
    outs = set()
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        outs.add(r.stdout.strip())
    assert len(outs) == 1
    s, w, c = cyclic_assignment_objective(random_distances(5, 9))
    m = compile_model(s, B.row_exchanges(5, "random_tree", seed=4) + B.segment_reversals(5),
                      identities=onehot_identities(5), readout="all")
    assert outs == {f"{m.topology_hash} {bind_weights(m, w, c).weight_hash}"}


def test_bind_weights_semantics(tmp_path):
    s = ObjectiveStructure(n_spins=3, terms=[(0,), (1, 0), (0,), (), (2, 2), (1,)])
    model = compile_model(s, B.single_flips(3))
    assert model.term_to_clause.tolist() == [0, 1, 0, -1, -1, 2]
    b = bind_weights(model, [1.0, 2.0, 0.5, 3.0, 4.0, -1.0], const=0.25)
    assert b.J[0].tolist() == [1.5, 2.0, -1.0, 0.0]  # + the zero-weight singleton of s2
    # constant terms () and (2,2) have T = 1: under E = E_const - sum J T they contribute -J
    assert b.const[0] == pytest.approx(0.25 - 3.0 - 4.0)
    X = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=np.int8)
    direct = 0.25 - np.array([sum(wt * np.prod(x[list(t)]) for wt, t in
                                  zip([1.0, 2.0, 0.5, 3.0, 4.0, -1.0], [(0,), (1, 0), (0,), (), (2, 2), (1,)]))
                              for x in X.astype(np.int64)])
    assert np.allclose(model.energy(X, b), direct)
    assert b.slot_names == ("run",) and b.topology_hash == model.topology_hash
    b2 = bind_weights(model, [1.0, 2.0, 0.5, 3.0, 4.0, -1.5], const=0.25)
    assert b2.weight_hash != b.weight_hash
    b.save(tmp_path / "b")
    back = WeightBinding.load(tmp_path / "b", model=model)
    assert back.weight_hash == b.weight_hash and np.array_equal(back.J, b.J)
    with pytest.raises(ValueError):
        bind_weights(model, [1.0, 2.0])  # wrong number of term weights
    with pytest.raises(ValueError):
        bind_weights(model, [1.0, 2.0, 0.5, 3.0, 4.0, np.nan])
    with pytest.raises(ValueError):
        bind_weights(model, [0.0] * 6, extra_slots={"run": ([0.0] * 6, 0.0)})
    other = compile_model(ObjectiveStructure(n_spins=3, terms=[(0,)]), B.single_flips(3))
    with pytest.raises(ValueError):
        b.validate_for(other)
    with pytest.raises(ValueError):
        WeightBinding(slot_names=("objective",), J=np.zeros((1, 3)), const=[0.0], topology_hash="x")
    # extra slot over the objective structure's terms
    b3 = bind_weights(model, [1.0] * 6, extra_slots={"objective": ([2.0] * 6, 1.0)})
    assert b3.slot_index("objective") == 1 and b3.J[1].tolist() == [4.0, 2.0, 2.0, 0.0]  # + zero-weight singleton


def test_describe_and_resources():
    model, *_ = _model()
    d = model.describe()
    assert d["n_clauses"] == model.n_clauses and d["topology_hash"] == model.topology_hash
    assert d["library"]["template"] == "complete_row_exchange"
    res = model.resources
    mat = res["materialized"]
    for key in ("M", "M_add", "M_read", "M_total", "R", "role_overlaps", "routes_weighted", "routes_decoder",
                "routes_gate_controlled", "routes_constant", "support_storage", "fan_in_mean", "fan_in_max",
                "control_fan_out_mean", "control_fan_out_max", "readout_routes", "memory_bytes", "compile_seconds",
                "closure_rounds"):
        assert key in mat
    assert mat["memory_bytes"] == model.nbytes
    assert res["estimated"]["estimated"] is True
    assert "compile_host" in res and "compiled_at_utc" in res
