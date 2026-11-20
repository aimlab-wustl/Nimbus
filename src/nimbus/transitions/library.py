"""Transition libraries: the ordered list of grouped transitions {s}_r -> {mu}_r, one transition
neuron each after compilation (Methods, Derivation of the transition-programmable architecture).

A library owns the physical spin actions, their labels and a description of the template that
built them (e.g. ``complete_row_exchange``, ``tree_row_exchange``, ``nullspace_basis``,
``chain_precedence``). It owns no temperature or logging policy. Group order is significant: it
defines the group index r used by the proposal stream.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence

import numpy as np

from nimbus.core.errors import UnsupportedTransition
from nimbus.transitions.descriptors import (
    KIND_NAMES,
    SpinAction,
    descriptor_from_dict,
)

__all__ = ["TransitionLibrary"]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class TransitionLibrary:
    """Ordered grouped transitions on ``n_spins`` spins.

    Parameters
    ----------
    n_spins: number of physical spins N.
    groups: sequence of SpinAction descriptors (one per transition group r).
    labels: optional human-readable labels (default: descriptor labels).
    name: library name (free text).
    template: generating template id, e.g. ``complete_row_exchange``.
    params: JSON-serializable template parameters (e.g. ``{"P": 5, "blocks": [...]}``).
    components: for concatenated libraries, the component metadata in order.
    """

    def __init__(self, n_spins: int, groups: Sequence[SpinAction], labels: Sequence[str] | None = None,
                 name: str = "library", template: str | None = None, params: dict | None = None,
                 components: list | None = None):
        self.n_spins = int(n_spins)
        if self.n_spins < 1:
            raise ValueError("a transition library needs n_spins >= 1")
        groups = tuple(groups)
        for r, g in enumerate(groups):
            if not isinstance(g, SpinAction):
                raise TypeError(f"group {r} is not a SpinAction descriptor: {g!r}")
            if g.max_index() >= self.n_spins:
                raise ValueError(f"group {r} ({g!r}) references a spin >= n_spins={self.n_spins}")
            n_tab = getattr(g, "n_spins", None)
            if n_tab is not None and n_tab != self.n_spins:
                raise ValueError(f"group {r}: table on {n_tab} spins in a library on {self.n_spins}")
        self.groups = groups
        if labels is None:
            labels = [g.default_label() for g in groups]
        labels = tuple(str(x) for x in labels)
        if len(labels) != len(groups):
            raise ValueError("labels and groups differ in length")
        self.labels = labels
        self.name = str(name)
        self.template = str(template) if template is not None else "custom"
        self.params = _jsonable(dict(params or {}))
        self.components = _jsonable(list(components or []))
        self._hash = None

    # -- container protocol ------------------------------------------------------------------
    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def __len__(self):
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)

    def __getitem__(self, r):
        return self.groups[r]

    def kinds(self) -> np.ndarray:
        return np.asarray([g.kind for g in self.groups], dtype=np.int8)

    def apply(self, r: int, spins):
        return self.groups[r].apply(spins)

    def touched_incidence(self) -> list[list[int]]:
        """spin -> sorted list of groups whose action may change that spin."""
        inc: list[list[int]] = [[] for _ in range(self.n_spins)]
        for r, g in enumerate(self.groups):
            for i in g.touched():
                inc[i].append(r)
        return inc

    @property
    def all_involutions(self) -> bool:
        return all(g.is_involution for g in self.groups)

    def missing_reverses(self) -> list[int]:
        """Groups that are not their own inverse and whose inverse is not in the library.

        A move that is not its own inverse (for example a cyclic exchange of three rows) needs its
        reverse as a separate move, so that every spin transition can be proposed in both directions.
        """
        have = {g.stable_hash() for g in self.groups}
        out = []
        for r, g in enumerate(self.groups):
            if g.is_involution:
                continue
            try:
                inv = g.inverse()
            except (NotImplementedError, ValueError, UnsupportedTransition):
                out.append(r)
                continue
            if inv.stable_hash() not in have:
                out.append(r)
        return out

    def with_reverses(self, name: str | None = None) -> "TransitionLibrary":
        """This library plus the reverse of every move that is not its own inverse (if missing).

        The reverse of move r follows the moves of the library, labelled ``"<label> (reverse)"``.
        Give a move and its reverse the same proposal weight, so that the proposal stays symmetric.
        """
        have = {g.stable_hash() for g in self.groups}
        groups, labels = list(self.groups), list(self.labels)
        for r in self.missing_reverses():
            inv = self.groups[r].inverse()
            if inv.stable_hash() not in have:
                groups.append(inv)
                labels.append(f"{self.labels[r]} (reverse)")
                have.add(inv.stable_hash())
        return TransitionLibrary(self.n_spins, groups, labels, name=name or f"{self.name}_with_reverses",
                                 template=self.template, params=self.params, components=self.components)

    # -- identity ------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"schema": "nimbus.library/1", "n_spins": self.n_spins, "name": self.name,
                "template": self.template, "params": self.params, "components": self.components,
                "labels": list(self.labels), "groups": [g.to_dict() for g in self.groups]}

    @classmethod
    def from_dict(cls, d: dict) -> "TransitionLibrary":
        return cls(d["n_spins"], [descriptor_from_dict(g) for g in d["groups"]], labels=d.get("labels"),
                   name=d.get("name", "library"), template=d.get("template"), params=d.get("params"),
                   components=d.get("components"))

    @property
    def hash(self) -> str:
        """sha256 over n_spins, ordered group descriptors, labels and template metadata."""
        if self._hash is None:
            h = hashlib.sha256()
            h.update(json.dumps({"n_spins": self.n_spins, "name": self.name, "template": self.template,
                                 "params": self.params, "components": self.components,
                                 "labels": list(self.labels)}, sort_keys=True,
                                separators=(",", ":")).encode())
            for g in self.groups:
                h.update(g.stable_hash().encode())
            self._hash = h.hexdigest()
        return self._hash

    @property
    def metadata(self) -> dict:
        kinds = {}
        for g in self.groups:
            kinds[KIND_NAMES[g.kind]] = kinds.get(KIND_NAMES[g.kind], 0) + 1
        return {"name": self.name, "template": self.template, "params": self.params,
                "components": self.components, "n_spins": self.n_spins, "n_groups": self.n_groups,
                "kinds": kinds, "all_involutions": self.all_involutions, "hash": self.hash}

    def __repr__(self):
        return f"TransitionLibrary(name={self.name!r}, template={self.template!r}, N={self.n_spins}, R={self.n_groups})"

    # -- composition ------------------------------------------------------------------------------
    @staticmethod
    def concat(*libraries: "TransitionLibrary", name: str | None = None, require_disjoint: bool = False,
               template: str = "union", params: dict | None = None) -> "TransitionLibrary":
        """Concatenate libraries on the same spins (groups in argument order).

        ``require_disjoint=True`` checks that the libraries act on disjoint blocks: no spin is
        touched by, or used as a condition in, groups of two different libraries.
        """
        libs = [lib for lib in libraries]
        if not libs:
            raise ValueError("concat needs at least one library")
        n = libs[0].n_spins
        if any(lib.n_spins != n for lib in libs):
            raise ValueError("cannot concatenate libraries on different numbers of spins")
        if require_disjoint:
            owner: dict[int, int] = {}
            for li, lib in enumerate(libs):
                used = set()
                for g in lib.groups:
                    used.update(g.touched())
                    used.update(getattr(g, "cond", ()))
                for i in used:
                    if owner.setdefault(i, li) != li:
                        raise ValueError(f"libraries {owner[i]} and {li} both use spin {i} (not disjoint)")
        groups, labels, comps = [], [], []
        for lib in libs:
            groups.extend(lib.groups)
            labels.extend(lib.labels)
            comps.append({"name": lib.name, "template": lib.template, "params": lib.params,
                          "n_groups": lib.n_groups, "hash": lib.hash})
        return TransitionLibrary(n, groups, labels, name=name or "+".join(lib.name for lib in libs),
                                 template=template, params=params, components=comps)

    union = concat

    def __add__(self, other: "TransitionLibrary") -> "TransitionLibrary":
        return TransitionLibrary.concat(self, other)

    def subset(self, indices: Iterable[int], name: str | None = None, template: str | None = None,
               params: dict | None = None) -> "TransitionLibrary":
        idx = [int(i) for i in indices]
        return TransitionLibrary(self.n_spins, [self.groups[i] for i in idx], [self.labels[i] for i in idx],
                                 name=name or f"{self.name}[subset]", template=template or f"{self.template}_subset",
                                 params=params if params is not None else {"parent": self.hash, "indices": idx})
