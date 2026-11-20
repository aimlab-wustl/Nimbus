"""Problem-kind registry: config ``problem.kind`` -> problem adapter.

A problem adapter module exposes ``PROBLEM_KIND`` (a kind name or a tuple of names) and
``from_config(cfg, base_dir) -> Problem``, where ``cfg`` is the configuration's ``problem`` mapping
(including ``kind``) and ``base_dir`` is the directory that relative input paths resolve against
(the configuration file's directory). Adapters validate their own fields strictly.

Built-in kinds are imported lazily (importing this module imports no adapter):

==================  ===============================================  =================================
kind                module                                           constraint family
==================  ===============================================  =================================
toy_cardinality     nimbus.problems.toy_cardinality                 exact cardinality (result toy)
surface_code        nimbus.problems.surface_code                    binary affine parity (CSS surface code)
matching            nimbus.problems.matching                        one-to-one assignment, cubic objective
spinpoly            nimbus.problems.constrained_blocks              constraint blocks (project schema)
opb                 nimbus.problems.constrained_blocks              constraint blocks (OPB subset)
==================  ===============================================  =================================

User code can add kinds with :func:`register_problem_kind` (a factory callable, or a module path
``"pkg.mod"`` / ``"pkg.mod:function"``). A kind whose module is missing raises
:class:`ProblemKindUnavailable` (a :class:`ConfigError`), never a silent substitute.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Mapping

from nimbus.core.errors import ConfigError

__all__ = ["BUILTIN_KINDS", "ProblemKindUnavailable", "register_problem_kind", "unregister_problem_kind",
           "problem_kinds", "problem_factory", "kind_status", "make_problem"]

#: built-in kinds -> adapter module (both block formats are served by the constrained-blocks adapter)
BUILTIN_KINDS: dict[str, str] = {
    "toy_cardinality": "nimbus.problems.toy_cardinality",
    "surface_code": "nimbus.problems.surface_code",
    "matching": "nimbus.problems.matching",
    "spinpoly": "nimbus.problems.constrained_blocks",
    "opb": "nimbus.problems.constrained_blocks",
}

_REGISTRY: dict[str, Any] = {}


class ProblemKindUnavailable(ConfigError):
    """The kind is known but its adapter cannot be imported here (module missing or broken)."""


def register_problem_kind(kind: str, factory, *, replace: bool = False) -> None:
    """Register ``kind``. ``factory`` is ``f(cfg, base_dir) -> Problem`` or a module path string."""
    kind = str(kind)
    if not kind or not kind.replace("_", "").replace("-", "").isalnum():
        raise ConfigError(f"invalid problem kind name {kind!r}")
    if not (callable(factory) or isinstance(factory, str)):
        raise ConfigError("problem factory must be callable or a module path string")
    if kind in _REGISTRY and not replace and _REGISTRY[kind] is not factory:
        raise ConfigError(f"problem kind {kind!r} is already registered (pass replace=True)")
    _REGISTRY[kind] = factory


def unregister_problem_kind(kind: str) -> None:
    _REGISTRY.pop(str(kind), None)


def problem_kinds() -> list[str]:
    """Registered plus built-in kind names (built-in does not mean importable here; see kind_status)."""
    return sorted(set(_REGISTRY) | set(BUILTIN_KINDS))


def _resolve_target(target: str) -> Callable:
    mod_name, _, func = target.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ProblemKindUnavailable(f"problem adapter module {mod_name!r} cannot be imported ({exc})") from exc
    fn = getattr(mod, func or "from_config", None)
    if not callable(fn):
        raise ProblemKindUnavailable(f"problem adapter {mod_name!r} has no callable {func or 'from_config'!r}")
    return fn


def problem_factory(kind: str) -> Callable:
    """The ``f(cfg, base_dir)`` for a kind (imports its module on first use)."""
    kind = str(kind)
    if kind in _REGISTRY:
        f = _REGISTRY[kind]
        return _resolve_target(f) if isinstance(f, str) else f
    if kind in BUILTIN_KINDS:
        fn = _resolve_target(BUILTIN_KINDS[kind])
        mod = importlib.import_module(BUILTIN_KINDS[kind])
        declared = getattr(mod, "PROBLEM_KIND", None)
        names = (declared,) if isinstance(declared, str) else tuple(declared or ())
        if names and kind not in names:
            raise ProblemKindUnavailable(f"module {BUILTIN_KINDS[kind]} declares PROBLEM_KIND={declared!r}, "
                                         f"which does not include {kind!r}")
        return fn
    raise ConfigError(f"unknown problem kind {kind!r}; known kinds: {problem_kinds()}")


def kind_status(kind: str) -> tuple[bool, str]:
    """(importable, module or reason) without building a problem."""
    try:
        f = problem_factory(kind)
    except ConfigError as exc:
        return False, str(exc)
    return True, getattr(f, "__module__", repr(f))


def make_problem(cfg: Mapping, base_dir=None):
    """Build the problem adapter for a ``problem`` config mapping (must contain ``kind``)."""
    from nimbus.problems.base import Problem

    if not isinstance(cfg, Mapping):
        raise ConfigError("problem: must be a mapping with a 'kind' field")
    if "kind" not in cfg:
        raise ConfigError("problem.kind is required")
    kind = cfg["kind"]
    fn = problem_factory(kind)
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    prob = fn(dict(cfg), base)
    if not isinstance(prob, Problem):
        raise ConfigError(f"problem kind {kind!r}: factory returned {type(prob).__name__}, not a Problem")
    return prob
