"""Surface-code decoding with parallel tempering (replica exchange), and the reference decoders it is compared with.

Two decoders, each a bit-exact software model of an FPGA design:

- :mod:`nimbus.decoding.one_shot`: one syndrome per shot (code-capacity depolarizing noise), maximum-likelihood class
  from the occupancy of the beta = 1 replica;
- :mod:`nimbus.decoding.real_time`: sliding-window decoding of a syndrome stream (phenomenological depolarizing noise
  with measurement errors), committing rounds as they are decoded.

Shots and noise models:

- :mod:`nimbus.decoding.code_capacity`: code-capacity shots (:func:`~nimbus.decoding.code_capacity.sample_shots`) and
  logical classes of a correction;
- :mod:`nimbus.decoding.phenomenological`: the memory experiment (N noisy rounds plus one error-free round) and its
  shots (:func:`~nimbus.decoding.phenomenological.memory_shots`);
- :mod:`nimbus.decoding.windows`: the sliding-window commit and syndrome-forwarding logic for any window decoder.

CPU reference decoders (PyMatching, correlated matching, BP+matching, BP-OSD, union-find, tensor network) are in
:mod:`nimbus.decoding.baselines`; they need the optional ``refs`` dependencies.

Submodules are imported on first attribute access (``nimbus.decoding.one_shot`` etc.).
"""
from __future__ import annotations

import importlib

__all__ = ["one_shot", "real_time", "code_capacity", "phenomenological", "windows", "baselines"]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
