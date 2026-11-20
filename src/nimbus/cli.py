"""``nimbus`` command-line entry point.

Subcommands: ``doctor``, ``problems``, ``inspect``, ``validate``, ``solve``, ``benchmarks fetch``,
``verify``, ``resume``, ``analyze``. ``--json`` makes a command print one machine-readable JSON
document on stdout instead of formatted text; ``--quiet`` suppresses the compact progress line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]


def _print(args, obj, text_fn):
    if getattr(args, "json", False):
        print(json.dumps(obj, indent=1, default=str))
    else:
        text_fn(obj)


def _cmd_doctor(args) -> int:
    from nimbus.backends import available_backends, get_backend
    from nimbus.core.errors import BackendUnavailable
    from nimbus.experiments.paths import data_root, runs_root

    report = {"backends": {}, "data_root": str(data_root()), "runs_root": str(runs_root())}
    for name in available_backends():
        try:
            get_backend(name)
            if name == "gpu":  # the gpu backend imports without CUDA; probe the device explicitly
                from nimbus.backends.gpu import _require_cuda

                _require_cuda()
            report["backends"][name] = "available"
        except BackendUnavailable as exc:
            report["backends"][name] = f"unavailable: {exc}"
    try:
        import numba  # noqa: F401

        report["numba"] = numba.__version__
    except ImportError:
        report["numba"] = "not installed"
    try:
        import cupy  # noqa: F401

        report["cupy"] = cupy.__version__
    except ImportError:
        report["cupy"] = "not installed"
    import os

    report["cpu_count"] = os.cpu_count()
    for p in (report["data_root"], report["runs_root"]):
        Path(p).mkdir(parents=True, exist_ok=True)
        report.setdefault("writable", {})[p] = os.access(p, os.W_OK)

    def show(r):
        print("nimbus doctor")
        for k, v in r["backends"].items():
            print(f"  backend {k:10s} {v}")
        print(f"  numba: {r['numba']}, cupy: {r['cupy']}, cpu_count: {r['cpu_count']}")
        print(f"  data_root: {r['data_root']}")
        print(f"  runs_root: {r['runs_root']}")

    _print(args, report, show)
    return 0


def _cmd_problems(args) -> int:
    from nimbus.problems.factory import problem_kinds, kind_status

    rows = {k: kind_status(k) for k in problem_kinds()}

    def show(r):
        for k, v in r.items():
            print(f"  {k:16s} {'ok' if v['available'] else 'unavailable'}  {v['detail']}")
    _print(args, {k: {"available": ok, "detail": d} for k, (ok, d) in rows.items()}, show)
    return 0


def _cmd_inspect(args) -> int:
    from nimbus.api import compile_problem

    problem, model, timings = compile_problem(args.config, _parse_overrides(args.set))

    def show(r):
        # the problem's own default library is not shown: the compiled library is on the next line
        prob = {k: (f"[{len(v)} values]" if isinstance(v, (list, tuple)) and len(v) > 16 else v)
                for k, v in r["problem"].items() if k != "library"}
        print(f"problem: {prob}")
        print(f"compiled: N={r['compile']['n_spins']} M={r['compile']['n_clauses']} R={r['compile']['n_groups']} "
              f"library={(r['compile'].get('library') or {}).get('name')} "
              f"kinds={(r['compile'].get('library') or {}).get('kinds')}")
        print(f"timings: {r['timings']}")
    out = {"problem": problem.metadata(), "compile": model.describe(), "timings": timings}
    _print(args, out, show)
    return 0


def _cmd_validate(args) -> int:
    from nimbus.experiments.config import load_config

    try:
        cfg = load_config(args.config, _parse_overrides(args.set))
    except Exception as exc:
        _print(args, {"ok": False, "error": str(exc)}, lambda r: print(f"INVALID: {r['error']}"))
        return 1
    out = {"ok": True, "trajectory_hash": cfg.trajectory_hash(), "full_hash": cfg.full_hash()}
    _print(args, out, lambda r: print(f"ok (trajectory_hash={r['trajectory_hash'][:12]})"))
    return 0


def _cmd_solve(args) -> int:
    from nimbus.experiments.runner import solve

    def progress(event, info):
        if not args.quiet and not args.json:
            print(f"[{event}] {info}", file=sys.stderr)

    outcome = solve(args.config, _parse_overrides(args.set), output_dir=args.output,
                    progress=None if args.quiet or args.json else progress)

    def show(r):
        print(f"status: {r['status']}  run_dir: {r['run_dir']}")
        print(f"rows: {r['rows']}  timings: {r['timings']}")
        if r["message"]:
            print(f"message: {r['message']}")
    out = {"status": outcome.status, "run_dir": outcome.run_dir, "rows": len(outcome.rows),
           "timings": outcome.timings, "message": outcome.message}
    _print(args, out, show)
    return 0 if outcome.status in ("ok", "target_hit", "budget_exhausted") else 1


def _cmd_resume(args) -> int:
    from nimbus.experiments.runner import resume

    outcome = resume(args.run_dir)
    _print(args, {"status": outcome.status, "run_dir": outcome.run_dir, "rows": len(outcome.rows)},
          lambda r: print(f"status: {r['status']}  run_dir: {r['run_dir']}"))
    return 0 if outcome.status in ("ok", "target_hit", "budget_exhausted") else 1


def _cmd_analyze(args) -> int:
    from nimbus.analysis.report import summarize_runs

    out = summarize_runs(args.run_dirs)
    _print(args, out, lambda r: print(json.dumps(r, indent=1, default=str)))
    return 0


def _cmd_benchmarks(args) -> int:
    from nimbus.experiments.fetch import main as fetch_main

    return fetch_main((["--set", args.set] if args.set else []) + (["--force"] if args.force else []))


def _cmd_verify(args) -> int:
    import pytest

    markers = {"quick": "quick", "acceptance": "acceptance", "public-smoke": "public_smoke"}[args.suite]
    return pytest.main(["-q", "-m", markers])


def _parse_overrides(pairs) -> dict:
    from nimbus.experiments.config import parse_set_value

    out = {}
    for item in pairs or ():
        if "=" not in item:
            raise SystemExit(f"--set expects path=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = parse_set_value(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nimbus")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor").set_defaults(fn=_cmd_doctor)

    sub.add_parser("problems").set_defaults(fn=_cmd_problems)

    sp = sub.add_parser("inspect")
    sp.add_argument("config")
    sp.add_argument("--set", action="append", default=[])
    sp.set_defaults(fn=_cmd_inspect)

    sp = sub.add_parser("validate")
    sp.add_argument("config")
    sp.add_argument("--set", action="append", default=[])
    sp.set_defaults(fn=_cmd_validate)

    sp = sub.add_parser("solve")
    sp.add_argument("config")
    sp.add_argument("--set", action="append", default=[])
    sp.add_argument("--output", default=None)
    sp.set_defaults(fn=_cmd_solve)

    sp = sub.add_parser("resume")
    sp.add_argument("run_dir")
    sp.set_defaults(fn=_cmd_resume)

    sp = sub.add_parser("analyze")
    sp.add_argument("run_dirs", nargs="+")
    sp.set_defaults(fn=_cmd_analyze)

    bp = sub.add_parser("benchmarks")
    bsub = bp.add_subparsers(dest="bcommand", required=True)
    fp = bsub.add_parser("fetch")
    fp.add_argument("--set", default=None)
    fp.add_argument("--force", action="store_true")
    fp.set_defaults(fn=_cmd_benchmarks)

    sp = sub.add_parser("verify")
    sp.add_argument("--suite", choices=("quick", "acceptance", "public-smoke"), default="quick")
    sp.set_defaults(fn=_cmd_verify)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.fn(args))
    except Exception as exc:  # CLI never raises a raw traceback for config/user errors
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
