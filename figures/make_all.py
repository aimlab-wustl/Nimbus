"""Draw every panel and table from ``source_data/`` into ``figures/out/``.

  python figures/make_all.py
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(HERE))
    scripts = sorted(p for p in HERE.glob("*.py") if p.name not in ("_style.py", "fig5_style.py", "event_style.py", "size_style.py", "make_all.py"))
    failed = []
    for p in scripts:
        print(f"--- {p.name}")
        try:
            runpy.run_path(str(p), run_name="__main__")
        except Exception as exc:  # report and continue with the other panels
            failed.append(p.name)
            print(f"FAILED {p.name}: {exc!r}")
    print(f"{len(scripts) - len(failed)} of {len(scripts)} scripts done" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
