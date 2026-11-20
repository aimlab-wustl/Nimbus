"""``python -m nimbus`` entry point (same as the ``nimbus`` console script)."""

from nimbus.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
