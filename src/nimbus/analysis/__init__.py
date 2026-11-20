"""Analysis of stored results only: TTS/BER-FER statistics, figures, reports.

Nothing here re-runs a solver; every function consumes ``results.parquet``/``manifest.json`` written
by :mod:`nimbus.experiments.runner`.
"""

from nimbus.analysis import figures, report, stats

__all__ = ["stats", "figures", "report"]
