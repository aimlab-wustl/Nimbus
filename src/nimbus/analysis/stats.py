"""Statistics over stored results (Methods, Benchmark problems and simulation methodology).

Time-to-solution (TTS) with censoring: let ``p`` be the per-run success probability and ``t`` the
per-run time/attempts. ``tts(p, t, p_target)`` reports the attempts needed for a ``p_target``
cumulative success probability by independent restarts:

    TTS = t * log(1 - p_target) / log(1 - p)      (0 < p < 1)

``p == 0`` has no finite TTS (reported as a lower bound, ``inf``, with the bound explained);
``p == 1`` needs exactly one run (``TTS = t``). Wilson and Clopper-Pearson intervals give the
uncertainty of a success count over a finite number of runs; the bootstrap resamples whole
*instances* (not individual replicas) so censoring and instance-to-instance variance both show up
in the interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["tts", "wilson_interval", "clopper_pearson_interval", "bootstrap_over_instances",
           "hitting_time_distribution"]


@dataclass
class TTSResult:
    tts: float
    p: float
    t: float
    p_target: float
    censored: bool
    note: str = ""


def tts(p: float, t: float, p_target: float = 0.99) -> TTSResult:
    p, t, p_target = float(p), float(t), float(p_target)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if p <= 0.0:
        return TTSResult(tts=math.inf, p=p, t=t, p_target=p_target, censored=True,
                         note="no successes observed: TTS is unbounded (reporting a lower bound of inf); "
                              "run longer or more restarts before trusting a finite estimate")
    if p >= 1.0:
        return TTSResult(tts=t, p=p, t=t, p_target=p_target, censored=False, note="p=1: one run suffices")
    value = t * math.log1p(-p_target) / math.log1p(-p)
    return TTSResult(tts=value, p=p, t=t, p_target=p_target, censored=False)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, (center - half) / denom), min(1.0, (center + half) / denom))


def clopper_pearson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval, via the regularized incomplete beta
    function (``scipy.stats.beta.ppf``, optional dependency used lazily)."""
    from scipy import stats as sstats

    if n <= 0:
        return (0.0, 1.0)
    k = int(successes)
    lo = 0.0 if k == 0 else sstats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else sstats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def bootstrap_over_instances(values_by_instance, statistic=np.mean, *, n_boot: int = 2000,
                             seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """``(point_estimate, ci_lo, ci_hi)`` of ``statistic`` over per-instance values, bootstrapped by
    resampling whole instances (each instance's value(s) kept together)."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(v, dtype=np.float64) for v in values_by_instance]
    n = len(groups)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(np.concatenate(groups)))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = statistic(np.concatenate([groups[i] for i in idx]))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def hitting_time_distribution(hit_attempts, n_total: int) -> dict:
    """Empirical hitting-time distribution over runs; ``hit_attempts`` has ``None``/``nan`` for runs
    that never hit (censored at ``n_total``)."""
    arr = np.asarray([a if a is not None else np.nan for a in hit_attempts], dtype=np.float64)
    hit = arr[~np.isnan(arr)]
    n = arr.size
    return {"n_runs": int(n), "n_hit": int(hit.size), "p_hit": float(hit.size / n) if n else float("nan"),
           "median_hit_attempt": float(np.median(hit)) if hit.size else None,
           "mean_hit_attempt": float(np.mean(hit)) if hit.size else None, "n_total": int(n_total)}
