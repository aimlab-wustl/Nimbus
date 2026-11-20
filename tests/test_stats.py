"""nimbus.analysis.stats: TTS/censoring, intervals, bootstrap, BER/FER (owned by experiments-cli)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.quick

import math

import numpy as np

from nimbus.analysis import stats


def test_tts_basic_formula():
    r = stats.tts(0.5, 100.0, 0.99)
    expected = 100.0 * math.log(0.01) / math.log(0.5)
    assert r.tts == pytest.approx(expected)
    assert not r.censored


def test_tts_zero_success_is_censored_inf():
    r = stats.tts(0.0, 100.0)
    assert math.isinf(r.tts)
    assert r.censored


def test_tts_full_success_needs_one_run():
    r = stats.tts(1.0, 42.0)
    assert r.tts == 42.0
    assert not r.censored


def test_tts_rejects_out_of_range_p():
    with pytest.raises(ValueError):
        stats.tts(1.5, 10.0)


def test_wilson_interval_contains_point_estimate():
    lo, hi = stats.wilson_interval(8, 10)
    assert lo < 0.8 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_degenerate_n_zero():
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)


def test_clopper_pearson_interval_contains_point_estimate():
    lo, hi = stats.clopper_pearson_interval(8, 10)
    assert lo < 0.8 < hi


def test_clopper_pearson_zero_successes():
    lo, hi = stats.clopper_pearson_interval(0, 10)
    assert lo == 0.0
    assert hi > 0.0


def test_clopper_pearson_all_successes():
    lo, hi = stats.clopper_pearson_interval(10, 10)
    assert hi == 1.0
    assert lo < 1.0


def test_bootstrap_over_instances_point_estimate():
    groups = [[1.0, 2.0], [3.0], [4.0, 4.0]]
    point, lo, hi = stats.bootstrap_over_instances(groups, n_boot=500, seed=0)
    assert lo <= point <= hi


def test_bootstrap_empty_groups():
    point, lo, hi = stats.bootstrap_over_instances([], n_boot=10)
    assert math.isnan(point)


def test_hitting_time_distribution_mixed_censoring():
    out = stats.hitting_time_distribution([10, None, 20, None], n_total=100)
    assert out["n_runs"] == 4
    assert out["n_hit"] == 2
    assert out["p_hit"] == pytest.approx(0.5)
    assert out["median_hit_attempt"] == pytest.approx(15.0)


def test_hitting_time_distribution_never_hit():
    out = stats.hitting_time_distribution([None, None], n_total=50)
    assert out["n_hit"] == 0
    assert out["median_hit_attempt"] is None
