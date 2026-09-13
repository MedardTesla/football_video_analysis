"""Péremption de l'homographie et couverture du match.

Sur un match réel, des intervalles de plusieurs minutes sans repère ont été
mesurés (ANALYSE_TERRAIN.md). Réutiliser l'homographie pendant tout ce temps
produit des distances inventées : c'est le défaut que ce module corrige.
"""
from __future__ import annotations

import numpy as np
import pytest

from football_analysis.analytics.stats import MatchStats
from football_analysis.pitch.view import HomographyCache, ViewTransformer

SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)


@pytest.fixture
def transformer():
    return ViewTransformer(source=SQUARE, target=SQUARE)


def test_fresh_homography_is_returned(transformer):
    cache = HomographyCache(max_age_frames=50)
    cache.update(transformer, frame_index=10)
    assert cache.get(10) is transformer
    assert cache.get(60) is transformer


def test_stale_homography_is_refused(transformer):
    cache = HomographyCache(max_age_frames=50)
    cache.update(transformer, frame_index=10)
    assert cache.get(61) is None


def test_cache_starts_empty():
    cache = HomographyCache(max_age_frames=50)
    assert cache.get(0) is None
    assert cache.age(0) is None


def test_refresh_resets_the_clock(transformer):
    cache = HomographyCache(max_age_frames=50)
    cache.update(transformer, frame_index=10)
    cache.update(transformer, frame_index=100)
    assert cache.get(140) is transformer
    assert cache.age(140) == 40


def test_coverage_counts_measured_frames():
    stats = MatchStats(fps=25.0)
    for _ in range(75):
        stats.mark_measured()
    for _ in range(25):
        stats.mark_unmeasured()
    assert stats.coverage == 0.75
    payload = stats.to_dict()
    assert payload["measured_seconds"] == 3.0
    assert payload["unmeasured_seconds"] == 1.0
    assert payload["total_seconds"] == 4.0


def test_gap_does_not_invent_distance():
    """Le cas qui motive tout ce module.

    Un joueur est vu à un bout du terrain, l'homographie se perd, il
    réapparaît à l'autre bout. Sans oubli des positions, la reprise compte le
    déplacement complet comme une course.
    """
    stats = MatchStats(fps=25.0)
    stats.mark_measured()
    stats.update_player(7, np.array([1000.0, 3000.0]), team=0)

    for _ in range(500):           # 20 secondes aveugles
        stats.mark_unmeasured()

    stats.mark_measured()
    stats.update_player(7, np.array([9000.0, 3000.0]), team=0)
    assert stats.players[7].distance_m == 0.0


def test_measurement_resumes_normally_after_a_gap():
    stats = MatchStats(fps=25.0)
    stats.update_player(7, np.array([0.0, 0.0]), team=0)
    stats.mark_unmeasured()
    stats.update_player(7, np.array([5000.0, 0.0]), team=0)   # oublié
    stats.update_player(7, np.array([5040.0, 0.0]), team=0)   # 0,4 m réels
    assert stats.players[7].distance_m == pytest.approx(0.4)


def test_coverage_is_zero_before_any_frame():
    assert MatchStats(fps=25.0).coverage == 0.0
