"""Tendance d'un club sur plusieurs matchs.

Un match isolé dit peu ; c'est la progression sur une saison qu'un entraîneur
cherche, et ce qui justifie un abonnement plutôt qu'un paiement à l'acte.
"""
from __future__ import annotations

import pytest

from service import season
from service.jobs import Job, JobState


def _match(nom, jour, possession, controle, equipe=0, etat=JobState.DONE):
    return Job(
        id=nom, token="t", club="Club", match_name=nom, video_path="/tmp/v.mp4",
        club_id="c1", state=etat, our_team=equipe,
        created_at=f"2026-09-{jour:02d}T10:00:00+00:00",
        stats={
            "possession": {"0": possession, "1": round(1 - possession, 3)},
            "control": {"0": controle, "1": round(1 - controle, 3)},
            "coverage": 0.85,
        },
    )


def test_a_season_reads_oldest_first():
    """Une tendance se lit dans le sens du temps ; le magasin rend l'inverse."""
    s = season.build([_match("récent", 20, 0.6, 0.6), _match("ancien", 10, 0.4, 0.4)])
    assert [p.match_name for p in s.points] == ["ancien", "récent"]


def test_the_designated_team_decides_which_share_is_read():
    """Le cœur du problème : « équipe A » est une étiquette de regroupement,
    pas une identité. La lire à l'envers inverserait la courbe."""
    a = season.build([_match("m", 10, 0.7, 0.7, equipe=0)]).points[0]
    b = season.build([_match("m", 10, 0.7, 0.7, equipe=1)]).points[0]
    assert a.possession == pytest.approx(0.7)
    assert b.possession == pytest.approx(0.3)


def test_a_match_without_a_designated_team_is_excluded():
    """L'inclure inverserait la courbe une fois sur deux, sans prévenir."""
    s = season.build([
        _match("désigné", 10, 0.6, 0.6, equipe=0),
        _match("non désigné", 12, 0.4, 0.4, equipe=None),
    ])
    assert [p.match_name for p in s.points] == ["désigné"]


def test_an_unfinished_match_is_excluded():
    s = season.build([
        _match("fini", 10, 0.6, 0.6),
        _match("en cours", 12, 0.4, 0.4, etat=JobState.PROCESSING),
    ])
    assert len(s.points) == 1


def test_a_single_match_is_not_a_trend():
    assert not season.build([_match("seul", 10, 0.6, 0.6)]).usable
    assert season.build([_match("a", 10, 0.6, 0.6), _match("b", 12, 0.5, 0.5)]).usable


def test_averages_cover_the_season():
    s = season.build([_match("a", 10, 0.6, 0.7), _match("b", 12, 0.4, 0.5)])
    assert s.possession_moyenne == pytest.approx(0.5)
    assert s.control_moyen == pytest.approx(0.6)


def test_a_match_missing_a_statistic_does_not_break_the_average():
    incomplet = _match("incomplet", 12, 0.4, 0.4)
    incomplet.stats["control"] = {}
    s = season.build([_match("complet", 10, 0.6, 0.6), incomplet])
    assert s.control_moyen == pytest.approx(0.6)
    assert s.possession_moyenne == pytest.approx(0.5)


def test_an_empty_season_is_harmless():
    s = season.build([])
    assert not s.usable
    assert s.possession_moyenne is None
