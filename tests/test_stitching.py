"""Recollement des pistes fragmentées.

Sur un extrait réel, le traqueur a produit 35 identités pour 22 joueurs,
chacune ne portant qu'une fraction de la distance parcourue.
"""
from __future__ import annotations

import numpy as np
import pytest

from football_analysis.tracking.stitching import Stitcher

FPS = 12.0


def _piste(s: Stitcher, track_id, debut, fin, depart, arrivee, team=0):
    s.observe(track_id, debut, np.array(depart, float), team)
    s.observe(track_id, fin, np.array(arrivee, float), team)


def test_two_close_segments_are_joined():
    """Le cas courant : un joueur masqué une seconde reparaît à côté."""
    s = Stitcher()
    _piste(s, 1, 0, 20, (1000, 3000), (1500, 3000))
    _piste(s, 2, 26, 50, (1700, 3000), (2200, 3000))   # 0,5 s d'écart, 2 m
    m = s.mapping(FPS)
    assert m[2] == 1


def test_an_impossible_jump_is_not_joined():
    """Deux joueurs différents, aux deux bouts du terrain."""
    s = Stitcher()
    _piste(s, 1, 0, 20, (1000, 3000), (1200, 3000))
    _piste(s, 2, 26, 50, (10000, 3000), (10200, 3000))
    assert s.mapping(FPS)[2] == 2


def test_a_long_gap_is_not_joined():
    """Au-delà de quelques secondes, un autre joueur a pu passer là."""
    s = Stitcher()
    _piste(s, 1, 0, 20, (1000, 3000), (1200, 3000))
    _piste(s, 2, 200, 240, (1300, 3000), (1500, 3000))
    assert s.mapping(FPS)[2] == 2


def test_overlapping_segments_are_never_joined():
    """Deux pistes visibles en même temps sont deux joueurs."""
    s = Stitcher()
    _piste(s, 1, 0, 60, (1000, 3000), (1200, 3000))
    _piste(s, 2, 30, 90, (1100, 3000), (1300, 3000))
    assert s.mapping(FPS)[2] == 2


def test_different_teams_are_never_joined():
    """Un maillot ne change pas en cours de match."""
    s = Stitcher()
    _piste(s, 1, 0, 20, (1000, 3000), (1200, 3000), team=0)
    _piste(s, 2, 26, 50, (1250, 3000), (1400, 3000), team=1)
    assert s.mapping(FPS)[2] == 2


def test_an_unknown_team_does_not_block_joining():
    """Le classifieur laisse des joueurs non attribués ; les exclure du
    recollement les priverait de leur distance."""
    s = Stitcher()
    _piste(s, 1, 0, 20, (1000, 3000), (1200, 3000), team=0)
    _piste(s, 2, 26, 50, (1250, 3000), (1400, 3000), team=None)
    assert s.mapping(FPS)[2] == 1


def test_a_player_lost_twice_keeps_one_identity():
    """Chaînage transitif : A se recolle à B, B à C."""
    s = Stitcher()
    _piste(s, 1, 0, 20, (1000, 3000), (1200, 3000))
    _piste(s, 2, 26, 46, (1250, 3000), (1450, 3000))
    _piste(s, 3, 52, 72, (1500, 3000), (1700, 3000))
    m = s.mapping(FPS)
    assert m[2] == 1 and m[3] == 1


def test_the_nearest_candidate_wins():
    """Deux joueurs perdus au même instant : chacun reprend le plus proche."""
    s = Stitcher()
    _piste(s, 1, 0, 20, (1000, 3000), (1000, 3000))
    _piste(s, 2, 0, 20, (5000, 3000), (5000, 3000))
    _piste(s, 3, 26, 46, (5100, 3000), (5200, 3000))
    assert s.mapping(FPS)[3] == 2


def test_an_empty_stitcher_maps_nothing():
    assert Stitcher().mapping(FPS) == {}


def test_a_single_track_maps_to_itself():
    s = Stitcher()
    _piste(s, 7, 0, 20, (1000, 3000), (1200, 3000))
    assert s.mapping(FPS) == {7: 7}
