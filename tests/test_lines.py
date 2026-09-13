"""Primitives de détection de lignes.

Le module ne rend aucun verdict — sa calibration a échoué sur vidéo réelle.
Ces tests couvrent la géométrie, qui est déterministe et vérifiable.
"""
from __future__ import annotations

import numpy as np

from football_analysis.pitch.lines import intersections, merge_segments


def test_collinear_fragments_become_one_line():
    """Un joueur qui traverse une ligne la coupe en morceaux pour Hough.

    Sans fusion, ces morceaux seraient comptés comme autant de lignes et la
    structure visible serait massivement surestimée.
    """
    fragments = np.array([
        [0, 100, 200, 100],
        [220, 100, 400, 100],
        [430, 101, 600, 101],
    ])
    assert len(merge_segments(fragments)) == 1


def test_distinct_lines_are_kept_apart():
    lines = np.array([
        [0, 100, 600, 100],     # horizontale
        [0, 400, 600, 400],     # horizontale, loin
        [300, 0, 300, 500],     # verticale
    ])
    assert len(merge_segments(lines)) == 3


def test_parallel_lines_never_intersect():
    lines = [np.array([0, 100, 600, 100]), np.array([0, 300, 600, 300])]
    assert intersections(lines, (500, 700)) == []


def test_crossing_lines_give_the_expected_point():
    lines = [np.array([0, 200, 600, 200]), np.array([300, 0, 300, 500])]
    points = intersections(lines, (500, 700))
    assert len(points) == 1
    assert points[0] == (300.0, 200.0)


def test_intersections_outside_the_frame_are_dropped():
    """Deux lignes quasi parallèles se croisent très loin : sans rejet, on
    compterait un repère qui n'est pas dans l'image."""
    # Pente de 1 px sur 600 : l'intersection tombe à x = 6000, hors d'une
    # image large de 700.
    lines = [np.array([0, 100, 600, 100]), np.array([0, 110, 600, 109])]
    assert intersections(lines, (500, 700)) == []
