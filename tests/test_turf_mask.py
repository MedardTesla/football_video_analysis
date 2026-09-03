"""Masque de pelouse : écarter le staff et les spectateurs.

Sur une caméra de bord de touche, un tiers des personnes détectées ne sont
pas des joueurs. Les compter fausse possession, Voronoï et effectifs.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from football_analysis.pitch.mask import on_pitch, playable_area, turf_mask

H, W = 400, 700
HORIZON = 150  # au-dessus : gradins ; en dessous : pelouse


@pytest.fixture
def stadium():
    """Gradins gris en haut, pelouse verte en bas, ligne blanche au milieu."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:HORIZON] = (120, 120, 125)          # béton
    frame[HORIZON:] = (60, 160, 70)            # gazon (BGR)
    cv2.line(frame, (0, 300), (W, 300), (245, 245, 245), 3)
    return frame


def test_mask_covers_the_turf_not_the_stands(stadium):
    mask = turf_mask(stadium)
    assert mask[350, 350] > 0        # pelouse
    assert mask[40, 350] == 0        # gradins


def test_white_lines_do_not_split_the_pitch(stadium):
    """Sans fermeture morphologique, une ligne coupe la pelouse en deux."""
    mask = playable_area(stadium)
    assert mask[290, 350] > 0
    assert mask[310, 350] > 0


def test_players_do_not_punch_holes(stadium):
    """Un joueur masque l'herbe sous ses pieds ; il doit rester sur le terrain."""
    cv2.rectangle(stadium, (330, 250), (360, 320), (40, 40, 90), cv2.FILLED)
    mask = playable_area(stadium)
    assert mask[300, 345] > 0


def test_staff_on_the_touchline_is_rejected(stadium):
    mask = playable_area(stadium)
    boxes = np.array([
        [340, 250, 370, 330],    # joueur, pieds sur la pelouse
        [200, 40, 230, 120],     # spectateur dans les gradins
    ], dtype=float)
    keep = on_pitch(mask, boxes)
    assert keep.tolist() == [True, False]


def test_feet_decide_not_the_centre(stadium):
    """Un entraîneur debout au bord a le torse au-dessus de la pelouse."""
    mask = playable_area(stadium)
    # Boîte à cheval sur l'horizon, pieds dans les gradins.
    boxes = np.array([[300, 60, 330, HORIZON - 5]], dtype=float)
    assert on_pitch(mask, boxes).tolist() == [False]


def test_empty_detections_are_handled(stadium):
    assert on_pitch(playable_area(stadium), np.empty((0, 4))).shape == (0,)


def test_frame_without_turf_does_not_crash():
    grey = np.full((H, W, 3), 120, dtype=np.uint8)
    mask = playable_area(grey)
    assert mask.shape == (H, W)
    assert mask.sum() == 0
