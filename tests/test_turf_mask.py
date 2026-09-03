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


def test_turf_hue_is_estimated_not_hardcoded():
    """Le masque doit suivre la teinte réelle, pas un seuil figé.

    Deux stades filmés à des heures différentes n'ont pas la même teinte de
    gazon ; un seuil calé sur l'un dérive sur l'autre.
    """
    from football_analysis.pitch.mask import dominant_turf_hue

    for bgr in [(60, 160, 70), (40, 120, 90), (35, 95, 45)]:
        frame = np.full((H, W, 3), bgr, dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Référence calculée par OpenCV, pas posée à la main.
        assert dominant_turf_hue(hsv) == int(hsv[0, 0, 0]), bgr


def test_two_turf_shades_both_segment():
    """Une pelouse claire et une pelouse sombre doivent toutes deux marcher."""
    for bgr in [(60, 160, 70), (35, 95, 45)]:
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:HORIZON] = (120, 120, 125)
        frame[HORIZON:] = bgr
        mask = playable_area(frame)
        assert mask[350, 350] > 0, bgr
        assert mask[40, 350] == 0, bgr


def test_off_pitch_vegetation_is_excluded():
    """Buissons hors terrain : même famille de couleur, teinte plus jaune.

    Le vrai piège du terrain filmé : la végétation derrière la clôture est
    verte elle aussi. Elle est séparée du gazon par un mur, et sa teinte
    diffère d'une dizaine de degrés.
    """
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:80] = (60, 150, 150)      # végétation vert-jaune
    frame[80:140] = (150, 150, 155)  # mur de séparation
    frame[140:] = (60, 160, 70)      # pelouse
    mask = playable_area(frame)
    assert mask[300, 350] > 0    # pelouse retenue
    assert mask[40, 350] == 0    # végétation écartée
