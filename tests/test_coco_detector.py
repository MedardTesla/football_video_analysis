"""Détecteur de secours COCO.

Ce qui est testé, c'est la traduction des classes : COCO ne connaît que
« personne » et « ballon de sport », le pipeline attend quatre classes.
"""
from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from football_analysis.config import BALL_ID, PLAYER_ID, DetectionConfig
from football_analysis.detection.coco import COCO_BALLON, COCO_PERSONNE, CocoDetector


class FauxYOLO:
    def __init__(self, classes):
        self.classes = classes

    def predict(self, lot, **kwargs):
        return [self] * len(lot)


@pytest.fixture
def detector(monkeypatch):
    d = CocoDetector.__new__(CocoDetector)
    d.config = DetectionConfig(batch_size=2)
    return d


def _detections(classes):
    n = len(classes)
    if n == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([[i * 50, 0, i * 50 + 40, 90] for i in range(n)], dtype=np.float32),
        confidence=np.full(n, 0.9, dtype=np.float32),
        class_id=np.array(classes),
    )


def test_persons_become_players(detector, monkeypatch):
    monkeypatch.setattr(
        sv.Detections, "from_ultralytics",
        staticmethod(lambda r: _detections([COCO_PERSONNE, COCO_PERSONNE])),
    )
    detector.model = FauxYOLO(None)
    _, sortie = next(detector.detect([np.zeros((90, 160, 3), np.uint8)]))
    assert set(sortie.class_id) == {PLAYER_ID}


def test_sports_balls_become_the_ball(detector, monkeypatch):
    monkeypatch.setattr(
        sv.Detections, "from_ultralytics",
        staticmethod(lambda r: _detections([COCO_PERSONNE, COCO_BALLON])),
    )
    detector.model = FauxYOLO(None)
    _, sortie = next(detector.detect([np.zeros((90, 160, 3), np.uint8)]))
    assert sorted(set(sortie.class_id)) == sorted({PLAYER_ID, BALL_ID})


def test_other_coco_classes_are_dropped(detector, monkeypatch):
    """Chaises, voitures et sacs abondent au bord d'un terrain."""
    monkeypatch.setattr(
        sv.Detections, "from_ultralytics",
        staticmethod(lambda r: _detections([COCO_PERSONNE, 56, 2, 24])),
    )
    detector.model = FauxYOLO(None)
    _, sortie = next(detector.detect([np.zeros((90, 160, 3), np.uint8)]))
    assert len(sortie) == 1
    assert sortie.class_id[0] == PLAYER_ID


def test_frames_are_returned_with_their_detections(detector, monkeypatch):
    monkeypatch.setattr(
        sv.Detections, "from_ultralytics",
        staticmethod(lambda r: _detections([COCO_PERSONNE])),
    )
    detector.model = FauxYOLO(None)
    frames = [np.full((90, 160, 3), i, np.uint8) for i in range(3)]
    sorties = list(detector.detect(frames))
    assert len(sorties) == 3
    for entree, (rendue, _) in zip(frames, sorties):
        assert np.array_equal(entree, rendue)


def test_no_detection_is_not_an_error(detector, monkeypatch):
    monkeypatch.setattr(
        sv.Detections, "from_ultralytics", staticmethod(lambda r: _detections([]))
    )
    detector.model = FauxYOLO(None)
    _, sortie = next(detector.detect([np.zeros((90, 160, 3), np.uint8)]))
    assert len(sortie) == 0
