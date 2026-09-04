"""Détecteur de secours, à partir d'un YOLO générique COCO.

Sert à faire tourner le pipeline sans modèle spécialisé — pour valider une
intégration, ou tester une vidéo avant d'investir dans l'entraînement.

Deux limites à connaître, et elles ne sont pas contournables :

- COCO ne connaît que « personne ». Gardiens et arbitres sont donc classés
  joueurs. L'affectation d'équipe par proximité au centroïde ne s'applique
  plus, et les arbitres polluent les effectifs — le cluster supplémentaire de
  `TeamClassifier` en absorbe une partie, pas la totalité.
- La classe « ballon de sport » de COCO n'est pas entraînée sur des ballons de
  football lointains : le rappel mesuré est de 7 images sur 10 en 1080p, mais
  d'une seule sur cinq en 720p.

À ne pas utiliser en production : les statistiques produites sont indicatives.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
import supervision as sv

from ..config import BALL_ID, PLAYER_ID, DetectionConfig

COCO_PERSONNE = 0
COCO_BALLON = 32


class CocoDetector:
    """Même protocole que `Detector`, sur des poids COCO génériques."""

    def __init__(self, config: DetectionConfig, weights: str = "yolov8m.pt") -> None:
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(weights)

    def detect(
        self, frames: Iterable[np.ndarray]
    ) -> Iterator[tuple[np.ndarray, sv.Detections]]:
        lot: list[np.ndarray] = []
        for frame in frames:
            lot.append(frame)
            if len(lot) == self.config.batch_size:
                yield from self._run(lot)
                lot = []
        if lot:
            yield from self._run(lot)

    def _run(self, lot: list[np.ndarray]) -> Iterator[tuple[np.ndarray, sv.Detections]]:
        resultats = self.model.predict(
            lot, conf=self.config.confidence, imgsz=self.config.imgsz, verbose=False
        )
        for frame, resultat in zip(lot, resultats):
            detections = sv.Detections.from_ultralytics(resultat)
            garde = np.isin(detections.class_id, (COCO_PERSONNE, COCO_BALLON))
            detections = detections[garde]
            detections.class_id = np.where(
                detections.class_id == COCO_BALLON, BALL_ID, PLAYER_ID
            )
            if self.config.class_agnostic_nms and len(detections):
                detections = detections.with_nms(
                    threshold=self.config.nms_iou, class_agnostic=True
                )
            yield frame, detections
