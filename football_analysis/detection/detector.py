"""Détection YOLOv8 des joueurs, gardiens, arbitres et du ballon."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import supervision as sv

from ..config import DetectionConfig


class Detector:
    def __init__(self, config: DetectionConfig) -> None:
        from ultralytics import YOLO

        if not Path(config.weights).exists():
            raise FileNotFoundError(
                f"poids introuvables : {config.weights}\n"
                "Voir data/README.md pour l'obtention des modèles."
            )
        self.config = config
        self.model = YOLO(str(config.weights))

    def detect(
        self, frames: Iterable[np.ndarray]
    ) -> Iterator[tuple[np.ndarray, sv.Detections]]:
        """Détecte par lots, en flux, et rend `(frame, détections)`.

        Rendre la frame avec ses détections évite à l'appelant d'ouvrir un
        second flux vidéo pour les récupérer — le décodage est le deuxième
        poste de coût après l'inférence.
        """
        batch: list[np.ndarray] = []
        for frame in frames:
            batch.append(frame)
            if len(batch) == self.config.batch_size:
                yield from self._run(batch)
                batch = []
        if batch:
            yield from self._run(batch)

    def _run(
        self, batch: list[np.ndarray]
    ) -> Iterator[tuple[np.ndarray, sv.Detections]]:
        results = self.model.predict(
            batch,
            conf=self.config.confidence,
            imgsz=self.config.imgsz,
            verbose=False,
        )
        for frame, result in zip(batch, results):
            detections = sv.Detections.from_ultralytics(result)
            # NMS agnostique de classe : le même joueur sort souvent à la fois
            # en `player` et en `goalkeeper`. Sans cela, il compte deux fois.
            if self.config.class_agnostic_nms:
                detections = detections.with_nms(
                    threshold=self.config.nms_iou, class_agnostic=True
                )
            yield frame, detections
