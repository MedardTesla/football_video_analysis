"""Détection des 32 points clés du terrain (YOLOv8-pose)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..config import PitchConfig
from .geometry import PITCH
from .view import ViewTransformer


class PitchKeypointDetector:
    def __init__(self, config: PitchConfig) -> None:
        from ultralytics import YOLO

        if not Path(config.weights).exists():
            raise FileNotFoundError(
                f"modèle de keypoints introuvable : {config.weights}\n"
                "Voir ROADMAP.md phase 2 pour l'entraînement."
            )
        self.config = config
        self.model = YOLO(str(config.weights))

    def detect_one(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Points clés d'une frame. Rend des confiances nulles si rien n'est vu."""
        result = self.model.predict(
            frame, conf=self.config.instance_confidence, verbose=False
        )[0]
        kp = result.keypoints
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            return np.zeros((32, 2)), np.zeros(32)
        points = kp.xy[0].cpu().numpy()
        conf = kp.conf[0].cpu().numpy() if kp.conf is not None else np.ones(len(points))
        return points, conf

    def detect(
        self, frames: Iterable[np.ndarray]
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for frame in frames:
            yield self.detect_one(frame)


def transformer_from_keypoints(
    points: np.ndarray, confidence: np.ndarray, config: PitchConfig
) -> ViewTransformer | None:
    """Construit l'homographie à partir des keypoints suffisamment confiants.

    Rend `None` si trop peu de points sont visibles — cas courant en caméra
    basse ou en plan serré. L'appelant doit alors réutiliser la dernière
    homographie valide plutôt que d'abandonner la frame.
    """
    mask = confidence >= config.confidence
    if int(mask.sum()) < config.min_keypoints:
        return None

    target = np.array(PITCH.vertices, dtype=np.float32)
    try:
        return ViewTransformer(source=points[mask], target=target[mask])
    except ValueError:
        # Points visibles mais colinéaires : homographie dégénérée.
        return None
