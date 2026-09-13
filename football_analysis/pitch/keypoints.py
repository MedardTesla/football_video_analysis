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
        """Points clés d'une frame. Rend des confiances nulles si rien n'est vu.

        Deux échelles d'inférence plutôt qu'une. Le modèle a appris une taille
        apparente de terrain ; un changement de zoom de la caméra l'en éloigne
        assez pour qu'il ne rende plus aucune instance — panne silencieuse, la
        frame est simplement marquée non mesurée. Le second essai n'a lieu que
        si le premier n'atteint pas `min_keypoints`, et on garde le meilleur
        des deux : agrandir sauve le plan large mais dégrade le plan serré.
        """
        points, conf = self._predire(frame, self.config.imgsz)
        retry = self.config.imgsz_retry
        if retry is None or retry == self.config.imgsz:
            return points, conf
        if self._exploitables(conf) >= self.config.min_keypoints:
            return points, conf
        autres_points, autres_conf = self._predire(frame, retry)
        if self._exploitables(autres_conf) > self._exploitables(conf):
            return autres_points, autres_conf
        return points, conf

    def _exploitables(self, confidence: np.ndarray) -> int:
        """Nombre de points assez confiants pour servir à l'homographie."""
        return int((confidence >= self.config.confidence).sum())

    def _predire(
        self, frame: np.ndarray, imgsz: int
    ) -> tuple[np.ndarray, np.ndarray]:
        result = self.model.predict(
            frame,
            conf=self.config.instance_confidence,
            imgsz=imgsz,
            verbose=False,
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
