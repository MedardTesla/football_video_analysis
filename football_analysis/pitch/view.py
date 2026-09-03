"""Transformation de perspective bidirectionnelle image <-> terrain."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class ViewTransformer:
    """Homographie entre le plan image et le plan terrain (en cm).

    `source` et `target` sont des points correspondants ; l'ordre importe.
    Utiliser `frame_to_pitch` pour le radar 2D, `pitch_to_frame` pour
    projeter des lignes tactiques virtuelles sur la vidéo.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if source.shape != target.shape:
            raise ValueError("source et target doivent avoir la même forme")
        if source.shape[0] < 4:
            raise ValueError("au moins 4 correspondances sont requises")

        source = source.astype(np.float32)
        target = target.astype(np.float32)
        # RANSAC plutôt que la méthode par défaut : les keypoints prédits
        # contiennent régulièrement un ou deux points aberrants.
        self.m, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if self.m is None:
            raise ValueError("homographie non calculable (points dégénérés)")
        self.m_inv = np.linalg.inv(self.m)

    def _apply(self, points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(reshaped, matrix).reshape(-1, 2)

    def frame_to_pitch(self, points: np.ndarray) -> np.ndarray:
        """Positions image (px) -> coordonnées terrain (cm), pour le radar."""
        return self._apply(points, self.m)

    def pitch_to_frame(self, points: np.ndarray) -> np.ndarray:
        """Coordonnées terrain (cm) -> positions image (px), pour l'overlay."""
        return self._apply(points, self.m_inv)


class SmoothedHomography:
    """Moyenne glissante de la matrice d'homographie.

    Les keypoints prédits vibrent d'une frame à l'autre ; sans lissage le
    radar tremble. On moyenne les matrices normalisées (m[2,2] == 1) sur une
    fenêtre, ce qui suffit tant que la caméra bouge lentement.
    """

    def __init__(self, window: int = 5) -> None:
        self._window: deque[np.ndarray] = deque(maxlen=window)

    def update(self, transformer: ViewTransformer) -> np.ndarray:
        m = transformer.m / transformer.m[2, 2]
        self._window.append(m)
        return np.mean(self._window, axis=0)

    @property
    def ready(self) -> bool:
        return len(self._window) > 0
