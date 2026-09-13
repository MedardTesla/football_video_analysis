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


class HomographyCache:
    """Conserve la dernière homographie valide, et la périme.

    Sans repères sur une frame, réutiliser la dernière homographie connue est
    correct sur quelques secondes : la caméra a peu bougé. Sur un match réel,
    des intervalles de plusieurs minutes sans aucun repère ont été mesurés
    (voir ANALYSE_TERRAIN.md) — la caméra y panoramique et zoome plusieurs
    fois. Une homographie de douze minutes projette alors n'importe où, sans
    que rien ne le signale, et le pipeline produit des distances inventées.

    Cette classe rend `None` passé le délai. L'appelant doit alors marquer la
    frame comme non mesurée plutôt que de deviner.
    """

    def __init__(self, max_age_frames: int) -> None:
        self.max_age_frames = max_age_frames
        self._transformer: ViewTransformer | None = None
        self._set_at: int | None = None

    def update(self, transformer: ViewTransformer, frame_index: int) -> None:
        self._transformer = transformer
        self._set_at = frame_index

    def get(self, frame_index: int) -> ViewTransformer | None:
        """Homographie utilisable à cette frame, ou None si périmée."""
        if self._transformer is None or self._set_at is None:
            return None
        if frame_index - self._set_at > self.max_age_frames:
            return None
        return self._transformer

    def age(self, frame_index: int) -> int | None:
        """Nombre de frames depuis la dernière homographie valide."""
        if self._set_at is None:
            return None
        return frame_index - self._set_at
