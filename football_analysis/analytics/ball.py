"""Nettoyage et lissage de la trajectoire du ballon, en coordonnées terrain.

Le ballon est exclu de ByteTrack (trop petit, trop rapide) : sa position vient
des détections brutes, avec les faux positifs que cela implique. Ce module
filtre en travaillant dans l'espace métrique du terrain, où une contrainte
physique simple ("un ballon ne saute pas de 20 m en une frame") est exprimable.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class BallTrajectory:
    """Rejette les sauts impossibles, puis lisse par moyenne glissante."""

    def __init__(self, max_displacement_cm: float = 500.0, window: int = 5) -> None:
        self.max_displacement_cm = max_displacement_cm
        self._raw: deque[np.ndarray] = deque(maxlen=window)
        self._last_accepted: np.ndarray | None = None

    def update(self, position: np.ndarray | None) -> np.ndarray | None:
        """Ajoute une position terrain (cm, forme (2,)) et rend la valeur lissée.

        `None` en entrée = ballon non détecté sur cette frame ; on ne propage
        pas l'ancienne position pour ne pas inventer de mouvement.
        """
        if position is None:
            return self._smoothed()

        if self._last_accepted is not None:
            jump = float(np.linalg.norm(position - self._last_accepted))
            if jump > self.max_displacement_cm:
                # Déplacement physiquement impossible : détection rejetée.
                return self._smoothed()

        self._last_accepted = position
        self._raw.append(position)
        return self._smoothed()

    def _smoothed(self) -> np.ndarray | None:
        if not self._raw:
            return None
        return np.mean(self._raw, axis=0)

    def reset(self) -> None:
        """À appeler sur un changement de plan / coupure caméra."""
        self._raw.clear()
        self._last_accepted = None
