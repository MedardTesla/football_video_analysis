"""Interface de détection.

Ultralytics est en AGPL-3.0. Le prototype l'utilise, mais la version
commerciale devra probablement passer à un modèle Apache-2.0 (RF-DETR,
RT-DETR, YOLOX). Tout le pipeline ne dépend que de ce protocole : la migration
se limite alors à écrire une nouvelle classe ici, sans toucher au reste.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Protocol

import numpy as np
import supervision as sv


class ObjectDetector(Protocol):
    """Rend `(frame, détections)` par frame, dans l'ordre de l'entrée."""

    def detect(
        self, frames: Iterable[np.ndarray]
    ) -> Iterator[tuple[np.ndarray, sv.Detections]]: ...


class KeypointDetector(Protocol):
    """Rend les 32 points clés du terrain par frame.

    Retourne `(points, confiances)` de formes (32, 2) et (32,). Les points sous
    le seuil de confiance sont à ignorer par l'appelant.
    """

    def detect(
        self, frames: Iterable[np.ndarray]
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
