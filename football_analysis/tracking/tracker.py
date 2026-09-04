"""Suivi des personnes. Le ballon est traité séparément.

Le suivi associe des boîtes d'une frame à l'autre. Le ballon est trop petit et
trop rapide pour que l'IoU entre deux frames consécutives soit non nul : on
l'exclut et on lisse sa trajectoire ailleurs (`analytics.ball.BallTrajectory`).

On utilise le paquet `trackers` de Roboflow plutôt que `sv.ByteTrack`, déprécié
depuis supervision 0.28 et supprimé en 0.31. BoT-SORT y est le défaut : il
intègre une compensation du mouvement de caméra (CMC), indispensable quand la
caméra balaie le terrain — sans elle, un panoramique déplace toutes les boîtes
d'un coup et le traqueur perd les identités.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import supervision as sv

from ..config import BALL_ID, TrackingConfig

# Rendu comme identifiant de piste tant qu'une détection n'est pas confirmée.
# Ces détections existent et doivent être dessinées, mais ne constituent pas
# une identité : les compter fusionnerait tous les joueurs non associés en un
# seul, qui cumulerait leurs distances.
UNTRACKED = -1


class PersonTracker:
    def __init__(
        self, config: TrackingConfig, algorithm: Literal["botsort", "bytetrack"] = "botsort"
    ) -> None:
        from trackers import BoTSORTTracker, ByteTrackTracker

        common = dict(
            lost_track_buffer=config.lost_track_buffer,
            frame_rate=config.frame_rate,
            track_activation_threshold=config.track_activation_threshold,
        )
        if algorithm == "botsort":
            self.tracker = BoTSORTTracker(enable_cmc=True, **common)
        else:
            self.tracker = ByteTrackTracker(**common)

    def update(
        self, detections: sv.Detections, frame: np.ndarray | None = None
    ) -> tuple[sv.Detections, sv.Detections]:
        """Sépare ballon et personnes, rend (personnes suivies, ballon).

        `frame` alimente la compensation de mouvement caméra ; l'omettre
        dégrade fortement le suivi sur plan mobile.

        Les gardiens conservent leur classe d'origine : leur équipe est
        décidée par `teams.classifier.assign_goalkeeper`, ce qui suppose de
        savoir lesquels sont gardiens.
        """
        ball = detections[detections.class_id == BALL_ID]
        people = detections[detections.class_id != BALL_ID]
        return self.tracker.update(people, frame=frame), ball

    def reset(self) -> None:
        """À appeler sur un changement de plan."""
        self.tracker.reset()


def is_tracked(detections: sv.Detections) -> np.ndarray:
    """Masque des détections portant un identifiant de piste confirmé."""
    if detections.tracker_id is None or len(detections) == 0:
        return np.zeros(len(detections), dtype=bool)
    return np.asarray(detections.tracker_id) > UNTRACKED


def ball_position(ball: sv.Detections) -> np.ndarray | None:
    """Point au sol du ballon : centre bas de la boîte la plus confiante."""
    if len(ball) == 0:
        return None
    best = int(np.argmax(ball.confidence))
    x1, _, x2, y2 = ball.xyxy[best]
    return np.array([(x1 + x2) / 2, y2])
