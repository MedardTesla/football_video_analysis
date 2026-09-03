"""Lecture/écriture vidéo en flux.

L'ancienne implémentation chargeait toute la vidéo en mémoire (une liste de
frames), ce qui plafonne la durée traitable à quelques minutes. Ici tout est
générateur : la consommation mémoire ne dépend plus de la durée du clip.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int

    def resampled(self, stride: int) -> "VideoInfo":
        """Mêmes dimensions, cadence divisée.

        Sert à écrire la vidéo annotée : traitée une image sur `stride`, elle
        doit être écrite à la cadence réduite pour se lire à la bonne vitesse.
        """
        return VideoInfo(
            width=self.width,
            height=self.height,
            fps=self.fps / stride,
            total_frames=self.total_frames // stride,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "VideoInfo":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"vidéo illisible : {path}")
        try:
            return cls(
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
                total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
        finally:
            cap.release()


def frames(path: str | Path, stride: int = 1) -> Iterator[np.ndarray]:
    """Itère sur les frames. `stride > 1` sous-échantillonne (calibration)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"vidéo illisible : {path}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if index % stride == 0:
                yield frame
            index += 1
    finally:
        cap.release()


@contextmanager
def video_sink(path: str | Path, info: VideoInfo, codec: str = "mp4v"):
    """Écrit un flux de frames. mp4v plutôt que XVID/AVI : lisible partout,
    y compris dans un navigateur, ce qui compte pour la livraison web."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), info.fps, (info.width, info.height)
    )
    try:
        yield writer.write
    finally:
        writer.release()
