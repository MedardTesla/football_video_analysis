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

import logging
import shutil
import subprocess

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Largeur de la vidéo livrée au club. Mesuré sur des images réelles : en
# 1080p avec le codec mp4v d'OpenCV, un match de 90 minutes pèse 3,5 Go —
# intéléchargeable sur une connexion mobile. En 1280 px et H.264, il pèse
# 0,37 Go pour une lisibilité qui reste suffisante à l'écran d'un téléphone.
LARGEUR_LIVREE = 1280

# Qualité H.264. 26 conserve les annotations nettes ; 28 gagne encore un tiers
# de poids mais fait baver les numéros de maillot au ralenti.
CRF = 26


def _ffmpeg_binaire() -> str | None:
    """Chemin d'un ffmpeg utilisable, ou None.

    OpenCV ne sait pas encoder en H.264 dans les distributions courantes :
    `avc1` et `H264` échouent silencieusement à l'ouverture et le fichier
    reste vide. On passe donc par ffmpeg quand il est là.
    """
    if chemin := shutil.which("ffmpeg"):
        return chemin
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                  # noqa: BLE001
        return None


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


def _dimensions_livrees(info: VideoInfo, largeur_max: int | None) -> tuple[int, int]:
    """Dimensions de sortie, paires — H.264 refuse les dimensions impaires."""
    if not largeur_max or info.width <= largeur_max:
        return info.width // 2 * 2, info.height // 2 * 2
    hauteur = round(info.height * largeur_max / info.width)
    return largeur_max // 2 * 2, hauteur // 2 * 2


@contextmanager
def video_sink(
    path: str | Path,
    info: VideoInfo,
    largeur_max: int | None = LARGEUR_LIVREE,
    crf: int = CRF,
):
    """Écrit un flux de frames, en H.264 si ffmpeg est disponible.

    Repli sur le codec mp4v d'OpenCV sinon : neuf fois plus lourd, mais un
    fichier lisible vaut mieux qu'une dépendance manquante qui fait échouer
    une analyse de quarante minutes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    largeur, hauteur = _dimensions_livrees(info, largeur_max)
    redimensionne = (largeur, hauteur) != (info.width, info.height)

    ffmpeg = _ffmpeg_binaire()
    if ffmpeg is None:
        log.warning("ffmpeg absent : vidéo encodée en mp4v, environ neuf fois "
                    "plus lourde qu'en H.264")
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (largeur, hauteur)
        )
        try:
            yield lambda frame: writer.write(
                cv2.resize(frame, (largeur, hauteur)) if redimensionne else frame
            )
        finally:
            writer.release()
        return

    commande = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{largeur}x{hauteur}", "-r", f"{info.fps:.4f}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        # yuv420p et faststart : sans eux, la vidéo ne se lit ni dans un
        # navigateur ni sur la plupart des téléphones.
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]
    processus = subprocess.Popen(
        commande, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )

    def ecrire(frame: np.ndarray) -> None:
        if redimensionne:
            frame = cv2.resize(frame, (largeur, hauteur))
        processus.stdin.write(np.ascontiguousarray(frame).tobytes())

    try:
        yield ecrire
    finally:
        try:
            processus.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        code = processus.wait()
        if code != 0:
            erreur = processus.stderr.read().decode("utf-8", "replace")[-400:]
            raise RuntimeError(f"encodage vidéo échoué (code {code}) : {erreur}")
