"""Écriture de la vidéo livrée au club.

En 1080p avec le codec mp4v d'OpenCV, un match de 90 minutes pèse 3,5 Go —
intéléchargeable sur une connexion mobile, qui est la norme du marché visé.
En 1280 px et H.264, il pèse 0,37 Go.
"""
from __future__ import annotations

import numpy as np
import pytest

from football_analysis.video import io as video_io


def _frames(n, largeur, hauteur):
    """Images compressibles : du bruit fausserait toute mesure de poids."""
    for i in range(n):
        img = np.full((hauteur, largeur, 3), (60, 160, 70), dtype=np.uint8)
        img[100 + i : 160 + i, 200 + i * 3 : 240 + i * 3] = (240, 240, 60)
        yield img


@pytest.fixture
def info():
    return video_io.VideoInfo(width=1920, height=1080, fps=12.0, total_frames=40)


def test_a_written_video_can_be_read_back(tmp_path, info):
    chemin = tmp_path / "sortie.mp4"
    with video_io.video_sink(chemin, info) as write:
        for frame in _frames(40, 1920, 1080):
            write(frame)

    relu = video_io.VideoInfo.from_path(chemin)
    assert relu.total_frames == 40
    assert abs(relu.fps - 12.0) < 0.5


def test_the_delivered_video_is_downscaled(tmp_path, info):
    chemin = tmp_path / "sortie.mp4"
    with video_io.video_sink(chemin, info) as write:
        for frame in _frames(20, 1920, 1080):
            write(frame)
    assert video_io.VideoInfo.from_path(chemin).width == video_io.LARGEUR_LIVREE


def test_downscaling_can_be_disabled(tmp_path, info):
    chemin = tmp_path / "sortie.mp4"
    with video_io.video_sink(chemin, info, largeur_max=None) as write:
        for frame in _frames(20, 1920, 1080):
            write(frame)
    assert video_io.VideoInfo.from_path(chemin).width == 1920


def test_a_small_video_is_not_upscaled(tmp_path):
    """Agrandir n'ajoute aucune information et alourdit le fichier."""
    petite = video_io.VideoInfo(width=640, height=360, fps=12.0, total_frames=20)
    chemin = tmp_path / "sortie.mp4"
    with video_io.video_sink(chemin, petite) as write:
        for frame in _frames(20, 640, 360):
            write(frame)
    assert video_io.VideoInfo.from_path(chemin).width == 640


def test_odd_dimensions_are_made_even(tmp_path):
    """H.264 refuse les dimensions impaires : le fichier serait vide."""
    impair = video_io.VideoInfo(width=1281, height=721, fps=12.0, total_frames=10)
    largeur, hauteur = video_io._dimensions_livrees(impair, None)
    assert largeur % 2 == 0 and hauteur % 2 == 0


def test_the_aspect_ratio_survives_downscaling():
    info = video_io.VideoInfo(width=1920, height=1080, fps=12.0, total_frames=1)
    largeur, hauteur = video_io._dimensions_livrees(info, 1280)
    assert abs(largeur / hauteur - 1920 / 1080) < 0.01


@pytest.mark.skipif(
    video_io._ffmpeg_binaire() is None, reason="ffmpeg absent de cet environnement"
)
def test_h264_is_far_lighter_than_the_opencv_codec(tmp_path, info):
    """Le chiffre qui décide si un club peut télécharger son match."""
    h264 = tmp_path / "h264.mp4"
    with video_io.video_sink(h264, info) as write:
        for frame in _frames(40, 1920, 1080):
            write(frame)

    import cv2

    brut = tmp_path / "mp4v.mp4"
    writer = cv2.VideoWriter(str(brut), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (1280, 720))
    for frame in _frames(40, 1920, 1080):
        writer.write(cv2.resize(frame, (1280, 720)))
    writer.release()

    assert h264.stat().st_size < brut.stat().st_size / 3


def test_missing_ffmpeg_falls_back_instead_of_failing(tmp_path, info, monkeypatch):
    """Un fichier lourd vaut mieux qu'une analyse de quarante minutes perdue."""
    monkeypatch.setattr(video_io, "_ffmpeg_binaire", lambda: None)
    chemin = tmp_path / "repli.mp4"
    with video_io.video_sink(chemin, info) as write:
        for frame in _frames(20, 1920, 1080):
            write(frame)
    assert chemin.exists()
    assert video_io.VideoInfo.from_path(chemin).width == video_io.LARGEUR_LIVREE
