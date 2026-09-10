"""Vérifie l'orchestration sans modèle entraîné.

Les détecteurs réels exigent des poids absents du dépôt. On les remplace par
des doublures déterministes : ce test valide le câblage du pipeline (flux
vidéo, suivi, projection, statistiques, écriture), pas la qualité des modèles.
"""
from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from football_analysis import pipeline
from football_analysis.analytics.stats import MAX_PLAUSIBLE_SPEED_MS
from football_analysis.config import BALL_ID, PLAYER_ID, Config
from football_analysis.pitch.geometry import PITCH
from football_analysis.video import io as video_io

WIDTH, HEIGHT, N_FRAMES = 640, 360, 12


@pytest.fixture
def video(tmp_path):
    """Vidéo synthétique : pelouse verte, deux joueurs qui se croisent.

    Le fond doit être une vraie pelouse : le pipeline filtre les détections
    hors gazon, donc un fond neutre ferait rejeter tous les joueurs — ce que
    ce test a effectivement attrapé.
    """
    path = tmp_path / "match.mp4"
    info = video_io.VideoInfo(width=WIDTH, height=HEIGHT, fps=25.0, total_frames=N_FRAMES)
    with video_io.video_sink(path, info) as write:
        for i in range(N_FRAMES):
            frame = np.full((HEIGHT, WIDTH, 3), (60, 160, 70), dtype=np.uint8)
            frame[100 + i : 140 + i, 100 + i : 120 + i] = (255, 0, 0)
            write(frame)
    return path


class FakeDetector:
    """Deux joueurs avançant de 4 px par frame, et un ballon entre les deux."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def detect(self, frames):
        for i, frame in enumerate(frames):
            xyxy = np.array(
                [
                    [100 + 4 * i, 200, 130 + 4 * i, 280],
                    [300 - 4 * i, 200, 330 - 4 * i, 280],
                    [210, 250, 220, 260],
                ],
                dtype=np.float32,
            )
            yield frame, sv.Detections(
                xyxy=xyxy,
                confidence=np.array([0.9, 0.9, 0.8], dtype=np.float32),
                class_id=np.array([PLAYER_ID, PLAYER_ID, BALL_ID]),
            )


class FakeKeypointDetector:
    """Homographie triviale : l'image entière correspond au terrain entier."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def detect_one(self, _frame):
        points = np.zeros((32, 2), dtype=np.float32)
        confidence = np.zeros(32, dtype=np.float32)
        corners = {0: (0, 0), 5: (0, HEIGHT), 24: (WIDTH, 0), 29: (WIDTH, HEIGHT),
                   13: (WIDTH / 2, 0), 16: (WIDTH / 2, HEIGHT)}
        for index, xy in corners.items():
            points[index] = xy
            confidence[index] = 0.99
        return points, confidence


class FakeTeamClassifier:
    """Alterne les équipes, pour que les deux clusters soient peuplés."""

    def predict(self, crops):
        return np.array([i % 2 for i in range(len(crops))])


@pytest.fixture(autouse=True)
def stub_models(monkeypatch):
    monkeypatch.setattr(pipeline, "Detector", FakeDetector)
    monkeypatch.setattr(pipeline, "PitchKeypointDetector", FakeKeypointDetector)
    monkeypatch.setattr(
        pipeline, "fit_team_classifier", lambda *a, **k: FakeTeamClassifier()
    )


def test_pipeline_writes_video_and_stats(video, tmp_path):
    config = Config()
    result = pipeline.run(video, tmp_path / "out.mp4", config)

    assert result.video_path.exists()
    assert result.stats_path.exists()

    # La sortie est échantillonnée : une image sur `stride`, écrite à la
    # cadence réduite pour se lire à la bonne vitesse.
    stride = pipeline.sampling_stride(25.0, config.processing.sample_fps)
    assert video_io.VideoInfo.from_path(result.video_path).total_frames == N_FRAMES // stride


def test_every_frame_is_processed_when_sampling_is_off(video, tmp_path):
    from dataclasses import replace

    config = Config()
    config.processing = replace(config.processing, sample_fps=None)
    result = pipeline.run(video, tmp_path / "out.mp4", config)
    assert video_io.VideoInfo.from_path(result.video_path).total_frames == N_FRAMES


def test_progress_is_reported_and_ends_at_one(video, tmp_path):
    from dataclasses import replace

    config = Config()
    config.processing = replace(config.processing, progress_every=1)
    vus = []
    pipeline.run(video, tmp_path / "out.mp4", config, on_progress=vus.append)

    assert vus, "aucune progression remontée"
    assert vus[-1] == 1.0
    assert all(0.0 <= v <= 1.0 for v in vus)
    assert vus == sorted(vus), "la progression doit être monotone"


def test_stats_record_the_sampling_used(video, tmp_path):
    result = pipeline.run(video, tmp_path / "out.mp4", Config())
    assert result.stats["source_fps"] == 25.0
    assert result.stats["sampled_fps"] <= result.stats["source_fps"]


def test_pipeline_tracks_both_players(video, tmp_path):
    result = pipeline.run(video, tmp_path / "out.mp4", Config())
    assert len(result.stats["players"]) == 2


def test_distances_are_physically_plausible(video, tmp_path):
    result = pipeline.run(video, tmp_path / "out.mp4", Config())
    seconds = N_FRAMES / 25.0
    for player in result.stats["players"]:
        # Personne ne dépasse la vitesse d'un sprinter de haut niveau. La borne
        # vient de la constante du module, jamais d'un nombre recopié : la
        # baisser sans mettre ce test à jour laisserait passer ce qu'elle
        # interdit désormais.
        assert player["distance_m"] <= MAX_PLAUSIBLE_SPEED_MS * seconds
        pointe = player["top_speed_ms"]
        # `None` est légitime : la piste n'a jamais été vue une demi-seconde
        # d'affilée. C'est une absence de mesure, pas une pointe nulle.
        assert pointe is None or pointe <= MAX_PLAUSIBLE_SPEED_MS


def test_players_stay_inside_pitch_bounds(video, tmp_path):
    """L'homographie doit projeter dans les limites du terrain."""
    result = pipeline.run(video, tmp_path / "out.mp4", Config())
    # Une projection partie à l'infini produirait des distances absurdes ;
    # la borne vient du terrain lui-même, jamais d'une constante recopiée.
    diagonale_m = ((PITCH.length / 100) ** 2 + (PITCH.width / 100) ** 2) ** 0.5
    for player in result.stats["players"]:
        assert player["distance_m"] <= diagonale_m


def test_runs_without_radar(video, tmp_path):
    result = pipeline.run(video, tmp_path / "out.mp4", Config(), with_radar=False)
    assert result.video_path.exists()


def test_pipeline_exports_radar_image(video, tmp_path):
    result = pipeline.run(video, tmp_path / "out.mp4", Config())
    assert result.radar_path is not None
    assert result.radar_path.exists()


def test_cli_produces_report(video, tmp_path, monkeypatch):
    from football_analysis import cli

    output = tmp_path / "out.mp4"
    assert cli.main([str(video), "-o", str(output), "--match-name", "US Test"]) == 0
    report = output.with_suffix(".html")
    assert report.exists()
    assert "US Test" in report.read_text(encoding="utf-8")
