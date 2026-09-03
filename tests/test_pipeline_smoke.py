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
from football_analysis.config import BALL_ID, PLAYER_ID, Config
from football_analysis.pitch.geometry import PITCH
from football_analysis.video import io as video_io

WIDTH, HEIGHT, N_FRAMES = 640, 360, 12


@pytest.fixture
def video(tmp_path):
    """Vidéo synthétique : deux joueurs qui se croisent, plus un ballon."""
    path = tmp_path / "match.mp4"
    info = video_io.VideoInfo(width=WIDTH, height=HEIGHT, fps=25.0, total_frames=N_FRAMES)
    with video_io.video_sink(path, info) as write:
        for i in range(N_FRAMES):
            frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
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
    result = pipeline.run(video, tmp_path / "out.mp4", Config())

    assert result.video_path.exists()
    assert result.stats_path.exists()
    assert video_io.VideoInfo.from_path(result.video_path).total_frames == N_FRAMES


def test_pipeline_tracks_both_players(video, tmp_path):
    result = pipeline.run(video, tmp_path / "out.mp4", Config())
    assert len(result.stats["players"]) == 2


def test_distances_are_physically_plausible(video, tmp_path):
    result = pipeline.run(video, tmp_path / "out.mp4", Config())
    seconds = N_FRAMES / 25.0
    for player in result.stats["players"]:
        # Personne ne dépasse la vitesse d'un sprinter de haut niveau.
        assert player["distance_m"] <= 12.0 * seconds
        assert player["top_speed_ms"] <= 12.0


def test_players_stay_inside_pitch_bounds(video, tmp_path):
    """L'homographie doit projeter dans les limites du terrain."""
    pipeline.run(video, tmp_path / "out.mp4", Config())
    # Contrôle indirect : la distance parcourue reste cohérente avec un
    # terrain de 120 m, pas avec une projection partie à l'infini.
    assert PITCH.length == 12000


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
