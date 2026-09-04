"""Interface en ligne de commande.

Une analyse dure des dizaines de minutes : sans retour visible, rien ne
distingue un traitement en cours d'un processus figé.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_analysis import cli
from football_analysis.video import io as video_io

WIDTH, HEIGHT, N = 320, 180, 10


@pytest.fixture
def video(tmp_path):
    chemin = tmp_path / "match.mp4"
    info = video_io.VideoInfo(width=WIDTH, height=HEIGHT, fps=25.0, total_frames=N)
    with video_io.video_sink(chemin, info) as write:
        for _ in range(N):
            write(np.full((HEIGHT, WIDTH, 3), (60, 160, 70), dtype=np.uint8))
    return chemin


@pytest.fixture(autouse=True)
def pipeline_simule(monkeypatch, tmp_path):
    """Le pipeline réel exige des poids absents du dépôt."""
    from football_analysis.pipeline import PipelineResult

    appels = {}

    def run(video_path, output_path, config, with_radar=True, on_progress=None):
        appels["config"] = config
        appels["progress"] = on_progress
        stats = Path(output_path).with_suffix(".json")
        stats.parent.mkdir(parents=True, exist_ok=True)
        stats.write_text('{"possession": {}, "players": []}')
        Path(output_path).write_bytes(b"video")
        if on_progress:
            for f in (0.0, 0.5, 1.0):
                on_progress(f)
        return PipelineResult(
            video_path=Path(output_path), stats_path=stats,
            stats={"possession": {}, "players": []},
        )

    monkeypatch.setattr(cli, "run", run)
    return appels


def test_a_run_reports_progress(video, tmp_path, capsys):
    assert cli.main([str(video), "-o", str(tmp_path / "out.mp4")]) == 0
    affiche = capsys.readouterr().err
    assert "100.0%" in affiche
    assert "█" in affiche


def test_quiet_suppresses_the_bar(video, tmp_path, capsys, pipeline_simule):
    cli.main([str(video), "-o", str(tmp_path / "out.mp4"), "--quiet"])
    assert capsys.readouterr().err == ""
    assert pipeline_simule["progress"] is None


def test_the_sampling_rate_can_be_overridden(video, tmp_path, pipeline_simule):
    cli.main([str(video), "-o", str(tmp_path / "out.mp4"), "--fps", "5"])
    assert pipeline_simule["config"].processing.sample_fps == 5.0


def test_the_default_sampling_rate_is_the_measured_one(video, tmp_path, pipeline_simule):
    """12 fps : mesuré sur extrait réel, pas choisi."""
    cli.main([str(video), "-o", str(tmp_path / "out.mp4")])
    assert pipeline_simule["config"].processing.sample_fps == 12.0


def test_a_report_is_written_next_to_the_video(video, tmp_path):
    sortie = tmp_path / "out.mp4"
    cli.main([str(video), "-o", str(sortie), "--match-name", "US Test"])
    rapport = sortie.with_suffix(".html")
    assert rapport.exists()
    assert "US Test" in rapport.read_text(encoding="utf-8")


def test_the_report_can_be_skipped(video, tmp_path):
    sortie = tmp_path / "out.mp4"
    cli.main([str(video), "-o", str(sortie), "--no-report"])
    assert not sortie.with_suffix(".html").exists()
