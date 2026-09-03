"""Orchestration du pipeline complet, en flux.

Déroulé par frame : détecter -> suivre -> projeter sur le terrain ->
classer les équipes -> accumuler les statistiques -> rendre.

Le pipeline est conçu pour dégrader proprement : une frame sans keypoints
terrain exploitables réutilise la dernière homographie valide, et une frame
sans homographie du tout est quand même annotée (sans radar ni statistiques
spatiales). C'est indispensable pour des vidéos de club filmées en caméra
basse, où le terrain est souvent partiellement hors champ.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv

from .analytics.ball import BallTrajectory
from .analytics.stats import MatchStats, nearest_player_team
from .config import GOALKEEPER_ID, PLAYER_ID, REFEREE_ID, Config
from .detection.detector import Detector
from .pitch.keypoints import PitchKeypointDetector, transformer_from_keypoints
from .pitch.view import ViewTransformer
from .render import annotators
from .teams.classifier import TeamClassifier, assign_goalkeeper
from .tracking.tracker import PersonTracker, ball_position
from .video import io as video_io


@dataclass
class PipelineResult:
    video_path: Path
    stats_path: Path
    stats: dict


def _crops(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    """Vignettes des joueurs, pour SigLIP. Les boîtes vides sont écartées."""
    crops = []
    for x1, y1, x2, y2 in detections.xyxy.astype(int):
        crop = frame[max(y1, 0) : y2, max(x1, 0) : x2]
        if crop.size:
            crops.append(crop)
    return crops


def _ground_points(detections: sv.Detections) -> np.ndarray:
    """Point de contact au sol : milieu du bord inférieur de la boîte."""
    if len(detections) == 0:
        return np.empty((0, 2))
    xyxy = detections.xyxy
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)


def fit_team_classifier(
    video_path: Path, detector: Detector, config: Config
) -> TeamClassifier:
    """Ajuste le classifieur d'équipes sur un échantillon du début de match.

    Un seul ajustement pour toute la vidéo : refaire tourner UMAP par frame
    produirait des clusters qui permutent d'une frame à l'autre, et les
    équipes changeraient de couleur en cours de route.
    """
    frames = video_io.frames(video_path, stride=config.teams.fit_stride)
    sample: list[np.ndarray] = []
    for index, (frame, detections) in enumerate(detector.detect(frames)):
        sample.extend(_crops(frame, detections[detections.class_id == PLAYER_ID]))
        if index + 1 >= config.teams.fit_max_frames:
            break
    if not sample:
        raise RuntimeError("aucun joueur détecté : vérifier les poids et la vidéo")

    classifier = TeamClassifier(
        model_name=config.teams.siglip_model,
        batch_size=config.teams.embedding_batch_size,
        n_components=config.teams.umap_components,
        n_teams=config.teams.n_teams,
    )
    return classifier.fit(sample)


def run(
    video_path: str | Path,
    output_path: str | Path,
    config: Config | None = None,
    with_radar: bool = True,
) -> PipelineResult:
    config = config or Config()
    video_path, output_path = Path(video_path), Path(output_path)

    info = video_io.VideoInfo.from_path(video_path)
    detector = Detector(config.detection)
    tracker = PersonTracker(config.tracking)
    classifier = fit_team_classifier(video_path, detector, config)

    keypoint_detector = PitchKeypointDetector(config.pitch)
    ball_trajectory = BallTrajectory(
        max_displacement_cm=config.ball.max_displacement_cm,
        window=config.ball.smoothing_window,
    )
    stats = MatchStats(fps=info.fps)
    last_transformer: ViewTransformer | None = None

    stats_path = output_path.with_suffix(".json")

    with video_io.video_sink(output_path, info) as write:
        for frame, detections in detector.detect(video_io.frames(video_path)):
            frame = frame.copy()
            people, ball = tracker.update(detections, frame=frame)

            points, confidence = keypoint_detector.detect_one(frame)
            transformer = transformer_from_keypoints(points, confidence, config.pitch)
            if transformer is not None:
                last_transformer = transformer
            transformer = transformer or last_transformer

            players = people[people.class_id == PLAYER_ID]
            keepers = people[people.class_id == GOALKEEPER_ID]
            referees = people[people.class_id == REFEREE_ID]

            teams = (
                classifier.predict(_crops(frame, players))
                if len(players)
                else np.empty(0, dtype=int)
            )

            players_xy = _ground_points(players)
            pitch_xy = (
                transformer.frame_to_pitch(players_xy)
                if transformer is not None and len(players_xy)
                else np.empty((0, 2))
            )

            # Statistiques : seulement si la projection terrain est disponible.
            if len(pitch_xy) == len(teams) and len(pitch_xy):
                for track_id, xy, team in zip(players.tracker_id, pitch_xy, teams):
                    stats.update_player(int(track_id), xy, int(team))

                ball_xy = ball_position(ball)
                ball_pitch = (
                    ball_trajectory.update(
                        transformer.frame_to_pitch(ball_xy[None, :])[0]
                    )
                    if ball_xy is not None
                    else ball_trajectory.update(None)
                )
                stats.update_possession(
                    nearest_player_team(ball_pitch, pitch_xy, teams)
                )

            # Rendu.
            for bbox, track_id, team in zip(players.xyxy, players.tracker_id, teams):
                annotators.draw_ellipse(
                    frame, bbox, annotators.TEAM_COLORS[int(team) % 2], str(track_id)
                )
            for bbox, track_id in zip(keepers.xyxy, keepers.tracker_id):
                team = 0
                if len(pitch_xy) and transformer is not None:
                    keeper_xy = transformer.frame_to_pitch(
                        _ground_points(keepers[:1])
                    )[0]
                    team = assign_goalkeeper(keeper_xy, pitch_xy, teams)
                annotators.draw_ellipse(
                    frame, bbox, annotators.TEAM_COLORS[team], str(track_id)
                )
            for bbox, track_id in zip(referees.xyxy, referees.tracker_id):
                annotators.draw_ellipse(
                    frame, bbox, annotators.REFEREE_COLOR, str(track_id)
                )
            for bbox in ball.xyxy:
                annotators.draw_ball_marker(frame, bbox)

            if with_radar and transformer is not None and len(pitch_xy):
                radar = annotators.draw_pitch()
                for team_id in (0, 1):
                    annotators.draw_on_pitch(
                        radar,
                        pitch_xy[teams == team_id],
                        annotators.TEAM_COLORS[team_id],
                    )
                annotators.overlay_radar(frame, radar)

            write(frame)

    payload = stats.to_dict()
    stats_path.write_text(json.dumps(payload, indent=2))
    return PipelineResult(video_path=output_path, stats_path=stats_path, stats=payload)
