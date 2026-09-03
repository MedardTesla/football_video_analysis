import numpy as np
import pytest

from football_analysis.analytics.ball import BallTrajectory
from football_analysis.analytics.stats import MatchStats, nearest_player_team
from football_analysis.analytics.voronoi import control_share


def test_ball_rejects_impossible_jump():
    trajectory = BallTrajectory(max_displacement_cm=500.0, window=3)
    trajectory.update(np.array([0.0, 0.0]))
    before = trajectory.update(np.array([100.0, 0.0]))
    after = trajectory.update(np.array([9000.0, 0.0]))
    # Le saut de 90 m est rejeté : la valeur lissée ne bouge pas.
    assert np.allclose(before, after)


def test_ball_accepts_plausible_movement():
    trajectory = BallTrajectory(max_displacement_cm=500.0, window=2)
    trajectory.update(np.array([0.0, 0.0]))
    result = trajectory.update(np.array([400.0, 0.0]))
    assert result[0] == 200.0


def test_missing_detection_does_not_invent_movement():
    trajectory = BallTrajectory(window=3)
    trajectory.update(np.array([100.0, 100.0]))
    assert np.allclose(trajectory.update(None), [100.0, 100.0])


def test_control_share_is_symmetric_for_mirrored_teams():
    """Positions dérivées du terrain, pas codées en dur : ses dimensions
    sont réglables et ont déjà changé une fois."""
    from football_analysis.pitch.geometry import PITCH

    x, y = PITCH.length * 0.25, PITCH.width / 2
    team_a = np.array([[x, y]])
    team_b = np.array([[PITCH.length - x, y]])
    share_a, share_b = control_share(team_a, team_b)
    assert abs(share_a - share_b) < 0.02
    assert abs(share_a + share_b - 1.0) < 1e-6


def test_distance_accumulates_in_metres():
    # A 25 fps, un joueur à 10 m/s parcourt 40 cm par frame.
    stats = MatchStats(fps=25.0)
    stats.update_player(1, np.array([0.0, 0.0]), team=0)
    stats.update_player(1, np.array([40.0, 0.0]), team=0)    # 0,4 m
    stats.update_player(1, np.array([80.0, 0.0]), team=0)    # 0,4 m
    assert stats.players[1].distance_m == pytest.approx(0.8)
    assert stats.players[1].top_speed_ms == pytest.approx(10.0)


def test_identity_switch_is_not_counted_as_distance():
    stats = MatchStats(fps=25.0)
    stats.update_player(1, np.array([0.0, 0.0]), team=0)
    # 50 m en une frame à 25 fps = 1250 m/s : impossible.
    stats.update_player(1, np.array([5000.0, 0.0]), team=0)
    assert stats.players[1].distance_m == 0.0


def test_possession_goes_to_closest_player():
    players = np.array([[100.0, 100.0], [5000.0, 3000.0]])
    teams = np.array([0, 1])
    assert nearest_player_team(np.array([150.0, 100.0]), players, teams) == 0


def test_distant_ball_belongs_to_nobody():
    players = np.array([[100.0, 100.0]])
    teams = np.array([0])
    assert nearest_player_team(np.array([9000.0, 5000.0]), players, teams) is None
