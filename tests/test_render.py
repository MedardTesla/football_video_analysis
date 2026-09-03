import numpy as np

from football_analysis.pitch.geometry import PITCH
from football_analysis.render import annotators


def test_pitch_render_matches_configured_dimensions():
    pitch = annotators.draw_pitch(scale=0.1, padding=50)
    assert pitch.shape[0] == int(PITCH.width * 0.1) + 100
    assert pitch.shape[1] == int(PITCH.length * 0.1) + 100


def test_drawing_a_player_changes_pixels():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    annotators.draw_ellipse(frame, np.array([100, 100, 140, 200]), (255, 0, 0), "7")
    assert frame.any()


def test_radar_overlay_keeps_frame_shape():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = annotators.overlay_radar(frame, annotators.draw_pitch())
    assert result.shape == (720, 1280, 3)
