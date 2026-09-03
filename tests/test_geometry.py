import numpy as np
import pytest

from football_analysis.pitch.geometry import PITCH
from football_analysis.pitch.view import SmoothedHomography, ViewTransformer


def test_pitch_has_32_unique_vertices():
    vertices = PITCH.vertices
    assert len(vertices) == 32
    assert len(set(vertices)) == 32


def test_vertices_stay_within_pitch():
    for x, y in PITCH.vertices:
        assert 0 <= x <= PITCH.length
        assert 0 <= y <= PITCH.width


def test_edges_reference_existing_vertices():
    for start, end in PITCH.edges:
        assert 1 <= start <= 32
        assert 1 <= end <= 32


def test_identity_homography_round_trips():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    transformer = ViewTransformer(source=square, target=square)
    result = transformer.frame_to_pitch(np.array([[50, 50]], dtype=np.float32))
    assert np.allclose(result, [[50, 50]], atol=1e-3)


def test_transform_is_reversible():
    source = np.array([[10, 10], [200, 30], [220, 180], [15, 200]], dtype=np.float32)
    target = np.array([[0, 0], [1200, 0], [1200, 700], [0, 700]], dtype=np.float32)
    transformer = ViewTransformer(source=source, target=target)

    points = np.array([[100, 100], [50, 150]], dtype=np.float32)
    round_trip = transformer.pitch_to_frame(transformer.frame_to_pitch(points))
    assert np.allclose(round_trip, points, atol=1e-2)


def test_fewer_than_four_points_is_rejected():
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    with pytest.raises(ValueError):
        ViewTransformer(source=points, target=points)


def test_smoothing_averages_matrices():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    smoother = SmoothedHomography(window=3)
    assert not smoother.ready
    result = smoother.update(ViewTransformer(source=square, target=square))
    assert smoother.ready
    assert np.allclose(result, np.eye(3), atol=1e-6)
