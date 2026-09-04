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


def test_flip_index_is_a_complete_involution():
    """Une table de symétrie fausse casse l'augmentation fliplr en silence."""
    flip = PITCH.flip_index
    assert len(flip) == 32
    assert sorted(flip) == list(range(32))
    for i, mirrored in enumerate(flip):
        assert flip[mirrored] == i


def test_flip_index_mirrors_coordinates():
    vertices = PITCH.vertices
    for i, mirrored in enumerate(PITCH.flip_index):
        x, y = vertices[i]
        mx, my = vertices[mirrored]
        assert mx == PITCH.length - x
        assert my == y


def test_pitch_matches_the_laws_of_the_game():
    """Dimensions ajustées sur 228 images annotées, pas choisies.

    La convention diffusée par les exemples Roboflow (120 x 70 m, surface de
    20,15 m) donne 0,960 % d'erreur de reprojection contre 0,398 % ici.
    La suivre gonflerait toute distance mesurée de 14 %.
    """
    assert PITCH.length == 10500          # 105 m
    assert PITCH.width == 6800            # 68 m
    assert PITCH.penalty_box_length == 1650   # 16,50 m
    assert PITCH.penalty_box_width == 4032    # 40,32 m
    assert PITCH.goal_box_length == 550       # 5,50 m
    assert PITCH.goal_box_width == 1832       # 18,32 m
    assert PITCH.centre_circle_radius == 915  # 9,15 m
    assert PITCH.penalty_spot_distance == 1100  # 11 m


def test_flip_index_survives_a_change_of_dimensions():
    """La table de symétrie est structurelle : elle ne dépend que de l'ordre.

    Les dimensions du terrain sont réglables par club ; le modèle entraîné,
    lui, dépend de `flip_idx`. Les deux doivent rester indépendants.
    """
    from football_analysis.pitch.geometry import SoccerPitch

    reference = PITCH.flip_index
    for length, width in [(10000, 6400), (11000, 7500), (12000, 7000)]:
        assert SoccerPitch(length=length, width=width).flip_index == reference


def test_flip_index_matches_the_public_dataset():
    """Contrat avec les poids publics : vérifié contre data.yaml du dataset
    football-field-detection-f07vi v15 (CC BY 4.0)."""
    assert PITCH.flip_index == [
        24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15,
        16, 9, 10, 11, 12, 8, 6, 7, 0, 1, 2, 3, 4, 5, 31, 30,
    ]


def test_sampling_stride_targets_the_requested_rate():
    """Le sous-échantillonnage divise le coût GPU par cinq sans perdre
    en précision : encore faut-il viser la bonne cadence."""
    from football_analysis.pipeline import sampling_stride

    assert sampling_stride(25.0, 10.0) == 2      # 12,5 fps effectifs
    assert sampling_stride(50.0, 10.0) == 5      # 10 fps exactement
    assert sampling_stride(30.0, 5.0) == 6
    # Cadence cible au-dessus de la source : on ne peut pas inventer d'images.
    assert sampling_stride(25.0, 50.0) == 1
    assert sampling_stride(25.0, None) == 1
    assert sampling_stride(25.0, 0) == 1


def test_instance_threshold_is_far_below_the_default():
    """La confiance de la boîte terrain n'est pas un indicateur de qualité.

    Mesuré sur vidéo réelle : elle varie de 0,05 à 0,89 sur des images où les
    points clés restaient bons. Le seuil par défaut d'Ultralytics (0,25)
    jetait l'instance entière, points compris, et divisait par deux le taux
    d'images exploitables. Le modèle ne connaît qu'une classe : il n'y a pas
    de faux positif à craindre d'un seuil bas.
    """
    from football_analysis.config import PitchConfig

    cfg = PitchConfig()
    assert cfg.instance_confidence <= 0.05
    # Le filtrage utile se fait sur les points, pas sur la boîte.
    assert cfg.confidence > cfg.instance_confidence
