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


def test_the_radar_leaves_the_centre_of_the_pitch_visible():
    """Un entraîneur regarde la vidéo pour revoir une action, pas le radar.

    Placé au centre et à 40 % de largeur, il couvrait la zone où se trouve
    presque toujours le ballon.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    avant = frame.copy()
    annotators.overlay_radar(frame, annotators.draw_pitch())

    h, w = frame.shape[:2]
    centre = (slice(int(h * 0.25), int(h * 0.75)), slice(int(w * 0.25), int(w * 0.65)))
    assert np.array_equal(frame[centre], avant[centre]), "le radar couvre le centre"
    # Mais il est bien dessiné quelque part.
    assert frame.any()


def test_the_radar_stays_inside_the_frame():
    for largeur, hauteur in ((1920, 1080), (1280, 720), (854, 480)):
        frame = np.zeros((hauteur, largeur, 3), dtype=np.uint8)
        resultat = annotators.overlay_radar(frame, annotators.draw_pitch())
        assert resultat.shape == (hauteur, largeur, 3)


def test_the_radar_never_overflows_a_tiny_frame():
    """Il se redimensionne proportionnellement : même sur une vignette il
    tient, et surtout il ne déborde pas dans une zone mémoire voisine."""
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    resultat = annotators.overlay_radar(frame, annotators.draw_pitch())
    assert resultat.shape == (80, 120, 3)
    # La marge du bas et de la droite reste vierge.
    assert not resultat[-8:, :].any()
    assert not resultat[:, -8:].any()
