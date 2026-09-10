import numpy as np
import pytest

from football_analysis.analytics.ball import BallTrajectory
from football_analysis.analytics.stats import (
    CM_PER_M, MAX_PLAUSIBLE_SPEED_MS, MatchStats, nearest_player_team,
)
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
    # La vitesse de pointe demande une fenêtre complète : voir les tests
    # dédiés plus bas.


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


def test_an_untracked_detection_is_refused_as_an_identity():
    """Le bug constaté sur la première analyse réelle.

    BoT-SORT rend -1 pour une détection non confirmée. Les accepter
    fusionnait tous les joueurs non suivis en une identité unique, qui
    cumulait leurs distances : sur un extrait de 24 secondes, elle affichait
    123,5 secondes de présence et la plus grande distance du match.
    """
    stats = MatchStats(fps=25.0)
    with pytest.raises(ValueError, match="non suivies"):
        stats.update_player(-1, np.array([0.0, 0.0]), team=0)
    assert stats.players == {}


def test_control_accumulates_across_frames():
    stats = MatchStats(fps=25.0)
    stats.update_control({0: 0.6, 1: 0.4})
    stats.update_control({0: 0.4, 1: 0.6})
    assert stats.control_share() == {0: pytest.approx(0.5), 1: pytest.approx(0.5)}
    # Arrondi à la décimale : 2 images à 25 fps font 0,08 s, soit 0,1 s.
    assert stats.to_dict()["control_seconds"] == pytest.approx(0.1)


def test_control_ignores_frames_without_both_teams():
    stats = MatchStats(fps=25.0)
    stats.update_control({})
    assert stats.control_frames == 0
    assert stats.control_share() == {}


def test_merging_identities_sums_the_distances():
    """Un joueur perdu puis retrouvé ne doit pas figurer deux fois avec la
    moitié de sa distance chacune."""
    stats = MatchStats(fps=12.0)
    stats.update_player(1, np.array([0.0, 0.0]), team=0)
    stats.update_player(1, np.array([40.0, 0.0]), team=0)      # 0,4 m
    stats.update_player(7, np.array([200.0, 0.0]), team=0)
    stats.update_player(7, np.array([260.0, 0.0]), team=0)     # 0,6 m

    stats.merge_identities({7: 1})
    assert set(stats.players) == {1}
    assert stats.players[1].distance_m == pytest.approx(1.0)


def test_merging_keeps_the_peak_speed_not_the_average():
    stats = MatchStats(fps=12.0)
    _course(stats, 1, vitesse_ms=2.0, fps=12.0)
    _course(stats, 7, vitesse_ms=7.0, fps=12.0)
    rapide = stats.players[7].top_speed_ms
    assert rapide > stats.players[1].top_speed_ms

    stats.merge_identities({7: 1})
    assert stats.players[1].top_speed_ms == pytest.approx(rapide)


def test_merging_does_not_invent_the_unobserved_gap():
    """Le trajet entre deux segments n'a pas été mesuré : l'ajouter
    fabriquerait la donnée que le recollement doit rendre crédible."""
    stats = MatchStats(fps=12.0)
    stats.update_player(1, np.array([0.0, 0.0]), team=0)
    stats.update_player(1, np.array([40.0, 0.0]), team=0)
    stats.update_player(7, np.array([5000.0, 0.0]), team=0)    # 50 m plus loin
    stats.update_player(7, np.array([5040.0, 0.0]), team=0)

    stats.merge_identities({7: 1})
    assert stats.players[1].distance_m == pytest.approx(0.8)


def test_merging_recovers_a_team_from_the_other_segment():
    stats = MatchStats(fps=12.0)
    stats.update_player(1, np.array([0.0, 0.0]), team=None)
    stats.update_player(7, np.array([0.0, 0.0]), team=1)
    stats.merge_identities({7: 1})
    assert stats.players[1].team == 1


def test_merging_nothing_leaves_the_players_untouched():
    stats = MatchStats(fps=12.0)
    stats.update_player(1, np.array([0.0, 0.0]), team=0)
    stats.update_player(2, np.array([0.0, 0.0]), team=1)
    stats.merge_identities({})
    assert set(stats.players) == {1, 2}


def _course(stats, track_id, vitesse_ms, fps, secondes=1.0, depart=0.0):
    """Fait courir un joueur en ligne droite, à vitesse constante."""
    pas_cm = vitesse_ms * CM_PER_M / fps
    for i in range(int(round(fps * secondes)) + 1):
        stats.update_player(track_id, np.array([depart + i * pas_cm, 0.0]), team=0)


def test_la_pointe_mesure_une_course_soutenue():
    """Une course régulière rend bien sa vitesse."""
    stats = MatchStats(fps=12.5)
    _course(stats, 1, vitesse_ms=8.0, fps=12.5)
    assert stats.players[1].top_speed_ms == pytest.approx(8.0, rel=1e-6)


def test_un_tremblement_isole_ne_fabrique_pas_une_pointe():
    """Le défaut corrigé : une seule image aberrante donnait un sprint.

    Un joueur presque immobile dont la position saute de 90 cm sur une image
    — bruit ordinaire d'homographie — affichait 11 m/s, soit 40 km/h. Mesuré
    sur un extrait réel, 22 joueurs sur 28 dépassaient ainsi 40 km/h.
    """
    stats = MatchStats(fps=12.5)
    for i in range(30):
        # Immobile, à 2 cm près, sauf une image décalée de 90 cm.
        x = 90.0 if i == 15 else (i % 2) * 2.0
        stats.update_player(1, np.array([x, 0.0]), team=0)

    pointe = stats.players[1].top_speed_ms
    assert pointe < 2.0, f"{pointe * 3.6:.1f} km/h pour un joueur immobile"


def test_la_pointe_ne_depasse_jamais_le_plausible():
    """Garde-fou : aucune valeur au-dessus du plafond ne sort du calcul."""
    stats = MatchStats(fps=12.5)
    _course(stats, 1, vitesse_ms=9.5, fps=12.5, secondes=3.0)
    assert stats.players[1].top_speed_ms <= MAX_PLAUSIBLE_SPEED_MS


def test_un_intervalle_non_mesure_ne_fabrique_pas_une_pointe():
    """Les deux bouts de la fenêtre ne doivent jamais encadrer un trou.

    Sinon la reprise compte le trajet non observé comme une course, ce que
    `mark_unmeasured` évite déjà pour la distance.
    """
    stats = MatchStats(fps=12.5)
    _course(stats, 1, vitesse_ms=3.0, fps=12.5)
    stats.mark_unmeasured()
    # Reprise 40 m plus loin : le joueur a bougé pendant le trou.
    _course(stats, 1, vitesse_ms=3.0, fps=12.5, depart=4000.0)
    assert stats.players[1].top_speed_ms == pytest.approx(3.0, rel=1e-6)
