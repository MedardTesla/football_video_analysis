"""Classification d'équipes : rejet des non-joueurs et couleurs.

SigLIP n'est pas chargé ici — seule la logique de décision est testée, en
injectant des embeddings connus.
"""
from __future__ import annotations

import numpy as np
import pytest

from football_analysis.render import annotators
from football_analysis.teams.classifier import UNASSIGNED, assign_goalkeeper


@pytest.fixture
def classifier():
    """TeamClassifier sans SigLIP : les embeddings sont fournis à la main."""
    from sklearn.cluster import KMeans
    from football_analysis.teams.classifier import TeamClassifier

    clf = TeamClassifier.__new__(TeamClassifier)
    clf.batch_size = 8
    clf.n_teams = 2
    clf._team_of_cluster = {}
    clf._fitted = False
    # n_teams + 1 : le cluster supplémentaire absorbe les officiels.
    clf.cluster = KMeans(n_clusters=3, n_init=10, random_state=0)

    class Identity:
        """Remplace UMAP : la réduction n'est pas ce qu'on teste ici."""

        def fit_transform(self, x):
            return np.asarray(x, dtype=float)

        def transform(self, x):
            return np.asarray(x, dtype=float)

    clf.reducer = Identity()
    return clf


def _match_crops(n=40, officials=6, spread=0.3, seed=0):
    """Deux maillots majoritaires plus un petit groupe d'officiels."""
    rng = np.random.default_rng(seed)
    a = rng.normal((0, 0, 0), spread, (n, 3))
    b = rng.normal((10, 0, 0), spread, (n, 3))
    c = rng.normal((5, 20, 0), spread, (officials, 3))
    return np.vstack([a, b, c]), n, officials


def test_the_two_biggest_groups_become_the_teams(classifier, monkeypatch):
    data, n, _ = _match_crops()
    monkeypatch.setattr(classifier, "_embed", lambda crops: np.asarray(crops))
    classifier.fit(list(data))
    labels = classifier.predict(list(data))
    assert set(labels[:n]) == {0} or set(labels[:n]) == {1}
    assert set(labels[n : 2 * n]) != set(labels[:n])
    assert UNASSIGNED not in labels[: 2 * n]


def test_officials_are_not_forced_into_a_team(classifier, monkeypatch):
    """Le défaut constaté sur match réel.

    Les arbitres en turquoise tombaient dans le cluster de l'équipe en rouge,
    faute de troisième option. Un rejet par distance au centroïde n'y changeait
    rien : présents à l'ajustement, ils rentraient dans la dispersion normale.
    """
    data, n, officials = _match_crops()
    monkeypatch.setattr(classifier, "_embed", lambda crops: np.asarray(crops))
    classifier.fit(list(data))
    labels = classifier.predict(list(data))
    assert set(labels[2 * n :]) == {UNASSIGNED}


def test_players_are_never_sacrificed_to_the_extra_cluster(classifier, monkeypatch):
    data, n, officials = _match_crops()
    monkeypatch.setattr(classifier, "_embed", lambda crops: np.asarray(crops))
    classifier.fit(list(data))
    labels = classifier.predict(list(data))
    assert (labels == UNASSIGNED).sum() == officials


def test_predict_before_fit_is_refused(classifier):
    with pytest.raises(RuntimeError):
        classifier.predict([np.zeros((10, 10, 3), np.uint8)])


def test_goalkeeper_joins_the_nearest_team():
    players = np.array([[1000.0, 3500.0], [1200.0, 3000.0],
                        [11000.0, 3500.0], [10800.0, 3000.0]])
    teams = np.array([0, 0, 1, 1])
    assert assign_goalkeeper(np.array([300.0, 3500.0]), players, teams) == 0
    assert assign_goalkeeper(np.array([11700.0, 3500.0]), players, teams) == 1


def test_goalkeeper_is_unassigned_when_a_team_is_absent():
    """Ne pas deviner : sans joueur de l'équipe 1 visible, aucun centroïde."""
    players = np.array([[1000.0, 3500.0]])
    teams = np.array([0])
    assert assign_goalkeeper(np.array([300.0, 3500.0]), players, teams) == UNASSIGNED


def test_unassigned_is_never_painted_as_a_team():
    """-1 % 2 vaut 1 en Python : indexer directement peindrait l'équipe B."""
    assert annotators.team_color(0) == annotators.TEAM_COLORS[0]
    assert annotators.team_color(1) == annotators.TEAM_COLORS[1]
    assert annotators.team_color(UNASSIGNED) == annotators.UNASSIGNED_COLOR
    assert annotators.team_color(UNASSIGNED) not in annotators.TEAM_COLORS


def test_untracked_detections_are_identified_as_such():
    """Le pipeline doit pouvoir écarter ce que le traqueur n'a pas confirmé."""
    import supervision as sv
    from football_analysis.tracking.tracker import UNTRACKED, is_tracked

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 20], [30, 0, 40, 20], [60, 0, 70, 20]], dtype=np.float32),
        confidence=np.full(3, 0.9, dtype=np.float32),
        class_id=np.array([2, 2, 2]),
        tracker_id=np.array([7, UNTRACKED, 12]),
    )
    assert is_tracked(detections).tolist() == [True, False, True]


def test_detections_without_any_tracker_id_are_all_untracked():
    import supervision as sv
    from football_analysis.tracking.tracker import is_tracked

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2]),
    )
    assert is_tracked(detections).tolist() == [False]
    assert is_tracked(sv.Detections.empty()).tolist() == []
