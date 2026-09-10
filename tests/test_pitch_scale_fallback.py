"""Repli à deux échelles du détecteur de points clés du terrain.

Le modèle pose est entraîné à une taille apparente de terrain donnée. Un plan
d'ensemble l'en éloigne assez pour qu'il ne rende plus aucune instance, sans
que rien ne le signale : la frame est simplement marquée non mesurée et la
couverture chute. Mesuré sur un extrait réel, les sept premières secondes
étaient entièrement perdues pour cette seule raison.
"""
from __future__ import annotations

import numpy as np
import pytest

from football_analysis.config import PitchConfig
from football_analysis.pitch.keypoints import PitchKeypointDetector


class _Tenseur:
    """Imite le tenseur torch rendu par Ultralytics (`.cpu().numpy()`)."""

    def __init__(self, valeurs: np.ndarray) -> None:
        self._valeurs = valeurs

    def __getitem__(self, index):
        return _Tenseur(self._valeurs[index])

    def __len__(self) -> int:
        return len(self._valeurs)

    def cpu(self) -> "_Tenseur":
        return self

    def numpy(self) -> np.ndarray:
        return self._valeurs


class _Keypoints:
    def __init__(self, points: np.ndarray, confiance: np.ndarray) -> None:
        self.xy = _Tenseur(points)
        self.conf = _Tenseur(confiance)


class _Resultat:
    def __init__(self, keypoints) -> None:
        self.keypoints = keypoints


class _FauxModele:
    """Rend un nombre de points confiants qui dépend de l'échelle demandée."""

    def __init__(self, points_par_echelle: dict[int, int]) -> None:
        self.points_par_echelle = points_par_echelle
        self.echelles_appelees: list[int] = []

    def predict(self, frame, conf, imgsz, verbose):  # noqa: ARG002
        self.echelles_appelees.append(imgsz)
        n = self.points_par_echelle[imgsz]
        if n == 0:
            return [_Resultat(None)]
        points = np.zeros((1, 32, 2), dtype=np.float32)
        confiance = np.zeros((1, 32), dtype=np.float32)
        confiance[0, :n] = 0.99
        points[0, :n] = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
        return [_Resultat(_Keypoints(points, confiance))]


def _detecteur(config: PitchConfig, modele: _FauxModele) -> PitchKeypointDetector:
    """Construit le détecteur sans charger de poids depuis le disque."""
    detecteur = object.__new__(PitchKeypointDetector)
    detecteur.config = config
    detecteur.model = modele
    return detecteur


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def test_la_seconde_echelle_nest_pas_tentee_si_la_premiere_suffit(frame):
    config = PitchConfig()
    modele = _FauxModele({config.imgsz: 13, config.imgsz_retry: 5})
    _, confiance = _detecteur(config, modele).detect_one(frame)

    assert modele.echelles_appelees == [config.imgsz]
    assert int((confiance >= config.confidence).sum()) == 13


def test_le_plan_large_est_rattrape_par_la_seconde_echelle(frame):
    """0 point à 640, 15 à 1280 — le cas mesuré sur les frames de plan large."""
    config = PitchConfig()
    modele = _FauxModele({config.imgsz: 0, config.imgsz_retry: 15})
    _, confiance = _detecteur(config, modele).detect_one(frame)

    assert modele.echelles_appelees == [config.imgsz, config.imgsz_retry]
    assert int((confiance >= config.confidence).sum()) == 15


def test_le_meilleur_des_deux_est_conserve(frame):
    """Un second essai plus mauvais ne doit pas écraser le premier.

    Sans cette comparaison, une frame passant de 5 points à 1 perdrait les
    quatre points qui, au tour suivant, auraient pu suffire.
    """
    config = PitchConfig()
    modele = _FauxModele({config.imgsz: 5, config.imgsz_retry: 1})
    _, confiance = _detecteur(config, modele).detect_one(frame)

    assert modele.echelles_appelees == [config.imgsz, config.imgsz_retry]
    assert int((confiance >= config.confidence).sum()) == 5


def test_le_repli_se_desactive(frame):
    config = PitchConfig(imgsz_retry=None)
    modele = _FauxModele({config.imgsz: 0})
    _, confiance = _detecteur(config, modele).detect_one(frame)

    assert modele.echelles_appelees == [config.imgsz]
    assert int((confiance >= config.confidence).sum()) == 0


def test_une_echelle_de_repli_identique_ne_double_pas_linference(frame):
    """Garde-fou : `imgsz_retry == imgsz` ne doit pas payer deux inférences."""
    config = PitchConfig(imgsz_retry=PitchConfig().imgsz)
    modele = _FauxModele({config.imgsz: 0})
    _detecteur(config, modele).detect_one(frame)

    assert modele.echelles_appelees == [config.imgsz]


def test_aucune_instance_aux_deux_echelles_rend_des_confiances_nulles(frame):
    """Le contrat de sortie tient même quand le modèle ne voit rien."""
    config = PitchConfig()
    modele = _FauxModele({config.imgsz: 0, config.imgsz_retry: 0})
    points, confiance = _detecteur(config, modele).detect_one(frame)

    assert points.shape == (32, 2)
    assert confiance.shape == (32,)
    assert not confiance.any()
