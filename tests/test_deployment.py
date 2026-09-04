"""Cohérence de la configuration de déploiement.

Ces fichiers ne s'exécutent pas dans la suite de tests : une divergence avec
le code ne se découvrirait qu'au premier déploiement, sur la machine du client.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parent.parent
DOCKERFILE = (RACINE / "Dockerfile").read_text(encoding="utf-8")
COMPOSE = yaml.safe_load((RACINE / "docker-compose.yml").read_text(encoding="utf-8"))


def _etapes(dockerfile: str) -> dict[str, list[str]]:
    """Instructions par étape, commentaires exclus.

    Découper naïvement sur « AS worker » coupait au milieu de la ligne FROM et
    comptait les commentaires — ce qui faisait échouer le test sur sa propre
    documentation.
    """
    etapes: dict[str, list[str]] = {}
    courante = None
    for ligne in dockerfile.splitlines():
        nue = ligne.split("#", 1)[0].strip()
        if not nue:
            continue
        if nue.upper().startswith("FROM ") and " AS " in nue.upper():
            courante = nue.rsplit(None, 1)[-1]
            etapes[courante] = [nue]
        elif courante:
            etapes[courante].append(nue)
    return etapes


ETAPES = _etapes(DOCKERFILE)


def test_both_stages_are_declared():
    assert set(ETAPES) == {"api", "worker"}


def test_the_api_image_does_not_install_the_calculation_stack():
    """Sinon l'image gonfle de plusieurs gigaoctets sans usage."""
    api = "\n".join(ETAPES["api"]).lower()
    assert "requirements-api.txt" in api
    assert "-r requirements.txt" not in api
    for lourd in ("torch", "ultralytics", "cuda"):
        assert lourd not in api, lourd


def test_the_worker_image_is_built_on_cuda():
    """Le pipeline est inutilisable sans GPU : environ 3 vignettes par
    seconde sur processeur, soit des heures par match."""
    worker = "\n".join(ETAPES["worker"]).lower()
    assert "cuda" in ETAPES["worker"][0].lower()      # la ligne FROM
    assert "service.run_worker" in worker


def test_opencv_system_libraries_are_installed():
    """OpenCV réclame libGL même en version sans interface ; l'oubli ne se
    voit qu'au premier match traité."""
    worker = "\n".join(ETAPES["worker"]).lower()
    assert "libgl1" in worker
    assert "libglib2.0-0" in worker


def test_both_services_share_the_same_data_root():
    """API et worker ne communiquent que par ce chemin."""
    chemins = {
        nom: service["environment"]["FA_DATA_ROOT"]
        for nom, service in COMPOSE["services"].items()
    }
    assert len(set(chemins.values())) == 1, chemins
    assert "FA_DATA_ROOT=/data" in DOCKERFILE


def test_only_the_worker_reserves_a_gpu():
    assert "deploy" in COMPOSE["services"]["worker"]
    assert "deploy" not in COMPOSE["services"]["api"]


def test_the_weights_are_mounted_not_copied():
    """Ils changent à chaque réentraînement et pèsent plus que le code."""
    montages = COMPOSE["services"]["worker"]["volumes"]
    assert any("./models:/app/models" in m for m in montages)
    assert "models/" in (RACINE / ".dockerignore").read_text(encoding="utf-8")


def test_neither_image_runs_as_root():
    """Le service reçoit des fichiers d'inconnus."""
    for nom, instructions in ETAPES.items():
        assert any(i.startswith("USER ") and "root" not in i for i in instructions), nom


def test_the_data_volume_is_shared_between_services():
    for nom in ("api", "worker"):
        assert any(
            m.startswith("donnees:") for m in COMPOSE["services"][nom]["volumes"]
        ), nom
    assert "donnees" in COMPOSE["volumes"]


@pytest.mark.parametrize("fichier", ["Dockerfile", "docker-compose.yml", "DEPLOIEMENT.md"])
def test_deployment_files_exist(fichier):
    assert (RACINE / fichier).exists()
