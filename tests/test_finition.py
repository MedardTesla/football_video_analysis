"""Finitions : erreurs lisibles, indexation, export, suppression.

Chacun de ces défauts se voit dès le premier client réel.
"""
from __future__ import annotations

import pytest

from service.export import players_csv
from service.jobs import JobState

STATS = {
    "players": [
        {"track_id": 4, "team": 0, "distance_m": 9812.4, "top_speed_ms": 8.33,
         "seconds_seen": 5200.0},
        {"track_id": 9, "team": 1, "distance_m": 8730.1, "top_speed_ms": 9.11,
         "seconds_seen": 5000.0},
        {"track_id": 12, "team": None, "distance_m": 400.0, "top_speed_ms": 5.0,
         "seconds_seen": 300.0},
    ]
}


# --- export tableur ----------------------------------------------------------

def test_the_csv_has_one_row_per_player():
    lignes = players_csv(STATS).strip().splitlines()
    assert len(lignes) == 4          # en-tête + trois joueurs


def test_the_csv_uses_the_french_conventions():
    """Excel en configuration française attend le point-virgule et la
    virgule décimale ; sans quoi tout tient dans une seule colonne."""
    csv = players_csv(STATS)
    assert csv.startswith("﻿")   # BOM, sans quoi les accents cassent
    assert ";" in csv.splitlines()[0]
    assert "9,81" in csv              # 9812 m en km, virgule décimale


def test_the_csv_carries_the_names_when_known():
    csv = players_csv(STATS, {"4": "Kossi Adjovi"})
    assert "Kossi Adjovi" in csv


def test_speeds_are_exported_in_kilometres_per_hour():
    """Un entraîneur pense en km/h, pas en m/s."""
    csv = players_csv(STATS)
    assert "30,0" in csv              # 8,33 m/s


def test_an_unassigned_team_is_left_blank():
    ligne = [l for l in players_csv(STATS).splitlines() if l.startswith("12;")][0]
    assert ligne.split(";")[2] == ""


def test_an_empty_match_still_produces_a_header():
    assert players_csv({"players": []}).strip().count("\n") == 0


# --- pages d'erreur et indexation -------------------------------------------

def test_a_wrong_link_shows_a_page_not_json(client):
    """Un club voyait la réponse JSON brute de l'API."""
    tc, _ = client
    reponse = tc.get("/m/inconnu/jeton")
    assert reponse.status_code == 404
    assert "text/html" in reponse.headers["content-type"]
    assert "Page introuvable" in reponse.text
    assert "detail" not in reponse.text


def test_the_error_page_suggests_what_to_do(client):
    tc, _ = client
    assert "se coupent souvent" in tc.get("/m/inconnu/jeton").text


def test_every_page_refuses_indexing(client):
    """Les adresses portent un jeton : indexées, elles deviendraient
    publiques."""
    tc, _ = client
    assert "noindex" in tc.get("/").text


def test_robots_forbids_the_whole_site(client):
    tc, _ = client
    reponse = tc.get("/robots.txt")
    assert reponse.status_code == 200
    assert "Disallow: /" in reponse.text


# --- suppression -------------------------------------------------------------

def _match_termine(api, nom="Match"):
    import json
    job = api.store.create("US Valmont", nom, "/tmp/v.mp4")
    dossier = api.storage.job_dir(job.id)
    stats = {"players": STATS["players"]}
    (dossier / "analyse.json").write_text(json.dumps(stats))
    (dossier / "rapport.html").write_text("<h1>R</h1>", encoding="utf-8")
    api.store.update(job.id, state=JobState.DONE, stats=stats,
                     report_path=str(dossier / "rapport.html"))
    return api.store.get(job.id)


def test_deleting_a_match_removes_its_files(client):
    """Un club se trompe de vidéo : sans ce bouton il faudrait nous écrire."""
    tc, api = client
    job = _match_termine(api)
    dossier = api.storage.job_dir(job.id)
    assert (dossier / "rapport.html").exists()

    reponse = tc.post(f"{job.public_url}/supprimer", follow_redirects=False)
    assert reponse.status_code == 303
    assert api.store.get(job.id) is None
    assert not dossier.exists()


def test_deleting_needs_the_match_token(client):
    tc, api = client
    job = _match_termine(api)
    assert tc.post(f"/m/{job.id}/faux/supprimer").status_code == 404
    assert api.store.get(job.id) is not None


def test_deletion_returns_to_the_club_space(client):
    tc, api = client
    club = api.store.create_club("US Valmont")
    job = _match_termine(api)
    api.store.update(job.id, club_id=club.id)
    job = api.store.get(job.id)

    reponse = tc.post(f"{job.public_url}/supprimer", follow_redirects=False)
    assert reponse.headers["location"] == club.public_url


def test_the_csv_is_served_as_a_download(client):
    tc, api = client
    job = _match_termine(api, "Valmont – Beaupré")
    reponse = tc.get(f"{job.public_url}/releve.csv")
    assert reponse.status_code == 200
    assert "attachment" in reponse.headers["content-disposition"]
    assert "9,81" in reponse.text


def test_the_csv_is_refused_before_the_analysis_ends(client):
    tc, api = client
    job = api.store.create("US Valmont", "Match", "/tmp/v.mp4")
    assert tc.get(f"{job.public_url}/releve.csv").status_code == 409


@pytest.mark.parametrize("nom", [
    "Valmont – Beaupré", "ASKO Kara vs Djoliba", "Été 2026 · Coupe", "!!!",
])
def test_an_accented_match_name_does_not_break_the_download(client, nom):
    """Les en-têtes HTTP sont en latin-1 : un tiret cadratin y lève une
    erreur d'encodage et le téléchargement échoue."""
    tc, api = client
    job = _match_termine(api, nom)
    reponse = tc.get(f"{job.public_url}/releve.csv")
    assert reponse.status_code == 200
    entete = reponse.headers["content-disposition"]
    assert "attachment" in entete
    assert "filename*=UTF-8''" in entete
    entete.encode("latin-1")          # ne doit pas lever


def test_an_unknown_route_also_shows_a_page(client):
    """Une route inexistante lève l'exception de Starlette, que le
    gestionnaire de FastAPI ne couvre pas."""
    tc, _ = client
    reponse = tc.get("/adresse-inventee")
    assert reponse.status_code == 404
    assert "text/html" in reponse.headers["content-type"]
    assert "Page introuvable" in reponse.text


def test_a_conflict_explains_the_analysis_is_running(client):
    tc, api = client
    job = api.store.create("US Valmont", "Match", "/tmp/v.mp4")
    reponse = tc.get(f"{job.public_url}/rapport")
    assert reponse.status_code == 409
    assert "Analyse en cours" in reponse.text
