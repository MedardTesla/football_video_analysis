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


def test_the_home_page_is_the_only_indexable_one(client):
    """La page d'accueil doit se trouver ; toutes les autres adressent un
    jeton et deviendraient publiques une fois indexées."""
    tc, api = client
    assert "noindex" not in tc.get("/").text

    job = _match_termine(api)
    club = api.store.create_club("US Valmont")
    for adresse in (job.public_url,
                    f"{job.public_url}/joueurs",
                    club.public_url,
                    f"{club.public_url}/deposer"):
        assert "noindex" in tc.get(adresse).text, adresse


def test_robots_forbids_everything_but_the_home_page(client):
    tc, _ = client
    reponse = tc.get("/robots.txt")
    assert reponse.status_code == 200
    assert "Disallow: /" in reponse.text
    assert "Allow: /$" in reponse.text      # la racine exacte, rien en dessous


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


# --- date du match -----------------------------------------------------------

def test_the_report_shows_the_match_date_not_the_analysis_date(client):
    """Un club dépose souvent un match joué des semaines plus tôt."""
    tc, api = client
    tc.post("/matches",
            data={"club": "US Valmont", "match_name": "Match",
                  "played_on": "2026-08-30"},
            files={"video": ("m.mp4", b"x" * 500, "video/mp4")})
    assert api.store.list_for_club("US Valmont")[0].played_on == "2026-08-30"


def test_an_unreadable_date_is_ignored_not_refused(client):
    """Bloquer un dépôt de 2 Go pour un champ accessoire serait absurde."""
    tc, api = client
    reponse = tc.post("/matches",
        data={"club": "US Valmont", "match_name": "Match", "played_on": "hier"},
        files={"video": ("m.mp4", b"x" * 500, "video/mp4")})
    assert reponse.status_code == 201
    assert api.store.list_for_club("US Valmont")[0].played_on == ""


def test_no_date_means_no_date_shown():
    """Mieux vaut aucune date qu'une date fausse."""
    from football_analysis.report import ReportMeta, render

    page = render({"players": [], "possession": {}}, ReportMeta("Match"))
    assert "/2026" not in page and "/2025" not in page


def test_the_worker_reads_the_stored_date():
    from datetime import date

    from service.jobs import Job
    from service.worker import _date_du_match

    modele = dict(id="a", token="t", club="C", match_name="M", video_path="/v.mp4")
    assert _date_du_match(Job(**modele, played_on="2026-08-30")) == date(2026, 8, 30)
    assert _date_du_match(Job(**modele, played_on="")) is None
    assert _date_du_match(Job(**modele, played_on="pas-une-date")) is None


def test_the_form_offers_the_match_date(client):
    tc, _ = client
    page = tc.get("/").text
    assert 'name="played_on"' in page
    assert "pas celle du dépôt" in page


# --- dépôt par lien ----------------------------------------------------------

@pytest.fixture
def dns_public(monkeypatch):
    import socket

    import service.fetch as fetch

    monkeypatch.setattr(
        fetch.socket, "getaddrinfo",
        lambda h, *a, **k: [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )


def test_a_link_is_accepted_instead_of_a_file(client, dns_public):
    """Un match de plusieurs gigaoctets ne se téléverse pas depuis une
    connexion mobile ; un lien part en une seconde."""
    tc, api = client
    reponse = tc.post("/matches", data={
        "club": "US Valmont", "match_name": "Match",
        "source_url": "https://www.youtube.com/watch?v=abc",
    })
    assert reponse.status_code == 201
    job = api.store.list_for_club("US Valmont")[0]
    assert job.source_url.endswith("v=abc")
    assert job.video_path == ""
    assert job.state is JobState.QUEUED


def test_neither_link_nor_file_is_refused(client):
    tc, _ = client
    reponse = tc.post("/matches", data={"club": "US Valmont", "match_name": "M"})
    assert reponse.status_code == 400
    assert "lien vers la vidéo" in reponse.text


def test_an_internal_link_is_refused_at_deposit(client, monkeypatch):
    """Refuser à la réception, pas après un téléversement : le club doit
    savoir tout de suite."""
    import socket

    import service.fetch as fetch

    monkeypatch.setattr(
        fetch.socket, "getaddrinfo",
        lambda h, *a, **k: [(socket.AF_INET, None, None, "", ("169.254.169.254", 0))],
    )
    tc, api = client
    reponse = tc.post("/matches", data={
        "club": "US Valmont", "match_name": "M",
        "source_url": "https://innocent.fr/v.mp4",
    })
    assert reponse.status_code == 400
    assert "accessible" in reponse.text
    assert api.store.list_for_club("US Valmont") == []


def test_a_file_wins_over_a_link(client, dns_public):
    """Les deux fournis : le fichier est déjà là, inutile de télécharger."""
    tc, api = client
    tc.post("/matches",
            data={"club": "US Valmont", "match_name": "M",
                  "source_url": "https://exemple.fr/v.mp4"},
            files={"video": ("m.mp4", b"x" * 500, "video/mp4")})
    job = api.store.list_for_club("US Valmont")[0]
    assert job.video_path and not job.source_url


def test_the_form_offers_both_ways(client):
    tc, _ = client
    page = tc.get("/").text
    assert 'name="source_url"' in page and 'name="video"' in page
    assert "déjà en ligne" in page


# --- limites d'entrée et de file ---------------------------------------------

def test_an_absurd_club_name_is_truncated(client, dns_public):
    """Le maxlength du formulaire n'engage que les navigateurs : sans coupe
    côté serveur, un nom de 50 000 caractères est stocké puis renvoyé sur
    chaque page du club."""
    from service.settings import LONGUEUR_CLUB

    tc, api = client
    tc.post("/matches", data={
        "club": "A" * 50_000, "match_name": "M" * 50_000,
        "source_url": "https://exemple.fr/v.mp4",
    })
    job = api.store.matches_of_club(
        api.store.list_for_club("A" * LONGUEUR_CLUB)[0].club_id
    )[0]
    assert len(job.club) == LONGUEUR_CLUB
    assert len(job.match_name) <= 120


def test_a_club_cannot_flood_the_queue(client, dns_public):
    """Un club déposant sa saison entière monopoliserait la file, et ses
    propres rapports arriveraient plus tard."""
    from service.settings import FILE_MAX_PAR_CLUB

    tc, api = client
    club = api.store.create_club("US Valmont")
    for i in range(FILE_MAX_PAR_CLUB):
        api.store.create("US Valmont", f"M{i}", "/tmp/v.mp4", club_id=club.id)

    reponse = tc.post("/matches", data={
        "match_name": "De trop", "club_id": club.id, "club_token": club.token,
        "source_url": "https://exemple.fr/v.mp4",
    })
    assert reponse.status_code == 429
    assert "déjà" in reponse.text


def test_a_finished_match_frees_the_slot(client, dns_public):
    from service.settings import FILE_MAX_PAR_CLUB

    tc, api = client
    club = api.store.create_club("US Valmont")
    faits = [api.store.create("US Valmont", f"M{i}", "/tmp/v.mp4", club_id=club.id)
             for i in range(FILE_MAX_PAR_CLUB)]
    api.store.update(faits[0].id, state=JobState.DONE)

    reponse = tc.post("/matches", data={
        "match_name": "Suivant", "club_id": club.id, "club_token": club.token,
        "source_url": "https://exemple.fr/v.mp4",
    })
    assert reponse.status_code == 201


def test_a_saturated_queue_refuses_politely(client, dns_public, monkeypatch):
    """Accepter des matchs qu'on ne traitera pas avant des jours serait pire
    que refuser."""
    import service.api as api_mod

    monkeypatch.setattr(api_mod, "FILE_MAX", 0)
    tc, _ = client
    reponse = tc.post("/matches", data={
        "club": "US Valmont", "match_name": "M",
        "source_url": "https://exemple.fr/v.mp4",
    })
    assert reponse.status_code == 503
    assert "quelques heures" in reponse.text
