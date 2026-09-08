"""Nommage des joueurs.

Les numéros affichés viennent du traqueur et ne correspondent pas aux
maillots : un entraîneur ne reconnaît pas « joueur 17 ».
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from football_analysis.report import ReportMeta, render, write
from service.jobs import JobState, JobStore
from service.web import pages

STATS = {
    "coverage": 0.9,
    "possession": {"0": 0.55, "1": 0.45},
    "players": [
        {"track_id": 4, "team": 0, "distance_m": 9800.0, "top_speed_ms": 8.2,
         "seconds_seen": 5200.0},
        {"track_id": 9, "team": 1, "distance_m": 8700.0, "top_speed_ms": 9.0,
         "seconds_seen": 5000.0},
        {"track_id": 31, "team": 0, "distance_m": 120.0, "top_speed_ms": 6.0,
         "seconds_seen": 90.0},
    ],
}


# --- rendu -------------------------------------------------------------------

def test_a_named_track_shows_the_name():
    page = render(STATS, ReportMeta("Match"), names={"4": "Kossi Adjovi"})
    assert "Kossi Adjovi" in page


def test_an_unnamed_track_still_shows_its_number():
    page = render(STATS, ReportMeta("Match"), names={"4": "Kossi Adjovi"})
    assert ">9<" in page


def test_the_number_survives_next_to_the_name():
    """Le numéro reste visible : c'est lui qui figure dans la vidéo annotée."""
    page = render(STATS, ReportMeta("Match"), names={"4": "Kossi Adjovi"})
    bloc = page[page.index("Kossi Adjovi") - 200 : page.index("Kossi Adjovi") + 200]
    assert "piste" in bloc


def test_names_are_escaped():
    page = render(STATS, ReportMeta("Match"), names={"4": "<script>x</script>"})
    assert "<script>x</script>" not in page


def test_a_report_without_names_is_unchanged():
    assert render(STATS, ReportMeta("Match")) == render(
        STATS, ReportMeta("Match"), names={}
    )


# --- sélection des pistes nommables -----------------------------------------

def test_a_brief_fragment_is_not_offered_for_naming():
    """Nommer une piste de trois secondes ajouterait du bruit."""
    retenus = pages.nommables(STATS)
    assert {j["track_id"] for j in retenus} == {4, 9}


def test_the_longest_tracks_come_first():
    assert [j["track_id"] for j in pages.nommables(STATS)] == [4, 9]


def test_no_players_means_nothing_to_name():
    assert pages.nommables({"players": []}) == []


# --- persistance -------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.db")


def test_names_survive_a_round_trip(store):
    job = store.create("Club", "Match", "/tmp/v.mp4")
    store.set_player_names(job.id, {"4": "Kossi Adjovi", "9": "Yao Mensah"})
    assert store.get(job.id).player_names == {"4": "Kossi Adjovi", "9": "Yao Mensah"}


def test_an_emptied_name_is_removed(store):
    job = store.create("Club", "Match", "/tmp/v.mp4")
    store.set_player_names(job.id, {"4": "Kossi", "9": "   "})
    assert store.get(job.id).player_names == {"4": "Kossi"}


def test_a_job_starts_without_names(store):
    assert store.create("Club", "Match", "/tmp/v.mp4").player_names == {}


def test_an_older_database_gains_the_names_column(tmp_path):
    import sqlite3

    chemin = tmp_path / "ancienne.db"
    db = sqlite3.connect(chemin)
    db.executescript(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, token TEXT NOT NULL,"
        " club TEXT NOT NULL, match_name TEXT NOT NULL, video_path TEXT NOT NULL,"
        " state TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, error TEXT,"
        " stats TEXT, report_path TEXT, video_output_path TEXT,"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
    )
    db.execute(
        "INSERT INTO jobs VALUES ('a','t','Club','M','/tmp/v.mp4','done',"
        "1,NULL,NULL,NULL,NULL,'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')"
    )
    db.commit(); db.close()

    store = JobStore(chemin)
    assert store.get("a").player_names == {}
    store.set_player_names("a", {"4": "Kossi"})
    assert store.get("a").player_names == {"4": "Kossi"}


# --- effectif du club --------------------------------------------------------
#
# Les noms sont ressaisis à chaque match alors que l'effectif change peu.
# Rien ne relie une piste d'un match à celle du suivant : l'effectif ne peut
# donc que suggérer, jamais nommer d'office.

def test_the_roster_offers_names_from_earlier_matches(store):
    club = store.create_club("ASKO")
    premier = store.create("ASKO", "J1", "/tmp/v.mp4", club_id=club.id)
    store.set_player_names(premier.id, {"4": "Kossi Adjovi", "9": "Yao Mensah"})
    assert store.club_roster(club.id) == ["Kossi Adjovi", "Yao Mensah"]


def test_the_roster_never_crosses_clubs(store):
    """L'effectif d'un club livré à un autre serait une fuite de données."""
    askо = store.create_club("ASKO")
    autre = store.create_club("Barracuda")
    match = store.create("ASKO", "J1", "/tmp/v.mp4", club_id=askо.id)
    store.set_player_names(match.id, {"4": "Kossi Adjovi"})
    assert store.club_roster(autre.id) == []


def test_a_match_deposited_alone_has_no_roster(store):
    """Sans espace de club, aucun lien démontrable entre deux dépôts."""
    match = store.create("ASKO", "J1", "/tmp/v.mp4")
    store.set_player_names(match.id, {"4": "Kossi Adjovi"})
    assert store.club_roster("") == []


def test_the_roster_lists_each_name_once(store):
    club = store.create_club("ASKO")
    for numero, noms in enumerate(
        ({"4": "Kossi Adjovi"}, {"7": "kossi adjovi"}, {"2": "Yao Mensah"}), 1
    ):
        match = store.create("ASKO", f"J{numero}", "/tmp/v.mp4", club_id=club.id)
        store.set_player_names(match.id, noms)
    effectif = store.club_roster(club.id)
    assert len(effectif) == 2
    assert {n.casefold() for n in effectif} == {"kossi adjovi", "yao mensah"}


def test_the_most_recent_match_leads_the_roster(store):
    """L'effectif du dernier match est celui qui ressemble au prochain."""
    club = store.create_club("ASKO")
    vieux = store.create("ASKO", "J1", "/tmp/v.mp4", club_id=club.id)
    store.set_player_names(vieux.id, {"4": "Parti En Janvier"})
    recent = store.create("ASKO", "J2", "/tmp/v.mp4", club_id=club.id)
    store.set_player_names(recent.id, {"4": "Arrivé En Février"})
    assert store.club_roster(club.id)[0] == "Arrivé En Février"


def test_the_roster_is_capped(store):
    club = store.create_club("ASKO")
    match = store.create("ASKO", "J1", "/tmp/v.mp4", club_id=club.id)
    store.set_player_names(match.id, {str(i): f"Joueur {i}" for i in range(100)})
    assert len(store.club_roster(club.id, limit=25)) == 25


# --- suggestions dans le formulaire ------------------------------------------

@pytest.fixture
def job_nommable(store):
    club = store.create_club("ASKO")
    job = store.create("ASKO", "J2", "/tmp/v.mp4", club_id=club.id)
    return store.get(job.id)


def test_the_naming_page_suggests_the_roster(job_nommable):
    page = pages.naming_page(job_nommable, pages.nommables(STATS), ["Kossi Adjovi"])
    assert 'list="effectif"' in page
    assert '<option value="Kossi Adjovi">' in page


def test_a_suggestion_is_never_prefilled(job_nommable):
    """Placer un nom d'office affirmerait un lien entre pistes qui n'existe pas."""
    page = pages.naming_page(job_nommable, pages.nommables(STATS), ["Kossi Adjovi"])
    assert page.count("Kossi Adjovi") == 1        # la suggestion, rien de plus
    assert 'placeholder="Nom du joueur"' in page


def test_a_club_without_a_roster_gets_no_empty_list(job_nommable):
    page = pages.naming_page(job_nommable, pages.nommables(STATS))
    assert "datalist" not in page
    assert 'list="effectif"' not in page


def test_roster_names_are_escaped(job_nommable):
    page = pages.naming_page(job_nommable, pages.nommables(STATS), ['"><script>x'])
    assert "<script>" not in page
