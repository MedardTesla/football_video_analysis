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
