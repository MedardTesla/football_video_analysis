"""Service de dépôt et de livraison.

Le pipeline n'a aucune valeur tant qu'un club ne peut pas s'en servir : ces
tests couvrent le chemin réel, du dépôt de la vidéo au rapport.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from service.jobs import JobState, JobStore
from service.storage import Storage, UploadRefuse


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.db")


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "videos")


def test_a_new_job_starts_queued(store):
    job = store.create("US Valmont", "Valmont – Beaupré", "/tmp/v.mp4")
    assert job.state is JobState.QUEUED
    assert store.get(job.id).match_name == "Valmont – Beaupré"


def test_the_link_carries_an_unguessable_token(store):
    a = store.create("Club", "Match A", "/tmp/a.mp4")
    b = store.create("Club", "Match B", "/tmp/b.mp4")
    assert a.token != b.token
    assert len(a.token) >= 20
    assert a.token in a.public_url


def test_a_wrong_token_reveals_nothing(store):
    job = store.create("Club", "Match", "/tmp/v.mp4")
    assert store.authenticate(job.id, "mauvais-jeton") is None
    assert store.authenticate("inexistant", job.token) is None
    assert store.authenticate(job.id, job.token) is not None


def test_a_job_is_claimed_only_once(store):
    """Deux workers en parallèle ne doivent pas traiter le même match."""
    store.create("Club", "Match", "/tmp/v.mp4")
    assert store.claim_next() is not None
    assert store.claim_next() is None


def test_claiming_marks_the_job_processing(store):
    job = store.create("Club", "Match", "/tmp/v.mp4")
    claimed = store.claim_next()
    assert claimed.id == job.id
    assert store.get(job.id).state is JobState.PROCESSING


def test_jobs_are_claimed_oldest_first(store):
    premier = store.create("Club", "Premier", "/tmp/1.mp4")
    store.create("Club", "Second", "/tmp/2.mp4")
    assert store.claim_next().id == premier.id


def test_pending_count_does_not_claim(store):
    """Compter la file ne doit pas réserver de job : le tester avec
    claim_next laisserait le suivant bloqué en « en cours »."""
    store.create("Club", "Match", "/tmp/v.mp4")
    assert store.pending_count() == 1
    assert store.pending_count() == 1
    assert store.claim_next() is not None


def test_stats_survive_a_round_trip(store):
    job = store.create("Club", "Match", "/tmp/v.mp4")
    store.update(job.id, state=JobState.DONE, stats={"coverage": 0.47, "players": []})
    relu = store.get(job.id)
    assert relu.state is JobState.DONE
    assert relu.stats["coverage"] == 0.47


def test_unsupported_format_is_refused(storage):
    with pytest.raises(UploadRefuse, match="Format"):
        storage.save_upload("abc", "match.txt", io.BytesIO(b"pas une video"))


def test_empty_file_is_refused(storage):
    with pytest.raises(UploadRefuse, match="vide"):
        storage.save_upload("abc", "match.mp4", io.BytesIO(b""))


def test_oversized_upload_is_refused_and_cleaned(storage, monkeypatch):
    """Un fichier trop gros doit être refusé sans laisser de résidu."""
    import service.storage as module

    monkeypatch.setattr(module, "TAILLE_MAX", 1024)
    with pytest.raises(UploadRefuse, match="volumineuse"):
        storage.save_upload("abc", "match.mp4", io.BytesIO(b"x" * 5000))
    assert not list(storage.job_dir("abc").glob("source*"))


def test_upload_is_written_to_disk(storage):
    chemin = storage.save_upload("abc", "Match Final.MP4", io.BytesIO(b"donnees" * 100))
    assert Path(chemin).exists()
    assert Path(chemin).suffix == ".mp4"


# --- Chemin HTTP complet -----------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    """API montée sur un stockage jetable."""
    monkeypatch.setenv("FA_DATA_ROOT", str(tmp_path))
    import importlib
    from fastapi.testclient import TestClient
    import service.api as api

    importlib.reload(api)
    return TestClient(api.app), api


def _deposer(client, nom="Valmont – Beaupré"):
    return client.post(
        "/matches",
        data={"club": "US Valmont", "match_name": nom},
        files={"video": ("match.mp4", b"contenu video" * 500, "video/mp4")},
    )


def test_the_upload_form_is_served(client):
    tc, _ = client
    page = tc.get("/")
    assert page.status_code == 200
    assert "Déposer une vidéo" in page.text


def test_uploading_returns_a_private_link(client):
    tc, api = client
    reponse = _deposer(tc)
    assert reponse.status_code == 201
    job = api.store.list_for_club("US Valmont")[0]
    assert job.public_url in reponse.text
    assert job.state is JobState.QUEUED


def test_the_status_page_needs_the_right_token(client):
    tc, api = client
    _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    assert tc.get(job.public_url).status_code == 200
    assert tc.get(f"/m/{job.id}/faux-jeton").status_code == 404


def test_an_unknown_match_looks_like_a_wrong_token(client):
    """Distinguer les deux confirmerait qu'un match existe à cette adresse."""
    tc, _ = client
    inconnu = tc.get("/m/000000/jeton")
    assert inconnu.status_code == 404


def test_the_report_is_refused_before_completion(client):
    tc, api = client
    _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    assert tc.get(f"{job.public_url}/rapport").status_code == 409


def test_the_report_is_served_once_done(client, tmp_path):
    tc, api = client
    _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    rapport = tmp_path / "rapport.html"
    rapport.write_text("<h1>Rapport du match</h1>", encoding="utf-8")
    api.store.update(job.id, state=JobState.DONE, report_path=str(rapport))

    reponse = tc.get(f"{job.public_url}/rapport")
    assert reponse.status_code == 200
    assert "Rapport du match" in reponse.text


def test_a_bad_format_is_explained_not_crashed(client):
    tc, _ = client
    reponse = tc.post(
        "/matches",
        data={"club": "US Valmont", "match_name": "Match"},
        files={"video": ("notes.txt", b"pas une video", "text/plain")},
    )
    assert reponse.status_code == 400
    assert "non pris en charge" in reponse.text


def test_the_status_endpoint_reports_progress(client):
    tc, api = client
    _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    etat = tc.get(f"{job.public_url}/etat").json()
    assert etat["state"] == "queued"
    assert etat["terminal"] is False

    api.store.update(job.id, state=JobState.DONE)
    assert tc.get(f"{job.public_url}/etat").json()["terminal"] is True


def test_a_failure_is_shown_in_plain_language(client):
    tc, api = client
    _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    api.store.update(job.id, state=JobState.FAILED, error="La vidéo n'a pas pu être lue.")
    page = tc.get(job.public_url)
    assert "Analyse impossible" in page.text
    assert "La vidéo n&#x27;a pas pu être lue." in page.text or "pas pu être lue" in page.text


def test_health_reports_the_queue(client):
    tc, _ = client
    _deposer(tc)
    sante = tc.get("/sante").json()
    assert sante["en_attente"] == 1
    assert sante["jobs"]["queued"] == 1
