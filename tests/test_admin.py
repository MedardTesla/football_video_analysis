"""Page d'exploitation.

Sert d'abord au support : un club qui perd son lien n'a aucun recours, et
sans cette page il faudrait interroger la base à la main pour le lui
renvoyer. Elle montre donc les jetons en clair — d'où sa protection.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from service.jobs import JobState

JETON = "jeton-exploitation-de-test"


@pytest.fixture
def admin(tmp_path, monkeypatch):
    monkeypatch.setenv("FA_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("FA_ADMIN_TOKEN", JETON)
    import service.api as api
    import service.settings as settings

    importlib.reload(settings)
    importlib.reload(api)
    return TestClient(api.app), api


@pytest.fixture
def sans_jeton(tmp_path, monkeypatch):
    monkeypatch.setenv("FA_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("FA_ADMIN_TOKEN", raising=False)
    import service.api as api
    import service.settings as settings

    importlib.reload(settings)
    importlib.reload(api)
    return TestClient(api.app), api


def test_the_page_lists_clubs_and_their_links(admin):
    """Le cas d'usage : un club a perdu son lien, il faut le retrouver."""
    tc, api = admin
    club = api.store.create_club("ASKO Kara")
    api.store.create("ASKO Kara", "ASKO – Djoliba", "/tmp/v.mp4", club_id=club.id)

    page = tc.get(f"/admin/{JETON}")
    assert page.status_code == 200
    assert "ASKO Kara" in page.text
    assert club.public_url in page.text
    assert "ASKO – Djoliba" in page.text


def test_a_wrong_token_looks_like_a_missing_page(admin):
    tc, _ = admin
    reponse = tc.get("/admin/mauvais-jeton")
    assert reponse.status_code == 404
    assert "Page introuvable" in reponse.text


def test_without_a_configured_token_the_page_does_not_exist(sans_jeton):
    """Mieux vaut pas d'administration qu'une administration ouverte."""
    tc, _ = sans_jeton
    assert tc.get("/admin/").status_code == 404
    assert tc.get("/admin/nimporte-quoi").status_code == 404


def test_failures_are_visible_with_their_cause(admin):
    tc, api = admin
    club = api.store.create_club("ASKO Kara")
    job = api.store.create("ASKO Kara", "Match", "/tmp/v.mp4", club_id=club.id)
    api.store.update(job.id, state=JobState.FAILED,
                     error="La vidéo n'a pas pu être lue.")

    page = tc.get(f"/admin/{JETON}").text
    assert "Échec" in page
    assert "pas pu être lue" in page


def test_a_dead_worker_is_announced(admin):
    tc, api = admin
    club = api.store.create_club("ASKO Kara")
    job = api.store.create("ASKO Kara", "Match", "/tmp/v.mp4", club_id=club.id)
    api.store.claim_next()

    import sqlite3
    from datetime import datetime, timedelta, timezone

    vieux = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    db = sqlite3.connect(api.store.path)
    db.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (vieux, job.id))
    db.commit(); db.close()

    page = tc.get(f"/admin/{JETON}").text
    assert "worker est probablement arrêté" in page


def test_an_empty_service_says_so(admin):
    tc, _ = admin
    assert "Aucun club" in tc.get(f"/admin/{JETON}").text


def test_the_page_warns_it_shows_private_links(admin):
    tc, api = admin
    api.store.create_club("ASKO Kara")
    assert "Ne pas la partager" in tc.get(f"/admin/{JETON}").text


def test_the_admin_page_is_not_indexable(admin):
    tc, api = admin
    api.store.create_club("ASKO Kara")
    assert "noindex" in tc.get(f"/admin/{JETON}").text
