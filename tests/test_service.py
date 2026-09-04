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
    import service.settings as settings
    import service.api as api

    # settings d'abord : c'est lui qui lit FA_DATA_ROOT, et api en dérive
    # son magasin. Ne recharger qu'api laisserait tous les tests partager
    # la même base.
    importlib.reload(settings)
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


def test_the_progress_bar_shows_during_analysis(client):
    tc, api = client
    _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    api.store.update(job.id, state=JobState.PROCESSING, progress=0.42)

    page = tc.get(job.public_url).text
    assert "42 % analysé" in page
    assert "width:42%" in page


def test_the_queued_page_has_no_progress_bar(client):
    """Rien n'a encore commencé : afficher 0 % laisserait croire à un blocage."""
    tc, api = client
    _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    page = tc.get(job.public_url).text
    # La classe existe dans la feuille de style ; c'est le balisage qui doit
    # être absent.
    assert 'class="jauge"' not in page
    assert "En attente" in page


def test_a_finished_analysis_stops_polling(client):
    tc, api = client
    _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    api.store.update(job.id, state=JobState.DONE, report_path="/tmp/r.html")
    assert "setInterval" not in tc.get(job.public_url).text


# --- Reprise après panne -----------------------------------------------------

def _vieillir_dossier(chemin, secondes=7200):
    """Fait comme si le dossier datait de plusieurs heures."""
    import os
    import time

    quand = time.time() - secondes
    for f in list(chemin.rglob("*")) + [chemin]:
        os.utime(f, (quand, quand))


def _vieillir(store, job_id, secondes):
    """Fait comme si le job n'avait plus donné signe de vie depuis N secondes."""
    import sqlite3
    from datetime import datetime, timedelta, timezone

    vieux = (datetime.now(timezone.utc) - timedelta(seconds=secondes)).isoformat()
    db = sqlite3.connect(store.path)
    db.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (vieux, job_id))
    db.commit()
    db.close()


def test_a_job_abandoned_by_a_dead_worker_is_requeued(store):
    """Sans cela, une coupure de courant laisse le club attendre un rapport
    qui ne viendra jamais."""
    job = store.create("Club", "Match", "/tmp/v.mp4")
    store.claim_next()
    _vieillir(store, job.id, 3600)

    assert store.reclaim_stale(timeout_seconds=900) == [job.id]
    assert store.get(job.id).state is JobState.QUEUED
    assert store.get(job.id).progress == 0.0


def test_a_job_still_alive_is_left_alone(store):
    job = store.create("Club", "Match", "/tmp/v.mp4")
    store.claim_next()
    assert store.reclaim_stale(timeout_seconds=900) == []
    assert store.get(job.id).state is JobState.PROCESSING


def test_a_heartbeat_prevents_reclaiming(store):
    """Un match long ne doit pas être repris pendant qu'il progresse."""
    job = store.create("Club", "Match", "/tmp/v.mp4")
    store.claim_next()
    _vieillir(store, job.id, 3600)
    store.heartbeat(job.id)
    assert store.reclaim_stale(timeout_seconds=900) == []


def test_a_job_that_keeps_killing_the_worker_is_given_up(store):
    """Un fichier qui fait planter le worker serait repris à l'infini et
    bloquerait toute la file derrière lui."""
    job = store.create("Club", "Match", "/tmp/v.mp4")
    for _ in range(3):
        store.claim_next()
        _vieillir(store, job.id, 3600)
        store.reclaim_stale(timeout_seconds=900, max_attempts=3)

    fini = store.get(job.id)
    assert fini.state is JobState.FAILED
    assert "plusieurs reprises" in fini.error
    assert store.pending_count() == 0


def test_attempts_are_counted_on_each_claim(store):
    job = store.create("Club", "Match", "/tmp/v.mp4")
    assert store.claim_next().attempts == 1
    store.update(job.id, state=JobState.QUEUED)
    assert store.claim_next().attempts == 2


def test_an_older_database_gains_the_new_column(tmp_path):
    """Les bases créées avant cette version n'ont pas la colonne attempts."""
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
        "INSERT INTO jobs VALUES ('a','t','Club','Match','/tmp/v.mp4','queued',"
        "0,NULL,NULL,NULL,NULL,'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')"
    )
    db.commit()
    db.close()

    store = JobStore(chemin)
    assert store.get("a").attempts == 0
    assert store.claim_next().attempts == 1


def test_health_flags_a_dead_worker(client):
    """Si personne ne reprend la file, la sonde doit le voir sans lire le corps."""
    tc, api = client
    _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    api.store.claim_next()
    _vieillir(api.store, job.id, 7200)

    reponse = tc.get("/sante")
    assert reponse.status_code == 503
    assert reponse.json()["bloques"] == 1


def test_health_is_green_when_the_queue_moves(client):
    tc, _ = client
    _deposer(tc)
    reponse = tc.get("/sante")
    assert reponse.status_code == 200
    assert reponse.json()["bloques"] == 0


def test_the_api_does_not_load_the_machine_learning_stack():
    """L'API reçoit des fichiers et sert des pages : elle n'a besoin ni de
    torch ni d'OpenCV.

    Le worker les charge, lui, mais tardivement. Faire transiter une simple
    constante par `worker.py` suffisait à imposer plusieurs gigaoctets de
    bibliothèques de calcul sur la machine qui sert le formulaire de dépôt.
    """
    import subprocess
    import sys

    code = (
        "import sys; import service.api; "
        "lourds = [m for m in ('torch','ultralytics','cv2','supervision','transformers') "
        "if m in sys.modules]; print(','.join(lourds))"
    )
    sortie = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert sortie.returncode == 0, sortie.stderr[-500:]
    assert sortie.stdout.strip() == "", f"modules lourds chargés : {sortie.stdout.strip()}"


def test_the_worker_module_is_importable_without_the_pipeline():
    """Permet de tester la file et les erreurs sans la pile de calcul."""
    import subprocess
    import sys

    code = (
        "import sys; import service.worker; "
        "print('torch' in sys.modules or 'ultralytics' in sys.modules)"
    )
    sortie = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert sortie.returncode == 0, sortie.stderr[-500:]
    assert sortie.stdout.strip() == "False"


# --- Protection du disque ----------------------------------------------------

def test_an_upload_is_refused_when_the_disk_is_nearly_full(storage, monkeypatch):
    """Le dépôt est ouvert sans compte : quelques envois simultanés
    suffiraient à saturer le disque et à faire tomber le service pour tous
    les clubs, y compris ceux dont l'analyse est en cours."""
    import service.storage as module

    monkeypatch.setattr(storage, "free_bytes", lambda: module.RESERVE_DISQUE - 1)
    with pytest.raises(UploadRefuse, match="saturé"):
        storage.save_upload("abc", "match.mp4", io.BytesIO(b"x" * 1000))
    assert not list(storage.job_dir("abc").glob("source*"))


def test_an_upload_proceeds_when_the_disk_has_room(storage, monkeypatch):
    import service.storage as module

    monkeypatch.setattr(storage, "free_bytes", lambda: module.RESERVE_DISQUE * 4)
    chemin = storage.save_upload("abc", "match.mp4", io.BytesIO(b"x" * 1000))
    assert Path(chemin).exists()


def test_old_matches_are_purged(storage, tmp_path):
    """La vidéo annotée est le seul poste qui grossit sans limite."""
    import os
    import time

    for nom, age_jours in (("vieux", 200), ("recent", 3)):
        dossier = storage.job_dir(nom)
        fichier = dossier / "analyse.mp4"
        fichier.write_bytes(b"x" * 100)
        quand = time.time() - age_jours * 86400
        os.utime(fichier, (quand, quand))
        os.utime(dossier, (quand, quand))

    assert storage.purge_older_than(days=90) == ["vieux"]
    assert not (storage.root / "vieux").exists()
    assert (storage.root / "recent").exists()


def test_purging_an_empty_store_is_harmless(storage):
    assert storage.purge_older_than(days=90) == []


def test_health_flags_a_full_disk(client, monkeypatch):
    """La sonde doit voir venir le disque plein, pas le constater après."""
    tc, api = client
    import service.storage as module

    monkeypatch.setattr(api.storage, "free_bytes", lambda: module.RESERVE_DISQUE - 1)
    reponse = tc.get("/sante")
    assert reponse.status_code == 503
    assert reponse.json()["sature"] is True


def test_health_reports_free_space_when_healthy(client):
    tc, _ = client
    corps = tc.get("/sante").json()
    assert corps["sature"] is False
    assert corps["disque_libre_go"] > 0


# --- Contact et notification -------------------------------------------------

def test_a_contact_is_stored_with_the_match(client):
    tc, api = client
    tc.post("/matches",
            data={"club": "US Valmont", "match_name": "Match",
                  "contact": "entraineur@club.fr"},
            files={"video": ("m.mp4", b"x" * 500, "video/mp4")})
    assert api.store.list_for_club("US Valmont")[0].contact == "entraineur@club.fr"


def test_an_invalid_address_is_refused_not_ignored(client):
    """L'ignorer ferait attendre au club un message qui ne viendrait jamais."""
    tc, api = client
    reponse = tc.post("/matches",
        data={"club": "US Valmont", "match_name": "Match", "contact": "pas-une-adresse"},
        files={"video": ("m.mp4", b"x" * 500, "video/mp4")})
    assert reponse.status_code == 400
    assert "invalide" in reponse.text


def test_the_contact_stays_optional(client):
    tc, api = client
    assert _deposer(tc).status_code == 201
    assert api.store.list_for_club("US Valmont")[0].contact == ""


def test_the_confirmation_mentions_the_address_when_given(client):
    tc, _ = client
    reponse = tc.post("/matches",
        data={"club": "US Valmont", "match_name": "Match",
              "contact": "entraineur@club.fr"},
        files={"video": ("m.mp4", b"x" * 500, "video/mp4")})
    assert "entraineur@club.fr" in reponse.text
    assert "prévenu" in reponse.text


def test_the_form_warns_that_the_link_travels_by_mail(client):
    """Une adresse mal saisie envoie le lien privé à un inconnu."""
    tc, _ = client
    page = tc.get("/").text
    assert "vérifiez l" in page.lower()
    assert "facultatif" in page


def test_an_older_database_gains_the_contact_column(tmp_path):
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
        "INSERT INTO jobs VALUES ('a','t','Club','Match','/tmp/v.mp4','queued',"
        "0,NULL,NULL,NULL,NULL,'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')"
    )
    db.commit(); db.close()

    store = JobStore(chemin)
    assert store.get("a").contact == ""
    nouveau = store.create("Club", "Autre", "/tmp/v.mp4", contact="a@b.fr")
    assert store.get(nouveau.id).contact == "a@b.fr"


# --- Espace du club ----------------------------------------------------------

def test_a_first_upload_creates_a_club_space(client):
    tc, api = client
    reponse = _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    assert job.club_id
    club = api.store.get_club(job.club_id)
    assert club.name == "US Valmont"
    assert club.public_url in reponse.text


def test_two_clubs_with_the_same_name_stay_separate(client):
    """Le cas qui interdit de rattacher les matchs au seul nom : deux
    catégories d'un même village portent souvent le même nom."""
    tc, api = client
    _deposer(tc, "Match A")
    _deposer(tc, "Match B")
    clubs = {j.club_id for j in api.store.list_for_club("US Valmont")}
    assert len(clubs) == 2
    for club_id in clubs:
        assert len(api.store.matches_of_club(club_id)) == 1


def test_the_club_page_lists_its_matches(client):
    tc, api = client
    _deposer(tc, "Valmont – Beaupré")
    club = api.store.get_club(api.store.list_for_club("US Valmont")[0].club_id)

    page = tc.get(club.public_url)
    assert page.status_code == 200
    assert "Valmont – Beaupré" in page.text
    assert "US Valmont" in page.text


def test_the_club_page_needs_its_token(client):
    tc, api = client
    _deposer(tc)
    club = api.store.get_club(api.store.list_for_club("US Valmont")[0].club_id)
    assert tc.get(f"/c/{club.id}/faux-jeton").status_code == 404
    assert tc.get("/c/inconnu/jeton").status_code == 404


def test_uploading_from_a_club_space_joins_it(client):
    tc, api = client
    _deposer(tc, "Premier")
    club = api.store.get_club(api.store.list_for_club("US Valmont")[0].club_id)

    formulaire = tc.get(f"{club.public_url}/deposer")
    assert formulaire.status_code == 200
    assert club.token in formulaire.text          # jeton en champ caché

    tc.post("/matches",
            data={"match_name": "Second", "club_id": club.id, "club_token": club.token},
            files={"video": ("m.mp4", b"x" * 500, "video/mp4")})

    matchs = api.store.matches_of_club(club.id)
    assert {m.match_name for m in matchs} == {"Premier", "Second"}


def test_a_forged_club_token_is_refused_at_upload(client):
    tc, api = client
    _deposer(tc)
    club = api.store.get_club(api.store.list_for_club("US Valmont")[0].club_id)

    reponse = tc.post("/matches",
        data={"match_name": "Intrus", "club_id": club.id, "club_token": "faux"},
        files={"video": ("m.mp4", b"x" * 500, "video/mp4")})
    assert reponse.status_code == 404
    assert len(api.store.matches_of_club(club.id)) == 1


def test_the_club_token_never_reaches_the_address_bar(client):
    """Le formulaire est parfois rempli sur un poste partagé au club : le
    jeton ne doit apparaître ni dans l'historique ni dans les journaux."""
    tc, api = client
    _deposer(tc)
    club = api.store.get_club(api.store.list_for_club("US Valmont")[0].club_id)
    page = tc.get(f"{club.public_url}/deposer").text
    assert f'name="club_token" value="{club.token}"' in page
    assert 'method="post"' in page.lower()


def test_an_empty_club_space_says_so(client):
    tc, api = client
    club = api.store.create_club("US Valmont")
    page = tc.get(club.public_url)
    assert "Aucun match" in page.text


def test_the_club_page_shows_each_match_state(client):
    tc, api = client
    _deposer(tc, "Terminé")
    job = api.store.list_for_club("US Valmont")[0]
    api.store.update(job.id, state=JobState.DONE, report_path="/tmp/r.html")

    page = tc.get(api.store.get_club(job.club_id).public_url).text
    assert "Prêt" in page
    assert f"{job.public_url}/rapport" in page


def test_an_older_database_gains_the_club_column(tmp_path):
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
        "INSERT INTO jobs VALUES ('a','t','Club','Match','/tmp/v.mp4','queued',"
        "0,NULL,NULL,NULL,NULL,'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')"
    )
    db.commit(); db.close()

    store = JobStore(chemin)
    assert store.get("a").club_id == ""
    club = store.create_club("Nouveau")
    nouveau = store.create("Nouveau", "Match", "/tmp/v.mp4", club_id=club.id)
    assert store.matches_of_club(club.id) == [store.get(nouveau.id)]


# --- Désignation de l'équipe et tendance -------------------------------------

def _match_termine(api, club, nom, jour, possession):
    job = api.store.create("US Valmont", nom, "/tmp/v.mp4", club_id=club.id)
    api.store.update(job.id, state=JobState.DONE, report_path="/tmp/r.html",
                     stats={"possession": {"0": possession, "1": 1 - possession},
                            "control": {"0": possession, "1": 1 - possession},
                            "coverage": 0.85})
    return api.store.get(job.id)


def test_a_finished_match_offers_the_team_selector(client):
    tc, api = client
    club = api.store.create_club("US Valmont")
    _match_termine(api, club, "Match", 10, 0.6)
    page = tc.get(club.public_url).text
    assert "Votre équipe" in page
    assert 'name="team" value="0"' in page and 'name="team" value="1"' in page


def test_designating_a_team_is_recorded(client):
    tc, api = client
    club = api.store.create_club("US Valmont")
    job = _match_termine(api, club, "Match", 10, 0.6)

    reponse = tc.post(f"{club.public_url}/match/{job.id}/equipe",
                      data={"team": "1"}, follow_redirects=False)
    assert reponse.status_code == 303
    assert api.store.get(job.id).our_team == 1


def test_clicking_the_same_team_again_clears_it(client):
    """Le seul moyen de revenir en arrière après une erreur."""
    tc, api = client
    club = api.store.create_club("US Valmont")
    job = _match_termine(api, club, "Match", 10, 0.6)

    tc.post(f"{club.public_url}/match/{job.id}/equipe", data={"team": "0"})
    tc.post(f"{club.public_url}/match/{job.id}/equipe", data={"team": "0"})
    assert api.store.get(job.id).our_team is None


def test_a_club_cannot_designate_another_clubs_match(client):
    tc, api = client
    mien = api.store.create_club("US Valmont")
    autre = api.store.create_club("AS Beaupré")
    job = _match_termine(api, autre, "Match", 10, 0.6)

    reponse = tc.post(f"{mien.public_url}/match/{job.id}/equipe", data={"team": "0"})
    assert reponse.status_code == 404
    assert api.store.get(job.id).our_team is None


def test_an_invalid_team_is_refused(client):
    tc, api = client
    club = api.store.create_club("US Valmont")
    job = _match_termine(api, club, "Match", 10, 0.6)
    assert tc.post(f"{club.public_url}/match/{job.id}/equipe",
                   data={"team": "7"}).status_code == 400


def test_the_trend_appears_once_two_matches_are_designated(client):
    tc, api = client
    club = api.store.create_club("US Valmont")
    for i, part in enumerate((0.45, 0.55, 0.62)):
        job = _match_termine(api, club, f"Match {i}", 10 + i, part)
        api.store.set_our_team(job.id, 0)

    page = tc.get(club.public_url).text
    assert "Tendance de la saison" in page
    assert "<svg" in page
    assert "Possession" in page and "Contrôle du terrain" in page


def test_without_designation_the_trend_explains_what_to_do(client):
    tc, api = client
    club = api.store.create_club("US Valmont")
    _match_termine(api, club, "Match", 10, 0.6)
    page = tc.get(club.public_url).text
    assert "Désignez votre équipe" in page
    assert "<svg" not in page


def test_the_trend_never_aggregates_individual_distances(client):
    """Leur imprécision se cumulerait au lieu de se compenser."""
    tc, api = client
    club = api.store.create_club("US Valmont")
    for i in range(2):
        job = _match_termine(api, club, f"Match {i}", 10 + i, 0.5)
        api.store.set_our_team(job.id, 0)
    page = tc.get(club.public_url).text
    assert "distances individuelles ne sont pas agrégées" in page


# --- Nommage des joueurs -----------------------------------------------------

def _match_avec_stats(api, nom="Match"):
    """Match terminé, avec rapport et statistiques sur disque."""
    import json as _json
    from pathlib import Path as _Path

    job = api.store.create("US Valmont", nom, "/tmp/v.mp4")
    dossier = api.storage.job_dir(job.id)
    stats = {
        "coverage": 0.9, "possession": {"0": 0.55, "1": 0.45},
        "players": [
            {"track_id": 4, "team": 0, "distance_m": 9800.0,
             "top_speed_ms": 8.2, "seconds_seen": 5200.0},
            {"track_id": 9, "team": 1, "distance_m": 8700.0,
             "top_speed_ms": 9.0, "seconds_seen": 5000.0},
        ],
    }
    (dossier / "analyse.json").write_text(_json.dumps(stats))
    rapport = dossier / "rapport.html"
    rapport.write_text("<h1>Rapport</h1>", encoding="utf-8")
    api.store.update(job.id, state=JobState.DONE, stats=stats,
                     report_path=str(rapport))
    return api.store.get(job.id)


def test_the_naming_page_lists_the_durable_tracks(client):
    tc, api = client
    job = _match_avec_stats(api)
    page = tc.get(f"{job.public_url}/joueurs")
    assert page.status_code == 200
    assert 'name="nom_4"' in page.text and 'name="nom_9"' in page.text


def test_naming_is_refused_before_the_analysis_ends(client):
    tc, api = client
    _deposer(tc)
    job = api.store.list_for_club("US Valmont")[0]
    assert tc.get(f"{job.public_url}/joueurs").status_code == 409


def test_saving_names_rewrites_the_report(client):
    """Sans régénération, les noms ne vivraient que dans le formulaire."""
    tc, api = client
    job = _match_avec_stats(api)

    reponse = tc.post(f"{job.public_url}/joueurs",
                      data={"nom_4": "Kossi Adjovi", "nom_9": ""},
                      follow_redirects=False)
    assert reponse.status_code == 303
    assert api.store.get(job.id).player_names == {"4": "Kossi Adjovi"}

    from pathlib import Path
    assert "Kossi Adjovi" in Path(job.report_path).read_text(encoding="utf-8")


def test_names_need_the_match_token(client):
    tc, api = client
    job = _match_avec_stats(api)
    assert tc.post(f"/m/{job.id}/faux/joueurs", data={"nom_4": "X"}).status_code == 404


def test_the_finished_page_offers_naming(client):
    tc, api = client
    job = _match_avec_stats(api)
    assert "Nommer les joueurs" in tc.get(job.public_url).text


def test_orphaned_folders_are_removed(store, storage):
    """Une panne ou une suppression pendant l'analyse laisse des fichiers
    sans ligne en base ; la purge par ancienneté ne les prendrait que
    quatre-vingt-dix jours plus tard."""
    job = store.create("Club", "Match", "/tmp/v.mp4")
    (storage.job_dir(job.id) / "rapport.html").write_text("connu")
    fantome = storage.job_dir("fantome")
    (fantome / "rapport.html").write_text("orphelin")
    _vieillir_dossier(fantome)

    assert storage.purge_orphans(store.all_ids()) == ["fantome"]
    assert (storage.root / job.id).exists()
    assert not (storage.root / "fantome").exists()


def test_purging_orphans_on_an_empty_store_is_harmless(storage):
    assert storage.purge_orphans(set()) == []


def test_all_ids_lists_every_match(store):
    a = store.create("Club", "A", "/tmp/v.mp4")
    b = store.create("Club", "B", "/tmp/v.mp4")
    assert store.all_ids() == {a.id, b.id}


def test_a_fresh_upload_is_never_mistaken_for_an_orphan(store, storage):
    """La liste des matchs connus est lue à un instant donné : un dépôt
    arrivé juste après n'y figure pas. Sans délai de grâce, la vidéo d'un
    club serait effacée pendant qu'il la téléverse."""
    connus = store.all_ids()

    nouveau = store.create("US Valmont", "Match", "/tmp/v.mp4")
    (storage.job_dir(nouveau.id) / "source.mp4").write_bytes(b"video")

    assert storage.purge_orphans(connus) == []
    assert (storage.root / nouveau.id / "source.mp4").exists()


def test_an_old_orphan_is_still_removed(store, storage):
    import os
    import time

    dossier = storage.job_dir("fantome")
    fichier = dossier / "rapport.html"
    fichier.write_text("orphelin")
    vieux = time.time() - 7200
    os.utime(fichier, (vieux, vieux))
    os.utime(dossier, (vieux, vieux))

    assert storage.purge_orphans(store.all_ids()) == ["fantome"]
