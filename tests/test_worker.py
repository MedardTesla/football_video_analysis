"""Worker : traitement de la file.

Le pipeline réel exige des poids absents du dépôt et un GPU ; il est remplacé
par des doublures. Ce qui est testé ici, c'est l'enchaînement des états, le
nettoyage, et la traduction des pannes en messages lisibles par un club.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from football_analysis.config import Config
from service import worker
from service.jobs import JobState, JobStore
from service.storage import Storage


@pytest.fixture
def contexte(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    storage = Storage(tmp_path / "videos")
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video" * 100)
    job = store.create("US Valmont", "Valmont – Beaupré", str(video))
    return store, storage, job, video


def _pipeline_reussi(dossier: Path):
    from football_analysis.pipeline import PipelineResult

    def run(video_path, output_path, config, **kwargs):
        stats = Path(output_path).with_suffix(".json")
        stats.parent.mkdir(parents=True, exist_ok=True)
        stats.write_text('{"coverage": 0.62, "possession": {"0": 0.55, "1": 0.45},'
                         ' "players": [], "unmeasured_seconds": 2000}')
        Path(output_path).write_bytes(b"video annotee")
        return PipelineResult(
            video_path=Path(output_path), stats_path=stats,
            stats={"coverage": 0.62, "possession": {"0": 0.55, "1": 0.45},
                   "players": [], "unmeasured_seconds": 2000},
        )
    return run


def test_a_successful_run_produces_a_report(contexte, monkeypatch, tmp_path):
    store, storage, job, video = contexte
    worker.process(job.id, store, storage, Config(), run=_pipeline_reussi(tmp_path))

    fini = store.get(job.id)
    assert fini.state is JobState.DONE
    assert fini.progress == 1.0
    assert Path(fini.report_path).exists()
    assert "Valmont" in Path(fini.report_path).read_text(encoding="utf-8")


def test_the_source_video_is_deleted_after_analysis(contexte, monkeypatch, tmp_path):
    """Ce sont les images du club, pas les nôtres — et c'est le poste de
    stockage dominant."""
    store, storage, job, video = contexte
    worker.process(job.id, store, storage, Config(), run=_pipeline_reussi(tmp_path))
    assert not video.exists()


def test_the_coverage_reaches_the_report(contexte, monkeypatch, tmp_path):
    store, storage, job, _ = contexte
    worker.process(job.id, store, storage, Config(), run=_pipeline_reussi(tmp_path))
    page = Path(store.get(job.id).report_path).read_text(encoding="utf-8")
    assert "62%" in page


def test_a_missing_model_is_not_blamed_on_the_club(contexte, monkeypatch):
    store, storage, job, _ = contexte

    def echoue(*a, **k):
        raise FileNotFoundError("poids introuvables : models/player_detection.pt")

    worker.process(job.id, store, storage, Config(), run=echoue)

    fini = store.get(job.id)
    assert fini.state is JobState.FAILED
    assert "indisponible" in fini.error
    assert "models/" not in fini.error      # pas de détail interne


def test_an_unreadable_video_tells_the_club_what_to_do(contexte, monkeypatch):
    store, storage, job, _ = contexte

    def echoue(*a, **k):
        raise FileNotFoundError("vidéo illisible : /data/source.mp4")

    worker.process(job.id, store, storage, Config(), run=echoue)
    assert "déposer à nouveau" in store.get(job.id).error.lower()


def test_an_unexpected_error_stays_vague_but_polite(contexte, monkeypatch):
    store, storage, job, _ = contexte

    def echoue(*a, **k):
        raise RuntimeError("CUDA out of memory at 0x7f2a")

    worker.process(job.id, store, storage, Config(), run=echoue)

    erreur = store.get(job.id).error
    assert "CUDA" not in erreur
    assert "0x7f2a" not in erreur


def test_the_loop_drains_the_queue_then_stops(contexte, monkeypatch, tmp_path):
    store, storage, _, _ = contexte
    for i in range(3):
        v = tmp_path / f"v{i}.mp4"; v.write_bytes(b"x")
        store.create("Club", f"Match {i}", str(v))
    worker.serve(store, storage, Config(), once=True, run=_pipeline_reussi(tmp_path))

    assert store.pending_count() == 0
    assert store.counts_by_state().get("processing") is None


def test_the_loop_returns_on_an_empty_queue(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    storage = Storage(tmp_path / "videos")
    worker.serve(store, storage, Config(), once=True,
                 run=lambda *a, **k: None)     # ne doit pas boucler


def test_progress_reaches_the_store(contexte, monkeypatch, tmp_path):
    """La page de suivi n'affiche rien pendant une heure sans cela."""
    store, storage, job, _ = contexte
    reussi = _pipeline_reussi(tmp_path)

    def run_avec_progression(video_path, output_path, config, on_progress=None, **k):
        for f in (0.0, 0.25, 0.5, 0.75):
            on_progress(f)
        return reussi(video_path, output_path, config)

    worker.process(job.id, store, storage, Config(), run=run_avec_progression)
    assert store.get(job.id).progress == 1.0


def test_tiny_progress_steps_do_not_hammer_the_database(contexte, monkeypatch, tmp_path):
    """Chaque écriture est une transaction que l'API doit traverser pour
    afficher la page d'état. Un pas d'un millième ne mérite pas la sienne."""
    store, storage, job, _ = contexte
    reussi = _pipeline_reussi(tmp_path)
    ecritures = []
    original = store.update

    def compter(job_id, **champs):
        if "progress" in champs:
            ecritures.append(champs["progress"])
        return original(job_id, **champs)

    monkeypatch.setattr(store, "update", compter)

    def run_bavard(video_path, output_path, config, on_progress=None, **k):
        for i in range(500):
            on_progress(i / 500)
        return reussi(video_path, output_path, config)

    worker.process(job.id, store, storage, Config(), run=run_bavard)

    # 500 appels, au plus une écriture par point de pourcentage.
    assert len(ecritures) <= 101, len(ecritures)
    assert ecritures == sorted(ecritures)


def test_a_dead_workers_job_is_recovered_at_startup(contexte, monkeypatch, tmp_path):
    """Le cas réel : coupure de courant pendant une analyse."""
    from datetime import datetime, timedelta, timezone
    import sqlite3

    store, storage, job, _ = contexte
    store.claim_next()                       # un worker l'avait pris
    vieux = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    db = sqlite3.connect(store.path)
    db.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (vieux, job.id))
    db.commit(); db.close()

    worker.serve(store, storage, Config(), once=True, stale_seconds=900,
                 run=_pipeline_reussi(tmp_path))

    assert store.get(job.id).state is JobState.DONE


def test_a_live_job_is_not_stolen_by_a_second_worker(contexte, tmp_path):
    """Deux workers en parallèle : le second ne doit pas reprendre un match
    que le premier traite encore."""
    store, storage, job, _ = contexte
    store.claim_next()
    worker.serve(store, storage, Config(), once=True, stale_seconds=900)
    assert store.get(job.id).state is JobState.PROCESSING


class NotifierEspion:
    def __init__(self, echoue: bool = False):
        self.envoyes = []
        self.echoue = echoue

    def send(self, message):
        if self.echoue:
            raise RuntimeError("serveur SMTP injoignable")
        self.envoyes.append(message)
        return True


def test_the_club_is_told_when_the_report_is_ready(contexte, tmp_path):
    store, storage, job, _ = contexte
    store.update(job.id, contact="entraineur@club.fr")
    espion = NotifierEspion()

    worker.process(job.id, store, storage, Config(),
                   run=_pipeline_reussi(tmp_path), notifier=espion)

    assert len(espion.envoyes) == 1
    assert espion.envoyes[0].destinataire == "entraineur@club.fr"
    assert job.public_url in espion.envoyes[0].corps


def test_the_club_is_told_when_the_analysis_fails(contexte):
    store, storage, job, _ = contexte
    store.update(job.id, contact="entraineur@club.fr")
    espion = NotifierEspion()

    def echoue(*a, **k):
        raise FileNotFoundError("vidéo illisible : /data/source.mp4")

    worker.process(job.id, store, storage, Config(), run=echoue, notifier=espion)
    assert len(espion.envoyes) == 1
    assert "déposer à nouveau" in espion.envoyes[0].corps.lower()


def test_no_contact_means_no_message(contexte, tmp_path):
    store, storage, job, _ = contexte
    espion = NotifierEspion()
    worker.process(job.id, store, storage, Config(),
                   run=_pipeline_reussi(tmp_path), notifier=espion)
    assert espion.envoyes == []


def test_a_failed_send_does_not_lose_the_report(contexte, tmp_path):
    """Perdre une analyse parce qu'un serveur SMTP est injoignable serait
    absurde : le rapport existe, le lien fonctionne."""
    store, storage, job, _ = contexte
    store.update(job.id, contact="entraineur@club.fr")

    worker.process(job.id, store, storage, Config(),
                   run=_pipeline_reussi(tmp_path), notifier=NotifierEspion(echoue=True))

    fini = store.get(job.id)
    assert fini.state is JobState.DONE
    assert Path(fini.report_path).exists()
