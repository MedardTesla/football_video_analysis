"""Worker : consomme la file et produit le rapport.

Séparé de l'API à dessein. L'API tient sur une petite machine ; le worker a
besoin d'un GPU, dont le coût horaire est sans commune mesure. Les deux ne
partagent que la base SQLite et le dossier de stockage, ce qui permet de les
déployer séparément — et d'arrêter le GPU quand la file est vide.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Callable

from football_analysis.config import Config
from football_analysis.report import ReportMeta, write as write_report

from .jobs import Job, JobState, JobStore
from .fetch import LienRefuse, telecharger
from .notify import LogNotifier, Notifier, analysis_failed, report_ready
from .settings import BASE_URL, STALE_SECONDS
from .storage import Storage

log = logging.getLogger("worker")


def _load_pipeline() -> Callable:
    """Import tardif du pipeline.

    `football_analysis.pipeline` tire torch, ultralytics et OpenCV. Les charger
    à l'import du module obligerait l'API — qui ne fait que recevoir des
    fichiers et servir des pages — à embarquer toute la pile de calcul.
    """
    from football_analysis.pipeline import run

    return run


def _date_du_match(job: Job) -> date | None:
    """Date saisie par le club, ou rien. Jamais celle de l'analyse."""
    try:
        return date.fromisoformat(job.played_on) if job.played_on else None
    except ValueError:
        return None


def _prevenir(job: Job, notifier: Notifier, message) -> None:
    """Envoie l'avis au club, sans jamais compromettre l'analyse.

    Un envoi raté n'annule pas un rapport produit : le lien fonctionne, seul
    l'avis manque. L'inverse — perdre une analyse parce qu'un serveur SMTP est
    injoignable — serait absurde.
    """
    if not job.contact:
        return
    message.destinataire = job.contact
    try:
        notifier.send(message)
    except Exception:                                  # noqa: BLE001
        log.exception("notification impossible pour le match %s", job.id)


def process(
    job_id: str,
    store: JobStore,
    storage: Storage,
    config: Config,
    run: Callable | None = None,
    notifier: Notifier | None = None,
) -> None:
    """`run` permet d'injecter le pipeline ; par défaut il est chargé tardivement."""
    run = run or _load_pipeline()
    notifier = notifier or LogNotifier()
    job = store.get(job_id)
    if job is None:
        return

    dossier = storage.job_dir(job.id)

    # La progression n'est écrite que si elle a bougé d'un point : le pipeline
    # peut rappeler souvent, et chaque écriture est une transaction que l'API
    # doit pouvoir traverser pour afficher la page d'état.
    dernier = 0.0

    def progression(fraction: float) -> None:
        nonlocal dernier
        if fraction - dernier >= 0.01 or fraction >= 1.0:
            dernier = fraction
            # Cette écriture fait aussi office de battement de cœur : elle
            # rafraîchit updated_at, ce qui empêche `reclaim_stale` de
            # considérer le match abandonné pendant qu'il progresse.
            store.update(job.id, progress=round(fraction, 3))

    try:
        # Récupérer la vidéo si le club a donné un lien plutôt qu'un fichier.
        # Fait ici et non au dépôt : le téléchargement peut durer, et bloquer
        # une requête HTTP pendant ce temps la ferait expirer.
        if job.source_url and not job.video_path:
            log.info("téléchargement de la source du match %s", job.id)
            chemin = telecharger(job.source_url, dossier / "source")
            store.update(job.id, video_path=str(chemin))
            job.video_path = str(chemin)

        resultat = run(
            job.video_path,
            dossier / "analyse.mp4",
            config,
            on_progress=progression,
        )
        rapport = write_report(
            resultat.stats_path,
            dossier / "rapport.html",
            # Pas de date par défaut : afficher celle de l'analyse à la place
            # de celle du match serait un contresens sur le livrable.
            ReportMeta(match_name=job.match_name, played_on=_date_du_match(job)),
            radar_png=resultat.radar_path,
        )
        store.update(
            job.id,
            state=JobState.DONE,
            progress=1.0,
            stats=resultat.stats,
            report_path=str(rapport),
            video_output_path=str(resultat.video_path),
        )
        # La source ne sert plus, et c'est le poste de stockage dominant.
        Path(job.video_path).unlink(missing_ok=True)
        log.info("match %s analysé", job.id)
        espace = store.get_club(job.club_id) if job.club_id else None
        _prevenir(job, notifier, report_ready(
            job.match_name, job.public_url, BASE_URL,
            espace.public_url if espace else "",
        ))

    except Exception as erreur:                      # noqa: BLE001
        # Le message est lu par un club, pas par un développeur : la trace
        # complète va dans les logs, pas dans le rapport.
        log.exception("échec du match %s", job.id)
        raison = _message_lisible(erreur)
        store.update(job.id, state=JobState.FAILED, error=raison)
        _prevenir(
            job, notifier, analysis_failed(job.match_name, raison, job.public_url, BASE_URL)
        )


def _message_lisible(erreur: Exception) -> str:
    """Traduit une exception technique en cause actionnable pour le club."""
    texte = str(erreur)
    if isinstance(erreur, LienRefuse):
        return texte
    if isinstance(erreur, FileNotFoundError) and "poids" in texte:
        return "Service momentanément indisponible : modèle d'analyse absent."
    if "vidéo illisible" in texte:
        return (
            "La vidéo n'a pas pu être lue. Vérifier qu'elle se lit correctement "
            "puis la déposer à nouveau."
        )
    if "aucun joueur détecté" in texte:
        return (
            "Aucun joueur détecté. La caméra est peut-être trop éloignée, ou la "
            "vidéo ne montre pas un match."
        )
    return "L'analyse a échoué. Nous avons été prévenus et revenons vers vous."


def serve(
    store: JobStore, storage: Storage, config: Config | None = None,
    poll_seconds: float = 5.0, once: bool = False,
    stale_seconds: float = STALE_SECONDS, run: Callable | None = None,
    notifier: Notifier | None = None,
) -> None:
    """Boucle de traitement. `once=True` vide la file puis rend la main."""
    config = config or Config()

    # Au démarrage : récupérer ce qu'un worker mort a laissé en plan. C'est le
    # cas normal après une coupure de courant ou un redémarrage de machine.
    repris = store.reclaim_stale(stale_seconds)
    if repris:
        log.warning("%d match(s) repris après interruption : %s", len(repris), repris)

    purges = storage.purge_older_than()
    if purges:
        log.info("%d match(s) purgés après rétention : %s", len(purges), purges)

    while True:
        job = store.claim_next()
        if job is None:
            if once:
                return
            # Un worker mort pendant que celui-ci tourne : on récupère aussi.
            store.reclaim_stale(stale_seconds)
            time.sleep(poll_seconds)
            continue

        # Purger avant de traiter, pas après : c'est maintenant qu'il faut de
        # la place, et un disque plein ferait échouer l'analyse en cours de
        # route après plusieurs dizaines de minutes de GPU.
        storage.purge_older_than()
        process(job.id, store, storage, config, run=run, notifier=notifier)
        # Ne pas tester la file avec claim_next : elle réserverait le job
        # suivant avant de l'abandonner en état « en cours », définitivement.
        if once and store.pending_count() == 0:
            return
