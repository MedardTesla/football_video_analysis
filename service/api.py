"""API du service d'analyse.

Trois usages seulement :
  - un club dépose une vidéo et reçoit un lien ;
  - il ouvre ce lien pour suivre l'avancement puis lire le rapport ;
  - l'exploitant surveille la file.

Pas de comptes ni de mots de passe : un club de village ne créera pas de
compte pour trois matchs par saison. Le lien porte un jeton imprévisible, ce
qui est le bon compromis — et se remplacera par une vraie authentification si
un client la demande.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, RedirectResponse,
)

from .jobs import Club, Job, JobState, JobStore
from .notify import looks_like_email
from . import season as saison_mod
from .settings import DATA_ROOT, STALE_SECONDS
from .storage import Storage, UploadRefuse
from .web import pages

app = FastAPI(title="Analyse de match", docs_url=None, redoc_url=None)
store = JobStore(DATA_ROOT / "jobs.db")
storage = Storage(DATA_ROOT / "videos")


def _authenticate(job_id: str, token: str) -> Job:
    job = store.authenticate(job_id, token)
    if job is None:
        # Même réponse qu'un identifiant inexistant : distinguer les deux cas
        # confirmerait au visiteur qu'un match existe à cette adresse.
        raise HTTPException(status_code=404, detail="Match introuvable.")
    return job


@app.get("/", response_class=HTMLResponse)
def accueil() -> str:
    return pages.upload_form()


def _authenticate_club(club_id: str, token: str) -> Club:
    club = store.authenticate_club(club_id, token)
    if club is None:
        raise HTTPException(status_code=404, detail="Espace introuvable.")
    return club


@app.get("/c/{club_id}/{token}", response_class=HTMLResponse)
def espace_club(club_id: str, token: str) -> str:
    club = _authenticate_club(club_id, token)
    matchs = store.matches_of_club(club.id)
    return pages.club_page(
        club, matchs, store.pending_count(), saison_mod.build(matchs)
    )


@app.post("/c/{club_id}/{token}/match/{job_id}/equipe")
def designer_equipe(
    club_id: str, token: str, job_id: str, team: int = Form(...)
) -> RedirectResponse:
    """Désigne l'équipe du club dans un match.

    Sans cette désignation, aucune tendance de saison n'est possible : les
    libellés « équipe A » et « équipe B » d'un rapport viennent d'un
    regroupement automatique et ne désignent pas la même équipe d'un match à
    l'autre.
    """
    club = _authenticate_club(club_id, token)
    job = store.get(job_id)
    if job is None or job.club_id != club.id:
        raise HTTPException(status_code=404, detail="Match introuvable.")
    if team not in (0, 1):
        raise HTTPException(status_code=400, detail="Équipe invalide.")

    # Recliquer sur l'équipe déjà désignée l'efface : c'est le seul moyen de
    # revenir en arrière après une erreur.
    store.set_our_team(job_id, None if job.our_team == team else team)
    return RedirectResponse(club.public_url, status_code=303)


@app.get("/c/{club_id}/{token}/deposer", response_class=HTMLResponse)
def deposer_depuis_espace(club_id: str, token: str) -> str:
    """Formulaire pré-rattaché au club : le match rejoindra son espace."""
    return pages.upload_form(club=_authenticate_club(club_id, token))


@app.post("/matches", response_class=HTMLResponse)
async def deposer(
    request: Request,
    match_name: str = Form(...),
    club: str = Form(""),
    club_id: str = Form(""),
    club_token: str = Form(""),
    contact: str = Form(""),
    video: UploadFile = None,
) -> HTMLResponse:
    # Un dépôt venant d'un espace de club porte son jeton ; sinon un nouvel
    # espace est créé. Rattacher au seul nom serait dangereux : deux clubs
    # homonymes, fréquents entre catégories d'un même village, se
    # partageraient leurs rapports.
    espace: Club | None = None
    if club_id and club_token:
        espace = store.authenticate_club(club_id, club_token)
        if espace is None:
            return HTMLResponse(
                pages.upload_form(erreur="Espace de club introuvable."), 404
            )
        club = espace.name

    if not club.strip():
        return HTMLResponse(pages.upload_form(erreur="Nom du club manquant."), 400)

    if video is None or not video.filename:
        return HTMLResponse(
            pages.upload_form(erreur="Aucune vidéo sélectionnée.", club=espace), 400
        )

    # Une adresse invalide est refusée plutôt qu'ignorée : le club croirait
    # être prévenu et attendrait un message qui ne viendrait jamais.
    contact = contact.strip()
    if contact and not looks_like_email(contact):
        return HTMLResponse(
            pages.upload_form(erreur=f"Adresse e-mail invalide : {contact}", club=espace),
            400,
        )

    if espace is None:
        espace = store.create_club(club.strip())

    job = store.create(
        espace.name, match_name.strip(), video_path="",
        contact=contact, club_id=espace.id,
    )
    try:
        chemin = storage.save_upload(job.id, video.filename, video.file)
    except UploadRefuse as refus:
        store.update(job.id, state=JobState.FAILED, error=str(refus))
        return HTMLResponse(pages.upload_form(erreur=str(refus), club=espace), 400)

    store.update(job.id, video_path=str(chemin))
    job.video_path = str(chemin)
    return HTMLResponse(pages.upload_done(job, store.pending_count(), espace), 201)


@app.get("/m/{job_id}/{token}", response_class=HTMLResponse)
def suivre(job_id: str, token: str) -> str:
    return pages.status_page(_authenticate(job_id, token), store.pending_count())


@app.get("/m/{job_id}/{token}/etat")
def etat(job_id: str, token: str) -> JSONResponse:
    """Interrogé par la page d'état, qui se rafraîchit toute seule."""
    job = _authenticate(job_id, token)
    return JSONResponse({
        "state": job.state.value,
        "progress": job.progress,
        "error": job.error,
        "terminal": job.state.terminal,
    })


@app.get("/m/{job_id}/{token}/rapport", response_class=HTMLResponse)
def rapport(job_id: str, token: str) -> FileResponse:
    job = _authenticate(job_id, token)
    if job.state is not JobState.DONE or not job.report_path:
        raise HTTPException(status_code=409, detail="Analyse pas encore terminée.")
    return FileResponse(job.report_path, media_type="text/html")


@app.get("/m/{job_id}/{token}/joueurs", response_class=HTMLResponse)
def nommer_joueurs(job_id: str, token: str) -> str:
    job = _authenticate(job_id, token)
    if job.state is not JobState.DONE or not job.stats:
        raise HTTPException(status_code=409, detail="Analyse pas encore terminée.")
    return pages.naming_page(job, pages.nommables(job.stats))


@app.post("/m/{job_id}/{token}/joueurs")
async def enregistrer_noms(job_id: str, token: str, request: Request) -> RedirectResponse:
    """Enregistre les noms puis régénère le rapport.

    Le rapport est un fichier produit une fois par le worker : sans
    régénération, les noms n'apparaîtraient que sur ce formulaire. La
    régénération est peu coûteuse — elle ne relit que les statistiques, jamais
    la vidéo.
    """
    from datetime import date

    from football_analysis.report import ReportMeta, write as write_report

    job = _authenticate(job_id, token)
    if job.state is not JobState.DONE or not job.report_path:
        raise HTTPException(status_code=409, detail="Analyse pas encore terminée.")

    formulaire = await request.form()
    noms = {
        cle[len("nom_"):]: str(valeur)
        for cle, valeur in formulaire.items()
        if cle.startswith("nom_")
    }
    store.set_player_names(job.id, noms)

    chemin = Path(job.report_path)
    stats_path = chemin.with_name("analyse.json")
    if stats_path.exists():
        write_report(
            stats_path, chemin,
            ReportMeta(match_name=job.match_name, played_on=date.today()),
            radar_png=chemin.with_name("analyse_radar.png"),
            names=store.get(job.id).player_names,
        )
    return RedirectResponse(f"{job.public_url}/rapport", status_code=303)


@app.get("/m/{job_id}/{token}/video")
def video_annotee(job_id: str, token: str) -> FileResponse:
    job = _authenticate(job_id, token)
    if job.state is not JobState.DONE or not job.video_output_path:
        raise HTTPException(status_code=409, detail="Analyse pas encore terminée.")
    return FileResponse(
        job.video_output_path,
        media_type="video/mp4",
        filename=f"{job.match_name}.mp4",
    )


@app.get("/sante")
def sante() -> JSONResponse:
    """Supervision : file, matchs bloqués, stockage occupé.

    `bloques` est le compteur à surveiller : s'il ne redescend pas, le worker
    est mort et personne ne reprend la file. La réponse passe en 503 dans ce
    cas, pour qu'une sonde externe le détecte sans lire le corps.
    """
    from .storage import RESERVE_DISQUE

    bloques = store.stale_count(STALE_SECONDS)
    libre = storage.free_bytes()
    sature = libre < RESERVE_DISQUE
    corps = {
        "jobs": store.counts_by_state(),
        "en_attente": store.pending_count(),
        "bloques": bloques,
        "stockage_go": round(storage.usage_bytes() / 1024**3, 2),
        "disque_libre_go": round(libre / 1024**3, 2),
        "sature": sature,
    }
    return JSONResponse(corps, status_code=503 if (bloques or sature) else 200)
