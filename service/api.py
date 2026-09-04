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
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse,
)

from .jobs import Club, Job, JobState, JobStore
from . import logs
from .fetch import LienRefuse, valider as valider_lien
from .notify import looks_like_email
from . import season as saison_mod
from .settings import (
    ADMIN_TOKEN, DATA_ROOT, FILE_MAX, FILE_MAX_PAR_CLUB, LONGUEUR_CLUB, LONGUEUR_CONTACT,
    LONGUEUR_LIEN, LONGUEUR_MATCH, STALE_SECONDS,
)
from .storage import Storage, UploadRefuse
from .web import pages

# Posé à l'import : uvicorn crée ses journaux avant de charger l'application,
# et un filtre installé plus tard laisserait passer les premières requêtes.
logs.install()

app = FastAPI(title="Analyse de match", docs_url=None, redoc_url=None)


# Enregistré sur l'exception de Starlette et non celle de FastAPI : une route
# inexistante lève la première, que la seconde ne couvre pas. Sans cela,
# `/adresse-inventee` renvoyait encore du JSON brut.
@app.exception_handler(StarletteHTTPException)
def erreur_lisible(request: Request, exc: StarletteHTTPException):
    """Les erreurs s'adressent à un club, pas à un client d'API.

    Une adresse mal recopiée renvoyait la réponse JSON brute de FastAPI, ce
    qui donne l'impression d'un service en panne plutôt que d'un lien erroné.
    """
    return HTMLResponse(
        pages.error_page(exc.status_code, str(exc.detail)),
        status_code=exc.status_code,
    )


@app.get("/admin/{token}", response_class=HTMLResponse)
def exploitation(token: str) -> str:
    """Vue d'exploitation, protégée par un jeton distinct de ceux des clubs.

    Sans jeton configuré, la page n'existe pas : mieux vaut pas
    d'administration qu'une administration ouverte à tous.
    """
    import secrets

    if not ADMIN_TOKEN or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=404, detail="Page introuvable.")
    return pages.admin_page(store.clubs(), store.recent(), sante().body and _sante_corps())


def _sante_corps() -> dict:
    from .storage import RESERVE_DISQUE

    libre = storage.free_bytes()
    return {
        "en_attente": store.pending_count(),
        "bloques": store.stale_count(STALE_SECONDS),
        "disque_libre_go": round(libre / 1024**3, 2),
        "sature": libre < RESERVE_DISQUE,
    }


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots() -> str:
    """Aucune page ne doit être indexée : toutes portent un jeton d'accès."""
    return "User-agent: *\nDisallow: /\n"
store = JobStore(DATA_ROOT / "jobs.db")
storage = Storage(DATA_ROOT / "videos")


def _etat_service() -> dict:
    """État de la file et du disque.

    Partagé par `/sante` et la page d'exploitation : deux vues de la même
    chose, qui divergeraient si chacune recalculait la sienne.
    """
    from .storage import RESERVE_DISQUE

    libre = storage.free_bytes()
    return {
        "jobs": store.counts_by_state(),
        "en_attente": store.pending_count(),
        "bloques": store.stale_count(STALE_SECONDS),
        "stockage_go": round(storage.usage_bytes() / 1024**3, 2),
        "disque_libre_go": round(libre / 1024**3, 2),
        "sature": libre < RESERVE_DISQUE,
    }


def _piece_jointe(nom: str) -> str:
    """En-tête de téléchargement portant un nom de fichier accentué.

    Les en-têtes HTTP sont en latin-1 : « Valmont – Beaupré » y lève une
    erreur d'encodage et le téléchargement échoue. On donne donc une version
    ASCII en repli et le vrai nom en RFC 5987, que tous les navigateurs
    récents préfèrent.
    """
    from urllib.parse import quote

    ascii_nom = "".join(c if c.isalnum() or c in " -_." else "_" for c in nom)
    ascii_nom = ascii_nom.encode("ascii", "ignore").decode().strip() or "releve.csv"
    return (
        f'attachment; filename="{ascii_nom}"; '
        f"filename*=UTF-8''{quote(nom, safe='')}"
    )


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
    played_on: str = Form(""),
    source_url: str = Form(""),
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

    # Couper côté serveur : le maxlength du formulaire n'engage que les
    # navigateurs, et un nom démesuré serait stocké puis renvoyé sur chaque
    # page du club.
    club = club.strip()[:LONGUEUR_CLUB]
    match_name = match_name.strip()[:LONGUEUR_MATCH]
    contact = contact.strip()[:LONGUEUR_CONTACT]
    source_url = source_url.strip()[:LONGUEUR_LIEN]

    if store.pending_count() >= FILE_MAX:
        return HTMLResponse(
            pages.upload_form(
                erreur="Trop de matchs sont en attente. Réessayez dans quelques heures.",
                club=espace,
            ), 503,
        )
    if espace is not None and store.pending_for_club(espace.id) >= FILE_MAX_PAR_CLUB:
        return HTMLResponse(
            pages.upload_form(
                erreur=f"Vous avez déjà {FILE_MAX_PAR_CLUB} matchs en cours "
                       "d'analyse. Attendez qu'ils se terminent.",
                club=espace,
            ), 429,
        )

    # Le lien est validé tout de suite : refuser après un téléversement de
    # plusieurs gigaoctets serait cruel, et le club ne saurait pas pourquoi.
    if source_url:
        try:
            source_url = valider_lien(source_url)
        except LienRefuse as refus:
            return HTMLResponse(
                pages.upload_form(erreur=str(refus), club=espace), 400
            )

    fichier_fourni = video is not None and bool(video.filename)
    if not fichier_fourni and not source_url:
        return HTMLResponse(
            pages.upload_form(
                erreur="Indiquez un lien vers la vidéo, ou choisissez un fichier.",
                club=espace,
            ), 400,
        )

    # Une adresse invalide est refusée plutôt qu'ignorée : le club croirait
    # être prévenu et attendrait un message qui ne viendrait jamais.
    if contact and not looks_like_email(contact):
        return HTMLResponse(
            pages.upload_form(erreur=f"Adresse e-mail invalide : {contact}", club=espace),
            400,
        )

    # Une date illisible est ignorée plutôt que refusée : elle est facultative,
    # et bloquer un dépôt de 2 Go pour un champ accessoire serait absurde.
    from datetime import date as _date

    played_on = played_on.strip()
    try:
        _date.fromisoformat(played_on) if played_on else None
    except ValueError:
        played_on = ""

    if espace is None:
        espace = store.create_club(club.strip())

    job = store.create(
        espace.name, match_name.strip(), video_path="",
        contact=contact, club_id=espace.id, played_on=played_on,
        source_url="" if fichier_fourni else source_url,
    )
    if fichier_fourni:
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
    from football_analysis.report import ReportMeta, write as write_report

    from .worker import _date_du_match

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
            ReportMeta(match_name=job.match_name, played_on=_date_du_match(job)),
            radar_png=chemin.with_name("analyse_radar.png"),
            names=store.get(job.id).player_names,
        )
    return RedirectResponse(f"{job.public_url}/rapport", status_code=303)


@app.get("/m/{job_id}/{token}/releve.csv")
def releve_csv(job_id: str, token: str) -> PlainTextResponse:
    """Relevé des joueurs en tableur, pour les clubs qui tiennent leurs
    propres statistiques."""
    from .export import players_csv

    job = _authenticate(job_id, token)
    if job.state is not JobState.DONE or not job.stats:
        raise HTTPException(status_code=409, detail="Analyse pas encore terminée.")

    return PlainTextResponse(
        players_csv(job.stats, job.player_names),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": _piece_jointe(f"{job.match_name}.csv")},
    )


@app.post("/m/{job_id}/{token}/supprimer")
def supprimer(job_id: str, token: str) -> RedirectResponse:
    """Efface un match et tous ses fichiers.

    Un club se trompe de vidéo, ou souhaite retirer un match : sans cette
    possibilité il faudrait nous écrire, et nous donner accès à sa base.
    """
    job = _authenticate(job_id, token)
    club = store.get_club(job.club_id) if job.club_id else None
    storage.purge(job.id)
    store.delete(job.id)
    return RedirectResponse(club.public_url if club else "/", status_code=303)


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
    corps = _etat_service()
    degrade = corps["bloques"] or corps["sature"]
    return JSONResponse(corps, status_code=503 if degrade else 200)
