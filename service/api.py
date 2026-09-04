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

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .jobs import Job, JobState, JobStore
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


@app.post("/matches", response_class=HTMLResponse)
async def deposer(
    request: Request,
    club: str = Form(...),
    match_name: str = Form(...),
    video: UploadFile = None,
) -> HTMLResponse:
    if video is None or not video.filename:
        return HTMLResponse(pages.upload_form(erreur="Aucune vidéo sélectionnée."), 400)

    job = store.create(club.strip(), match_name.strip(), video_path="")
    try:
        chemin = storage.save_upload(job.id, video.filename, video.file)
    except UploadRefuse as refus:
        store.update(job.id, state=JobState.FAILED, error=str(refus))
        return HTMLResponse(pages.upload_form(erreur=str(refus)), 400)

    store.update(job.id, video_path=str(chemin))
    job.video_path = str(chemin)
    return HTMLResponse(pages.upload_done(job, store.pending_count()), 201)


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
    bloques = store.stale_count(STALE_SECONDS)
    corps = {
        "jobs": store.counts_by_state(),
        "en_attente": store.pending_count(),
        "bloques": bloques,
        "stockage_go": round(storage.usage_bytes() / 1024**3, 2),
    }
    return JSONResponse(corps, status_code=503 if bloques else 200)
