"""Prépare un jeu de démonstration : un match déjà analysé, visible dans l'app.

À montrer devant un club, on ne lance pas une analyse : elle dure des dizaines
de minutes et la démonstration se transforme en attente. On sert donc
l'application avec un match déjà terminé, exactement dans l'état où le worker
l'aurait laissé.

Le script ne fabrique aucun chiffre : il reprend la sortie d'un vrai passage de
`football_analysis.cli` et la range là où l'API la cherche.

    .venv/bin/python -m football_analysis.cli input_video/extrait.mp4 \\
        -o output_video/demo_match.mp4 --match-name "..."
    .venv/bin/python scripts/preparer_demo.py --video output_video/demo_match.mp4

Puis, dans le terminal que l'on gardera ouvert pendant le rendez-vous :

    FA_DATA_ROOT=data/demo .venv/bin/uvicorn service.api:app
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from football_analysis.report import ReportMeta, write as ecrire_rapport  # noqa: E402
from service.jobs import JobState, JobStore  # noqa: E402
from service.storage import Storage  # noqa: E402


def _date_ou_none(texte: str) -> date | None:
    if not texte:
        return None
    try:
        return date.fromisoformat(texte)
    except ValueError:
        raise SystemExit(f"date invalide : {texte!r} (attendu AAAA-MM-JJ)")


def preparer(
    video: Path, racine: Path, club: str, match: str, jouee_le: str
) -> tuple[str, str]:
    stats_json = video.with_suffix(".json")
    if not video.exists():
        raise SystemExit(f"vidéo analysée introuvable : {video}")
    if not stats_json.exists():
        raise SystemExit(
            f"statistiques introuvables : {stats_json}\n"
            "Le passage de la CLI a-t-il été jusqu'au bout ?"
        )

    store = JobStore(racine / "jobs.db")
    storage = Storage(racine / "videos")

    espace = store.create_club(club)
    job = store.create(
        club=club,
        club_id=espace.id,
        match_name=match,
        video_path="",
        played_on=jouee_le,
    )

    dossier = storage.job_dir(job.id)
    shutil.copy2(video, dossier / "analyse.mp4")

    radar = video.with_name(video.stem + "_radar.png")
    rapport = ecrire_rapport(
        stats_json,
        dossier / "rapport.html",
        # `demo=False` : ces chiffres sont mesurés sur un vrai extrait. Le
        # bandeau « données de démonstration » signalerait des valeurs
        # fabriquées et décrédibiliserait à tort un rapport exact.
        ReportMeta(match_name=match, played_on=_date_ou_none(jouee_le)),
        radar_png=radar if radar.exists() else None,
    )

    store.update(
        job.id,
        state=JobState.DONE,
        progress=1.0,
        stats=json.loads(stats_json.read_text()),
        report_path=str(rapport),
        video_output_path=str(dossier / "analyse.mp4"),
    )
    return job.public_url, espace.public_url


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=RACINE / "output_video/demo_match.mp4",
                   help="vidéo annotée produite par la CLI ; le .json voisin est lu")
    p.add_argument("--data-root", type=Path, default=RACINE / "data/demo",
                   help="racine des données servie ensuite par FA_DATA_ROOT")
    p.add_argument("--club", default="Club de démonstration")
    p.add_argument("--match", default="Match de démonstration")
    p.add_argument("--played-on", default="", help="date de la rencontre, AAAA-MM-JJ")
    a = p.parse_args()

    lien_match, lien_club = preparer(
        a.video.resolve(), a.data_root.resolve(), a.club, a.match, a.played_on
    )

    print("Jeu de démonstration prêt.\n")
    print("  1. Servir l'application :")
    print(f"       FA_DATA_ROOT={a.data_root} .venv/bin/uvicorn service.api:app\n")
    print("  2. Ouvrir dans le navigateur :")
    print("       http://127.0.0.1:8000/                    la page d'accueil")
    print(f"       http://127.0.0.1:8000{lien_match}    le rapport du match")
    print(f"       http://127.0.0.1:8000{lien_club}     l'espace du club\n")
    print("  Ces adresses portent un jeton : elles ne se devinent pas, et")
    print("  changent à chaque exécution de ce script.")


if __name__ == "__main__":
    main()
