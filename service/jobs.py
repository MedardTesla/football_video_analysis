"""Cycle de vie d'une analyse de match.

Un club dépose une vidéo, la reprend plus tard. Entre les deux, le traitement
dure des dizaines de minutes sur GPU : tout est donc asynchrone, et l'état doit
survivre au redémarrage du serveur comme du worker.

Le stockage est en SQLite. Un club représente quelques matchs par mois ; une
base serveur ne se justifiera qu'à plusieurs centaines de clubs, et SQLite se
migre sans réécrire ce module.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class JobState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (JobState.DONE, JobState.FAILED)


@dataclass
class Job:
    id: str
    token: str
    club: str
    match_name: str
    video_path: str
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    error: str | None = None
    attempts: int = 0
    stats: dict | None = None
    report_path: str | None = None
    video_output_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def public_url(self) -> str:
        """Le lien remis au club.

        Le jeton fait office d'authentification : un club n'a pas de compte à
        créer, il reçoit une adresse impossible à deviner. Un identifiant seul
        serait énumérable et exposerait les matchs des autres clubs.
        """
        return f"/m/{self.id}/{self.token}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    token TEXT NOT NULL,
    club TEXT NOT NULL,
    match_name TEXT NOT NULL,
    video_path TEXT NOT NULL,
    state TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    stats TEXT,
    report_path TEXT,
    video_output_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, created_at);
"""


class JobStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            self._migrate(db)

    @staticmethod
    def _migrate(db: sqlite3.Connection) -> None:
        """Ajoute les colonnes absentes des bases créées par une version
        antérieure. SQLite ne sait pas ajouter une colonne « si absente »."""
        existantes = {r["name"] for r in db.execute("PRAGMA table_info(jobs)")}
        if "attempts" not in existantes:
            db.execute("ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        # Le worker écrit pendant que l'API lit : sans WAL, les lecteurs
        # bloquent l'écrivain et la page d'état se fige pendant le traitement.
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def create(self, club: str, match_name: str, video_path: str) -> Job:
        job = Job(
            id=secrets.token_hex(8),
            token=secrets.token_urlsafe(24),
            club=club,
            match_name=match_name,
            video_path=str(video_path),
        )
        with self._connect() as db:
            db.execute(
                "INSERT INTO jobs (id, token, club, match_name, video_path, state,"
                " progress, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (job.id, job.token, job.club, job.match_name, job.video_path,
                 job.state.value, job.progress, job.created_at, job.updated_at),
            )
        return job

    def get(self, job_id: str) -> Job | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._to_job(row) if row else None

    def authenticate(self, job_id: str, token: str) -> Job | None:
        """Rend le job seulement si le jeton correspond.

        `secrets.compare_digest` plutôt que `==` : la comparaison naïve sort au
        premier caractère différent, ce qui laisse deviner le jeton caractère
        par caractère en mesurant le temps de réponse.
        """
        job = self.get(job_id)
        if job is None or not secrets.compare_digest(job.token, token):
            return None
        return job

    def claim_next(self) -> Job | None:
        """Prend le prochain job en attente, en le marquant aussitôt.

        La mise à jour conditionnée sur `state='queued'` rend l'opération
        atomique : deux workers lancés en parallèle ne peuvent pas traiter le
        même match deux fois.
        """
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY created_at LIMIT 1",
                (JobState.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            now = datetime.now(timezone.utc).isoformat()
            changed = db.execute(
                "UPDATE jobs SET state = ?, updated_at = ?, attempts = attempts + 1"
                " WHERE id = ? AND state = ?",
                (JobState.PROCESSING.value, now, row["id"], JobState.QUEUED.value),
            ).rowcount
        if changed == 0:
            return None
        job = self._to_job(row)
        job.state, job.updated_at = JobState.PROCESSING, now
        job.attempts += 1
        return job

    def heartbeat(self, job_id: str) -> None:
        """Signale que le worker est toujours vivant sur ce match.

        Sans ce battement, un match long serait considéré abandonné et repris
        par un autre worker pendant que le premier travaille encore.
        """
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), job_id),
            )

    def reclaim_stale(self, timeout_seconds: float, max_attempts: int = 3) -> list[str]:
        """Remet en file les matchs abandonnés par un worker mort.

        Sans cela, une coupure de courant laisse un match en « en cours »
        indéfiniment, et le club attend un rapport qui ne viendra jamais.

        `max_attempts` protège d'un tout autre risque : un fichier qui fait
        planter le worker serait repris à l'infini et bloquerait la file
        derrière lui. Au-delà, le match est déclaré en échec.
        """
        limite = datetime.now(timezone.utc).timestamp() - timeout_seconds
        repris: list[str] = []
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, attempts, updated_at FROM jobs WHERE state = ?",
                (JobState.PROCESSING.value,),
            ).fetchall()
            for row in rows:
                if datetime.fromisoformat(row["updated_at"]).timestamp() > limite:
                    continue
                now = datetime.now(timezone.utc).isoformat()
                if row["attempts"] >= max_attempts:
                    db.execute(
                        "UPDATE jobs SET state = ?, error = ?, updated_at = ? WHERE id = ?",
                        (JobState.FAILED.value,
                         "L'analyse a échoué à plusieurs reprises. "
                         "Nous avons été prévenus et revenons vers vous.",
                         now, row["id"]),
                    )
                else:
                    db.execute(
                        "UPDATE jobs SET state = ?, progress = 0, updated_at = ? WHERE id = ?",
                        (JobState.QUEUED.value, now, row["id"]),
                    )
                    repris.append(row["id"])
        return repris

    def update(self, job_id: str, **champs) -> None:
        if "stats" in champs and champs["stats"] is not None:
            champs["stats"] = json.dumps(champs["stats"])
        if isinstance(champs.get("state"), JobState):
            champs["state"] = champs["state"].value
        champs["updated_at"] = datetime.now(timezone.utc).isoformat()
        colonnes = ", ".join(f"{k} = ?" for k in champs)
        with self._connect() as db:
            db.execute(
                f"UPDATE jobs SET {colonnes} WHERE id = ?",
                (*champs.values(), job_id),
            )

    def pending_count(self) -> int:
        """Nombre de matchs en attente, sans en réserver aucun."""
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) n FROM jobs WHERE state = ?", (JobState.QUEUED.value,)
            ).fetchone()
        return int(row["n"])

    def list_for_club(self, club: str, limit: int = 50) -> list[Job]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE club = ? ORDER BY created_at DESC LIMIT ?",
                (club, limit),
            ).fetchall()
        return [self._to_job(r) for r in rows]

    def stale_count(self, timeout_seconds: float) -> int:
        """Matchs en cours sans nouvelle depuis trop longtemps.

        Remonté par la supervision : si ce compteur ne redescend pas, c'est
        que le worker est mort et que personne ne reprend la file.
        """
        limite = datetime.now(timezone.utc).timestamp() - timeout_seconds
        with self._connect() as db:
            rows = db.execute(
                "SELECT updated_at FROM jobs WHERE state = ?",
                (JobState.PROCESSING.value,),
            ).fetchall()
        return sum(
            1 for r in rows if datetime.fromisoformat(r["updated_at"]).timestamp() <= limite
        )

    def counts_by_state(self) -> dict[str, int]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT state, COUNT(*) n FROM jobs GROUP BY state"
            ).fetchall()
        return {r["state"]: r["n"] for r in rows}

    @staticmethod
    def _to_job(row: sqlite3.Row) -> Job:
        data = dict(row)
        data["state"] = JobState(data["state"])
        data["stats"] = json.loads(data["stats"]) if data["stats"] else None
        return Job(**data)
