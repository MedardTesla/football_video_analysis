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
class Club:
    """Un club et son espace privé.

    Identifié par un jeton, jamais par son nom : deux clubs homonymes — ce qui
    arrive souvent entre catégories d'un même village — verraient sinon les
    matchs l'un de l'autre.
    """

    id: str
    token: str
    name: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def public_url(self) -> str:
        return f"/c/{self.id}/{self.token}"


@dataclass
class Job:
    id: str
    token: str
    club: str
    match_name: str
    video_path: str
    club_id: str = ""
    contact: str = ""
    # Date de la rencontre, saisie par le club. Distincte de la date
    # d'analyse : un club dépose souvent un match joué des semaines plus tôt,
    # et afficher la seconde à la place de la première est un contresens.
    played_on: str = ""
    # Lien fourni au lieu d'un téléversement. La vidéo est récupérée par le
    # worker : un match de plusieurs gigaoctets ne se téléverse pas depuis une
    # connexion mobile, alors qu'un lien s'envoie en une seconde.
    source_url: str = ""
    # Laquelle des deux équipes du rapport est celle du club. Les libellés
    # « équipe A » et « équipe B » sont des étiquettes de regroupement, pas
    # des identités : rien ne garantit que l'équipe A d'un match soit la même
    # que celle du match suivant. Sans cette désignation, aucune comparaison
    # de saison n'a de sens.
    our_team: int | None = None
    # Numéro de piste -> nom donné par le club. Les numéros sont attribués par
    # le traqueur et ne veulent rien dire pour un entraîneur.
    player_names: dict[str, str] = field(default_factory=dict)
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
CREATE TABLE IF NOT EXISTS clubs (
    id TEXT PRIMARY KEY,
    token TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    token TEXT NOT NULL,
    club TEXT NOT NULL,
    match_name TEXT NOT NULL,
    video_path TEXT NOT NULL,
    club_id TEXT NOT NULL DEFAULT '',
    contact TEXT NOT NULL DEFAULT '',
    played_on TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    our_team INTEGER,
    player_names TEXT,
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
"""

# Créés après la migration : un index porte sur des colonnes qui peuvent
# manquer à une base créée par une version antérieure.
INDEX = """
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, created_at);
CREATE INDEX IF NOT EXISTS jobs_club ON jobs(club_id, created_at);
"""


class JobStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            self._migrate(db)
            db.executescript(INDEX)

    @staticmethod
    def _migrate(db: sqlite3.Connection) -> None:
        """Ajoute les colonnes absentes des bases créées par une version
        antérieure. SQLite ne sait pas ajouter une colonne « si absente »."""
        existantes = {r["name"] for r in db.execute("PRAGMA table_info(jobs)")}
        if "attempts" not in existantes:
            db.execute("ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        if "contact" not in existantes:
            db.execute("ALTER TABLE jobs ADD COLUMN contact TEXT NOT NULL DEFAULT ''")
        if "club_id" not in existantes:
            db.execute("ALTER TABLE jobs ADD COLUMN club_id TEXT NOT NULL DEFAULT ''")
        if "our_team" not in existantes:
            db.execute("ALTER TABLE jobs ADD COLUMN our_team INTEGER")
        if "player_names" not in existantes:
            db.execute("ALTER TABLE jobs ADD COLUMN player_names TEXT")
        if "played_on" not in existantes:
            db.execute("ALTER TABLE jobs ADD COLUMN played_on TEXT NOT NULL DEFAULT ''")
        if "source_url" not in existantes:
            db.execute(
                "ALTER TABLE jobs ADD COLUMN source_url TEXT NOT NULL DEFAULT ''"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        # Le worker écrit pendant que l'API lit : sans WAL, les lecteurs
        # bloquent l'écrivain et la page d'état se fige pendant le traitement.
        db.execute("PRAGMA journal_mode=WAL")
        return db

    # --- clubs ---------------------------------------------------------

    def create_club(self, name: str) -> Club:
        club = Club(id=secrets.token_hex(8), token=secrets.token_urlsafe(24), name=name)
        with self._connect() as db:
            db.execute(
                "INSERT INTO clubs (id, token, name, created_at) VALUES (?,?,?,?)",
                (club.id, club.token, club.name, club.created_at),
            )
        return club

    def get_club(self, club_id: str) -> Club | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM clubs WHERE id = ?", (club_id,)).fetchone()
        return Club(**dict(row)) if row else None

    def authenticate_club(self, club_id: str, token: str) -> Club | None:
        club = self.get_club(club_id)
        if club is None or not secrets.compare_digest(club.token, token):
            return None
        return club

    def matches_of_club(self, club_id: str, limit: int = 200) -> list[Job]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE club_id = ? ORDER BY created_at DESC LIMIT ?",
                (club_id, limit),
            ).fetchall()
        return [self._to_job(r) for r in rows]

    # --- matchs --------------------------------------------------------

    def create(
        self, club: str, match_name: str, video_path: str, contact: str = "",
        club_id: str = "", played_on: str = "", source_url: str = "",
    ) -> Job:
        job = Job(
            id=secrets.token_hex(8),
            token=secrets.token_urlsafe(24),
            club=club,
            match_name=match_name,
            video_path=str(video_path),
            contact=contact,
            club_id=club_id,
            played_on=played_on,
            source_url=source_url,
        )
        with self._connect() as db:
            db.execute(
                "INSERT INTO jobs (id, token, club, match_name, video_path, club_id,"
                " contact, played_on, source_url, state, progress, created_at,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job.id, job.token, job.club, job.match_name, job.video_path,
                 job.club_id, job.contact, job.played_on, job.source_url,
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

    def pending_for_club(self, club_id: str) -> int:
        """Matchs d'un club encore à traiter ou en cours."""
        if not club_id:
            return 0
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) n FROM jobs WHERE club_id = ? AND state IN (?, ?)",
                (club_id, JobState.QUEUED.value, JobState.PROCESSING.value),
            ).fetchone()
        return int(row["n"])

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

    def set_player_names(self, job_id: str, names: dict[str, str]) -> None:
        """Enregistre les noms saisis par le club, les vides étant effacés."""
        propres = {
            str(k): v.strip()[:60] for k, v in names.items() if v and v.strip()
        }
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET player_names = ?, updated_at = ? WHERE id = ?",
                (json.dumps(propres, ensure_ascii=False),
                 datetime.now(timezone.utc).isoformat(), job_id),
            )

    def set_our_team(self, job_id: str, team: int | None) -> None:
        """Désigne l'équipe du club dans ce match, ou l'efface."""
        if team is not None and team not in (0, 1):
            raise ValueError(f"équipe invalide : {team}")
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET our_team = ?, updated_at = ? WHERE id = ?",
                (team, datetime.now(timezone.utc).isoformat(), job_id),
            )

    def delete(self, job_id: str) -> None:
        """Efface un match de la base. Les fichiers sont purgés à part."""
        with self._connect() as db:
            db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def recent(self, limit: int = 100) -> list[Job]:
        """Derniers matchs, tous clubs confondus. Pour l'exploitation."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._to_job(r) for r in rows]

    def clubs(self, limit: int = 200) -> list[Club]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM clubs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Club(**dict(r)) for r in rows]

    def all_ids(self) -> set[str]:
        """Identifiants de tous les matchs connus, pour repérer les orphelins."""
        with self._connect() as db:
            return {r["id"] for r in db.execute("SELECT id FROM jobs")}

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
        data["player_names"] = (
            json.loads(data["player_names"]) if data.get("player_names") else {}
        )
        return Job(**data)
