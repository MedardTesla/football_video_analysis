"""Stockage des fichiers d'un match.

Interface volontairement étroite : déposer, situer, supprimer. Le passage à un
stockage objet (S3, R2) ne touchera que ce module — les vidéos de match pèsent
plusieurs gigaoctets et ne resteront pas longtemps sur un disque local.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import BinaryIO

# Formats acceptés. La liste est restrictive à dessein : un fichier refusé
# coûte un message d'erreur, un fichier accepté puis illisible coûte une place
# dans la file et vingt minutes de GPU.
EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
TAILLE_MAX = 8 * 1024**3      # 8 Go : environ 2 h en 1080p

# Espace libre à préserver quoi qu'il arrive. Le dépôt est ouvert sans compte :
# sans cette réserve, quelques envois simultanés suffisent à saturer le disque,
# et le service tombe pour tous les clubs — y compris ceux dont l'analyse est
# déjà en cours et dont le travail serait perdu.
RESERVE_DISQUE = 5 * 1024**3

# Un rapport reste consultable trois mois. Au-delà, un club l'a lu ou ne le
# lira pas, et la vidéo annotée pèse plus lourd que tout le reste.
RETENTION_JOURS = 90


class UploadRefuse(ValueError):
    """Le fichier ne sera pas accepté ; le message est destiné au club."""


class Storage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        d = self.root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_upload(self, job_id: str, filename: str, source: BinaryIO) -> Path:
        """Écrit la vidéo déposée par flux, sans la charger en mémoire."""
        suffixe = Path(filename).suffix.lower()
        if suffixe not in EXTENSIONS:
            raise UploadRefuse(
                f"Format {suffixe or 'inconnu'} non pris en charge. "
                f"Formats acceptés : {', '.join(sorted(EXTENSIONS))}."
            )

        libre = self.free_bytes()
        if libre < RESERVE_DISQUE:
            raise UploadRefuse(
                "Service momentanément saturé. Réessayez dans quelques heures."
            )

        cible = self.job_dir(job_id) / f"source{suffixe}"
        ecrits = 0
        with cible.open("wb") as sortie:
            while morceau := source.read(1024 * 1024):
                ecrits += len(morceau)
                if ecrits > TAILLE_MAX:
                    sortie.close()
                    cible.unlink(missing_ok=True)
                    raise UploadRefuse(
                        f"Vidéo trop volumineuse : maximum {TAILLE_MAX // 1024**3} Go."
                    )
                # Vérifié pendant l'écriture, pas seulement avant : la taille
                # annoncée par un client n'engage à rien, et plusieurs envois
                # simultanés se partagent le même disque.
                if ecrits % (256 * 1024**2) == 0 and self.free_bytes() < RESERVE_DISQUE:
                    sortie.close()
                    cible.unlink(missing_ok=True)
                    raise UploadRefuse(
                        "Service momentanément saturé. Réessayez dans quelques heures."
                    )
                sortie.write(morceau)

        if ecrits == 0:
            cible.unlink(missing_ok=True)
            raise UploadRefuse("Fichier vide.")
        return cible

    def purge(self, job_id: str) -> None:
        """Supprime la source après analyse.

        Les vidéos sont le poste de stockage dominant et n'ont plus d'usage
        une fois le rapport produit. Les conserver serait aussi un risque :
        ce sont les images du club, pas les nôtres.
        """
        shutil.rmtree(self.root / job_id, ignore_errors=True)

    def usage_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())

    def free_bytes(self) -> int:
        """Espace libre sur le disque qui porte le stockage."""
        return shutil.disk_usage(self.root).free

    def purge_older_than(self, days: int = RETENTION_JOURS) -> list[str]:
        """Supprime les dossiers de match plus vieux que `days`.

        Sans purge, la vidéo annotée de chaque match s'accumule indéfiniment.
        C'est le seul poste qui grossit sans limite, et le disque plein arrête
        le service entier — pas seulement les nouveaux dépôts.
        """
        limite = time.time() - days * 86400
        supprimes = []
        for dossier in self.root.iterdir():
            if not dossier.is_dir():
                continue
            recent = max(
                (f.stat().st_mtime for f in dossier.rglob("*") if f.is_file()),
                default=dossier.stat().st_mtime,
            )
            if recent < limite:
                shutil.rmtree(dossier, ignore_errors=True)
                supprimes.append(dossier.name)
        return supprimes
