"""Récupérer une vidéo depuis un lien plutôt qu'un téléversement.

Un match de 90 minutes pèse plusieurs gigaoctets. Le téléverser depuis une
connexion mobile — la norme du marché visé — prend des heures et échoue à la
moindre coupure, sans reprise possible. Or beaucoup de clubs publient déjà
leurs matchs sur une plateforme vidéo : un lien s'envoie en une seconde, et le
téléchargement se fait depuis nos serveurs, sur une liaison stable.

C'est le service qui télécharge, donc c'est le service qu'on peut faire
pointer n'importe où : la validation de l'adresse ci-dessous n'est pas une
formalité mais la seule barrière entre un lien saisi par un inconnu et le
réseau interne de la machine.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("fetch")

SCHEMAS = {"http", "https"}
TAILLE_MAX_GO = 8


class LienRefuse(ValueError):
    """Le lien ne sera pas récupéré ; le message est destiné au club."""


def _est_prive(hote: str) -> bool:
    """L'adresse pointe-t-elle vers le réseau interne ?

    Un lien vers 169.254.169.254 exposerait les identifiants d'accès de la
    machine chez la plupart des hébergeurs. La résolution est faite ici, avant
    tout téléchargement.
    """
    try:
        infos = socket.getaddrinfo(hote, None)
    except socket.gaierror:
        return True                      # inconnu : on refuse plutôt que d'essayer
    for info in infos:
        adresse = ipaddress.ip_address(info[4][0])
        if (
            adresse.is_private or adresse.is_loopback or adresse.is_link_local
            or adresse.is_reserved or adresse.is_multicast
        ):
            return True
    return False


def valider(lien: str) -> str:
    """Rend le lien nettoyé, ou lève `LienRefuse`."""
    lien = lien.strip()
    if not lien:
        raise LienRefuse("Aucun lien fourni.")

    analyse = urlparse(lien)
    if analyse.scheme not in SCHEMAS:
        raise LienRefuse(
            "Le lien doit commencer par http:// ou https://."
        )
    if not analyse.hostname:
        raise LienRefuse("Lien incomplet : l'adresse du site manque.")
    if _est_prive(analyse.hostname):
        raise LienRefuse("Ce lien n'est pas accessible depuis nos serveurs.")
    return lien


def telecharger(lien: str, destination: Path, timeout: float = 3600) -> Path:
    """Télécharge la vidéo. Lève `LienRefuse` avec un message pour le club."""
    lien = valider(lien)
    destination.parent.mkdir(parents=True, exist_ok=True)
    modele = str(destination.with_suffix(".%(ext)s"))

    commande = [
        "yt-dlp", "--no-playlist", "--no-warnings", "--quiet",
        # Une seule vidéo, jamais une chaîne entière : un lien de chaîne
        # remplirait le disque en silence.
        "--max-downloads", "1",
        "--max-filesize", f"{TAILLE_MAX_GO}G",
        "-f", "bv*[height<=1080][ext=mp4]/b[height<=1080]/b",
        "-o", modele, lien,
    ]
    try:
        resultat = subprocess.run(
            commande, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        raise LienRefuse("Le téléchargement par lien est indisponible.") from None
    except subprocess.TimeoutExpired:
        raise LienRefuse(
            "Le téléchargement a pris trop de temps. La vidéo est peut-être "
            "trop longue."
        ) from None

    fichiers = sorted(destination.parent.glob(f"{destination.stem}.*"))
    fichiers = [f for f in fichiers if f.suffix.lower() != ".part"]
    if resultat.returncode != 0 or not fichiers:
        log.warning("échec du téléchargement de %s : %s", lien, resultat.stderr[-300:])
        raise LienRefuse(
            "La vidéo n'a pas pu être récupérée. Vérifiez que le lien est "
            "public et qu'il pointe bien vers une vidéo."
        )
    return fichiers[0]
