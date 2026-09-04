"""Journalisation sans fuite de jetons.

Les adresses portent le jeton d'accès : `/m/{id}/{jeton}`. Le journal d'accès
d'uvicorn les écrit telles quelles, si bien que l'hébergeur, l'agrégateur de
journaux et tout administrateur voient les jetons de tous les clubs — sans
jamais avoir eu à les demander.

Couper le journal d'accès ferait perdre toute observabilité. On le garde donc,
en masquant le seul segment sensible.
"""
from __future__ import annotations

import logging
import re

# Deuxième segment des adresses de match et d'espace : c'est le jeton.
JETON = re.compile(r"(/[mc]/[^/\s]+/)[^/\s?]+")
MASQUE = r"\1***"


class MasqueJetons(logging.Filter):
    """Remplace tout jeton par des étoiles dans les messages journalisés."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                JETON.sub(MASQUE, a) if isinstance(a, str) else a
                for a in record.args
            )
        if isinstance(record.msg, str):
            record.msg = JETON.sub(MASQUE, record.msg)
        return True


def install() -> None:
    """Pose le filtre sur les journaux d'uvicorn et de l'application.

    À appeler au démarrage de l'API comme du worker : le worker écrit lui
    aussi des adresses de match dans ses messages.
    """
    filtre = MasqueJetons()
    for nom in ("uvicorn.access", "uvicorn.error", "worker", "notify", "fetch"):
        logging.getLogger(nom).addFilter(filtre)
    logging.getLogger().addFilter(filtre)
