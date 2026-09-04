"""Prévenir le club quand son rapport est prêt.

Le lien remis au dépôt est le seul accès au rapport, et il ne s'affiche qu'une
fois. Sans notification, un club doit le conserver puis y revenir de lui-même
sans savoir quand — ce qui revient à lui demander de surveiller une page
pendant une heure.

Le message contient donc ce lien privé. Une adresse mal saisie l'envoie à un
inconnu : c'est pourquoi le contact est facultatif, jamais deviné, et que le
message ne mentionne ni le nom du club ni celui de l'adversaire au-delà de ce
que le titre du match contient déjà.
"""
from __future__ import annotations

import logging
import os
import re
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol

log = logging.getLogger("notify")

# Volontairement permissif : refuser une adresse valide est plus coûteux que
# d'en accepter une fausse, qui échouera simplement à l'envoi.
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


def looks_like_email(contact: str) -> bool:
    return bool(EMAIL.match(contact.strip()))


@dataclass
class Message:
    destinataire: str
    sujet: str
    corps: str


class Notifier(Protocol):
    def send(self, message: Message) -> bool:
        """Rend True si le message est parti. Ne lève jamais."""


class LogNotifier:
    """Défaut : trace sans envoyer.

    Permet de faire tourner le service sans configuration SMTP — un club
    récupère alors son rapport par le lien, comme avant.
    """

    def send(self, message: Message) -> bool:
        log.info("notification non envoyée (aucun expéditeur configuré) : %s",
                 message.destinataire)
        return False


class SmtpNotifier:
    def __init__(self, host: str, port: int, user: str, password: str,
                 expediteur: str, timeout: float = 20.0) -> None:
        self.host, self.port = host, port
        self.user, self.password = user, password
        self.expediteur = expediteur
        self.timeout = timeout

    def send(self, message: Message) -> bool:
        mail = EmailMessage()
        mail["From"] = self.expediteur
        mail["To"] = message.destinataire
        mail["Subject"] = message.sujet
        mail.set_content(message.corps)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as smtp:
                smtp.starttls()
                smtp.login(self.user, self.password)
                smtp.send_message(mail)
        except Exception:                              # noqa: BLE001
            # Un envoi raté ne doit pas faire échouer une analyse réussie :
            # le rapport existe, le lien fonctionne, seul l'avis manque.
            log.exception("envoi impossible vers %s", message.destinataire)
            return False
        log.info("notification envoyée à %s", message.destinataire)
        return True


def from_environment() -> Notifier:
    """Construit l'expéditeur depuis l'environnement, ou renonce."""
    host = os.environ.get("FA_SMTP_HOST")
    expediteur = os.environ.get("FA_SMTP_FROM")
    if not host or not expediteur:
        return LogNotifier()
    return SmtpNotifier(
        host=host,
        port=int(os.environ.get("FA_SMTP_PORT", "587")),
        user=os.environ.get("FA_SMTP_USER", ""),
        password=os.environ.get("FA_SMTP_PASSWORD", ""),
        expediteur=expediteur,
    )


def report_ready(
    match_name: str, lien: str, base_url: str = "", espace: str = ""
) -> Message:
    espace_ligne = (
        f"\nTous vos matchs : {base_url}{espace}\n"
        "C'est cette adresse à conserver sur la durée.\n" if espace else ""
    )
    return Message(
        destinataire="",
        sujet=f"Analyse terminée : {match_name}",
        corps=(
            f"L'analyse de {match_name} est terminée.\n\n"
            f"Votre rapport : {base_url}{lien}\n"
            f"{espace_ligne}\n"
            "Ces liens sont personnels : toute personne qui les obtient accède "
            "aux rapports. Ils restent valables trois mois.\n"
        ),
    )


def analysis_failed(match_name: str, raison: str, lien: str, base_url: str = "") -> Message:
    return Message(
        destinataire="",
        sujet=f"Analyse impossible : {match_name}",
        corps=(
            f"L'analyse de {match_name} n'a pas abouti.\n\n"
            f"{raison}\n\n"
            f"Détails : {base_url}{lien}\n"
        ),
    )
