"""Notification du club.

Le lien remis au dépôt est le seul accès au rapport. Sans avis, un club doit
le conserver puis y revenir sans savoir quand.
"""
from __future__ import annotations

import pytest

from service.notify import (
    LogNotifier, Message, SmtpNotifier, analysis_failed, looks_like_email,
    report_ready,
)


@pytest.mark.parametrize("adresse", [
    "entraineur@club.fr", "a.b+tag@sous.domaine.tg", "x@y.io",
])
def test_plausible_addresses_are_accepted(adresse):
    assert looks_like_email(adresse)


@pytest.mark.parametrize("adresse", [
    "", "pas-une-adresse", "manque@point", "@club.fr", "deux@@club.fr",
    "espace dans@club.fr",
])
def test_implausible_addresses_are_refused(adresse):
    assert not looks_like_email(adresse)


def test_the_ready_message_carries_the_private_link():
    m = report_ready("Valmont – Beaupré", "/m/abc/jeton", "https://analyse.example")
    assert "https://analyse.example/m/abc/jeton" in m.corps
    assert "Valmont – Beaupré" in m.sujet


def test_the_ready_message_warns_the_link_is_shareable():
    """Toute personne qui reçoit ce lien accède au rapport : le dire."""
    m = report_ready("Match", "/m/abc/jeton")
    assert "personnel" in m.corps
    assert "trois mois" in m.corps


def test_the_failure_message_states_the_reason():
    m = analysis_failed("Match", "La vidéo n'a pas pu être lue.", "/m/abc/j")
    assert "pas pu être lue" in m.corps
    assert "impossible" in m.sujet.lower()


def test_without_configuration_nothing_is_sent():
    """Le service doit tourner sans SMTP : le club a son lien."""
    assert LogNotifier().send(Message("a@b.fr", "sujet", "corps")) is False


def test_a_dead_smtp_server_does_not_raise():
    """Un envoi raté ne doit pas faire échouer une analyse réussie."""
    notifier = SmtpNotifier(
        host="127.0.0.1", port=1, user="u", password="p",
        expediteur="no-reply@example.fr", timeout=0.5,
    )
    assert notifier.send(Message("club@example.fr", "sujet", "corps")) is False


def test_the_environment_builds_a_sender(monkeypatch):
    from service import notify

    monkeypatch.setenv("FA_SMTP_HOST", "smtp.example.fr")
    monkeypatch.setenv("FA_SMTP_FROM", "no-reply@example.fr")
    assert isinstance(notify.from_environment(), SmtpNotifier)


def test_an_incomplete_environment_falls_back_to_logging(monkeypatch):
    from service import notify

    monkeypatch.setenv("FA_SMTP_HOST", "smtp.example.fr")
    monkeypatch.delenv("FA_SMTP_FROM", raising=False)
    assert isinstance(notify.from_environment(), LogNotifier)
