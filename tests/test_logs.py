"""Masquage des jetons dans les journaux.

Les adresses portent le jeton d'accès. Le journal d'accès d'uvicorn les écrit
telles quelles : l'hébergeur, l'agrégateur de journaux et tout administrateur
verraient les jetons de tous les clubs sans avoir eu à les demander.
"""
from __future__ import annotations

import logging

import pytest

from service.logs import MasqueJetons


@pytest.fixture
def journal(caplog):
    logger = logging.getLogger("test.jetons")
    logger.addFilter(MasqueJetons())
    caplog.set_level(logging.INFO, logger="test.jetons")
    return logger


def test_a_match_token_is_masked(journal, caplog):
    journal.info('GET /m/96e28fa7bd2400c7/nZFR8TskVYAxt3PM0x13dkRCEl0 HTTP/1.1')
    trace = caplog.text
    assert "nZFR8TskVYAxt3PM0x13dkRCEl0" not in trace
    assert "/m/96e28fa7bd2400c7/***" in trace


def test_a_club_token_is_masked(journal, caplog):
    journal.info("GET /c/da085a6e49f51b8b/EcomWZn1KpQ HTTP/1.1")
    assert "EcomWZn1KpQ" not in caplog.text
    assert "/c/da085a6e49f51b8b/***" in caplog.text


def test_the_rest_of_the_path_survives(journal, caplog):
    """Masquer trop empêcherait de distinguer un rapport d'une vidéo."""
    journal.info("GET /m/abc/JETON/rapport HTTP/1.1")
    assert "/m/abc/***/rapport" in caplog.text


def test_tokens_passed_as_arguments_are_masked(journal, caplog):
    """uvicorn journalise l'adresse en argument de formatage, pas dans le
    message : ne filtrer que le message laisserait tout passer."""
    journal.info('%s - "%s %s"', "127.0.0.1", "GET", "/m/abc/JETON_SECRET")
    assert "JETON_SECRET" not in caplog.text
    assert "/m/abc/***" in caplog.text


def test_an_ordinary_message_is_untouched(journal, caplog):
    journal.info("worker démarré, données dans /data")
    assert "worker démarré, données dans /data" in caplog.text


def test_other_paths_are_not_mangled(journal, caplog):
    journal.info("GET /sante HTTP/1.1")
    assert "/sante" in caplog.text
    assert "***" not in caplog.text
