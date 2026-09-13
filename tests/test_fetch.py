"""Récupération d'une vidéo par lien.

C'est le service qui télécharge : c'est donc le service qu'on peut faire
pointer n'importe où. La validation de l'adresse est la seule barrière entre
un lien saisi par un inconnu et le réseau interne de la machine.

La résolution de noms est simulée : un test qui interroge le vrai DNS échoue
hors ligne et selon l'hébergeur.
"""
from __future__ import annotations

import socket

import pytest

import service.fetch as fetch
from service.fetch import LienRefuse, valider

PUBLIQUE = "93.184.216.34"


@pytest.fixture
def dns(monkeypatch):
    """Résolution simulée : chaque hôte reçoit l'adresse qu'on lui assigne."""
    table: dict[str, str] = {}

    def resoudre(hote, *a, **k):
        if hote not in table:
            raise socket.gaierror("inconnu")
        return [(socket.AF_INET, None, None, "", (table[hote], 0))]

    monkeypatch.setattr(fetch.socket, "getaddrinfo", resoudre)
    return table


@pytest.mark.parametrize("lien", [
    "https://www.youtube.com/watch?v=abc",
    "http://video.club.fr/match.mp4",
])
def test_a_public_link_is_accepted(dns, lien):
    from urllib.parse import urlparse

    dns[urlparse(lien).hostname] = PUBLIQUE
    assert valider(lien) == lien


@pytest.mark.parametrize("lien", [
    "file:///etc/passwd", "ftp://exemple.fr/v.mp4", "javascript:alert(1)",
    "data:video/mp4;base64,AAAA",
])
def test_only_http_links_are_accepted(dns, lien):
    with pytest.raises(LienRefuse, match="http"):
        valider(lien)


@pytest.mark.parametrize("adresse", [
    "127.0.0.1", "169.254.169.254", "10.0.0.5", "192.168.1.10", "172.16.0.1",
])
def test_the_internal_network_is_unreachable(dns, adresse):
    """169.254.169.254 exposerait les identifiants d'accès de la machine
    chez la plupart des hébergeurs."""
    dns["interne.exemple"] = adresse
    with pytest.raises(LienRefuse, match="accessible"):
        valider("https://interne.exemple/video.mp4")


def test_a_public_name_pointing_inside_is_still_refused(dns):
    """Un nom public peut résoudre vers une adresse interne : c'est
    l'adresse qui décide, pas le nom."""
    dns["cdn-innocent.fr"] = "169.254.169.254"
    with pytest.raises(LienRefuse, match="accessible"):
        valider("https://cdn-innocent.fr/v.mp4")


def test_an_unresolvable_host_is_refused(dns):
    with pytest.raises(LienRefuse):
        valider("https://inexistant.exemple/v.mp4")


def test_an_empty_link_is_refused(dns):
    with pytest.raises(LienRefuse, match="Aucun lien"):
        valider("   ")


def test_a_link_without_a_host_is_refused(dns):
    with pytest.raises(LienRefuse, match="incomplet"):
        valider("https:///video.mp4")


def test_a_missing_downloader_says_so(dns, tmp_path, monkeypatch):
    """Le service doit tourner sans yt-dlp : le téléversement reste possible."""
    dns["video.club.fr"] = PUBLIQUE

    def absent(*a, **k):
        raise FileNotFoundError("yt-dlp")

    monkeypatch.setattr(fetch.subprocess, "run", absent)
    with pytest.raises(LienRefuse, match="indisponible"):
        fetch.telecharger("https://video.club.fr/v.mp4", tmp_path / "source")


def test_a_failed_download_explains_what_to_check(dns, tmp_path, monkeypatch):
    import subprocess

    dns["video.club.fr"] = PUBLIQUE
    monkeypatch.setattr(
        fetch.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "erreur"),
    )
    with pytest.raises(LienRefuse, match="public"):
        fetch.telecharger("https://video.club.fr/v.mp4", tmp_path / "source")


def test_a_timeout_is_explained(dns, tmp_path, monkeypatch):
    import subprocess

    dns["video.club.fr"] = PUBLIQUE

    def trop_long(*a, **k):
        raise subprocess.TimeoutExpired("yt-dlp", 1)

    monkeypatch.setattr(fetch.subprocess, "run", trop_long)
    with pytest.raises(LienRefuse, match="trop de temps"):
        fetch.telecharger("https://video.club.fr/v.mp4", tmp_path / "source")


def test_a_successful_download_returns_the_file(dns, tmp_path, monkeypatch):
    import subprocess

    dns["video.club.fr"] = PUBLIQUE
    cible = tmp_path / "source"

    def reussit(*a, **k):
        cible.with_suffix(".mp4").write_bytes(b"video")
        return subprocess.CompletedProcess(a, 0, "", "")

    monkeypatch.setattr(fetch.subprocess, "run", reussit)
    assert fetch.telecharger("https://video.club.fr/v.mp4", cible).suffix == ".mp4"


def test_a_partial_file_is_not_mistaken_for_the_video(dns, tmp_path, monkeypatch):
    """yt-dlp laisse un .part derrière lui quand il échoue en cours de route."""
    import subprocess

    dns["video.club.fr"] = PUBLIQUE
    cible = tmp_path / "source"

    def echoue(*a, **k):
        cible.with_suffix(".mp4.part").write_bytes(b"incomplet")
        return subprocess.CompletedProcess(a, 1, "", "interrompu")

    monkeypatch.setattr(fetch.subprocess, "run", echoue)
    with pytest.raises(LienRefuse):
        fetch.telecharger("https://video.club.fr/v.mp4", cible)
