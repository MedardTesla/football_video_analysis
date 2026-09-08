"""Aucune classe utilisée dans le balisage ne doit rester sans style.

Plusieurs règles avaient silencieusement disparu au fil des modifications :
le tableau de l'espace du club et les courbes de saison s'affichaient avec
les styles par défaut du navigateur. Rien ne le signalait — les pages se
rendaient, simplement mal.
"""
from __future__ import annotations

import re

import pytest

from football_analysis.report import STYLE as STYLE_RAPPORT, ReportMeta, render
from service.jobs import Club, Job, JobState
from service.season import build
from service.web import pages

CLUB = Club(id="c1", token="jeton", name="ASKO Kara")


def _match(nom="ASKO – Djoliba", etat=JobState.DONE, **kw):
    return Job(id="m1", token="t", club="ASKO Kara", match_name=nom,
               video_path="/tmp/v.mp4", club_id="c1", state=etat,
               report_path="/tmp/r.html", **kw)


def _classes(page: str) -> set[str]:
    trouvees: set[str] = set()
    for attribut in re.findall(r'class="([^"]+)"', page):
        trouvees.update(attribut.split())
    return trouvees


def _stylees(feuille: str) -> set[str]:
    return set(re.findall(r"\.([A-Za-z_][\w-]*)", feuille))


def _match_avec_stats(**kw):
    stats = {"coverage": 0.83, "possession": {"0": 0.47, "1": 0.53},
             "control": {"0": 0.53, "1": 0.47},
             "players": [{"track_id": 4, "team": 0, "distance_m": 9000.0,
                          "top_speed_ms": 8.0, "seconds_seen": 4800.0}]}
    return _match(stats=stats, **kw)


PAGES = {
    "accueil": lambda: pages.home_page(),
    "dépôt": lambda: pages.upload_form(),
    "dépôt rattaché": lambda: pages.upload_form(club=CLUB),
    "dépôt en erreur": lambda: pages.upload_form(erreur="Format non pris en charge."),
    "confirmation": lambda: pages.upload_done(_match(), 2, CLUB),
    "suivi en attente": lambda: pages.status_page(_match(etat=JobState.QUEUED), 3),
    "suivi en cours": lambda: pages.status_page(
        _match(etat=JobState.PROCESSING, progress=0.62), 0),
    "suivi terminé": lambda: pages.status_page(_match(), 0),
    "suivi en échec": lambda: pages.status_page(
        _match(etat=JobState.FAILED, error="La vidéo n'a pas pu être lue."), 0),
    "espace du club": lambda: pages.club_page(CLUB, [_match_avec_stats(our_team=0)], 0),
    "nommage": lambda: pages.naming_page(
        _match_avec_stats(), pages.nommables(_match_avec_stats().stats)),
    "erreur": lambda: pages.error_page(404, "Match introuvable."),
}


@pytest.mark.parametrize("nom", sorted(PAGES))
def test_every_class_used_has_a_rule(nom):
    utilisees = _classes(PAGES[nom]())
    manquantes = utilisees - _stylees(pages.STYLE)
    assert not manquantes, f"{nom} : classes sans style — {sorted(manquantes)}"


def test_the_season_chart_classes_are_styled():
    matchs = [_match_avec_stats(our_team=0), _match_avec_stats(our_team=0)]
    page = pages.club_page(CLUB, matchs, 0, build(matchs))
    manquantes = _classes(page) - _stylees(pages.STYLE)
    assert not manquantes, sorted(manquantes)


def test_the_report_classes_are_styled():
    stats = {"coverage": 0.83, "possession": {"0": 0.5, "1": 0.5},
             "control": {"0": 0.5, "1": 0.5},
             "players": [{"track_id": 4, "team": 0, "distance_m": 9000.0,
                          "top_speed_ms": 8.0, "seconds_seen": 4800.0}]}
    page = render(stats, ReportMeta("Match"), names={"4": "Kossi Adjovi"})
    manquantes = _classes(page) - _stylees(STYLE_RAPPORT)
    assert not manquantes, sorted(manquantes)


def test_no_rule_references_an_undefined_token():
    """Une couleur définie seulement dans un bloc de thème n'existe pas dans
    l'état par défaut, et la page se rend illisible."""
    for nom, feuille in (("pages", pages.STYLE), ("rapport", STYLE_RAPPORT)):
        racine = re.search(r":root \{(.*?)\n\}", feuille, re.S).group(1)
        definis = set(re.findall(r"--([a-z-]+):", racine))
        utilises = set(re.findall(r"var\(--([a-z-]+)\)", feuille))
        assert not utilises - definis, f"{nom} : {sorted(utilises - definis)}"


def test_the_report_rows_carry_their_column_labels():
    """Empilée sur un téléphone, une ligne perd son en-tête."""
    stats = {"players": [{"track_id": 4, "team": 0, "distance_m": 9000.0,
                          "top_speed_ms": 8.0, "seconds_seen": 4800.0}]}
    page = render(stats, ReportMeta("Match"))
    for intitule in ("Équipe", "Distance", "Pointe", "Temps"):
        assert f'data-champ="{intitule}"' in page


def test_the_report_table_becomes_cards_on_a_narrow_screen():
    assert "@media (max-width:620px)" in STYLE_RAPPORT
    assert ".players tbody td::before" in STYLE_RAPPORT
