"""Accessibilité et rendu sur téléphone.

Le marché visé consulte au téléphone, souvent sur des appareils d'entrée de
gamme. Ces défauts ne se voient pas sur un écran de développeur.
"""
from __future__ import annotations

import re

import pytest

from service.jobs import Club, Job, JobState
from service.web import pages

CLUB = Club(id="c1", token="jeton", name="ASKO Kara")


def _match(nom="ASKO – Djoliba", etat=JobState.DONE, equipe=None, progres=0.0):
    return Job(
        id="m1", token="t", club="ASKO Kara", match_name=nom,
        video_path="/tmp/v.mp4", club_id="c1", state=etat, our_team=equipe,
        progress=progres, report_path="/tmp/r.html",
    )


# --- lecteurs d'écran --------------------------------------------------------

def test_the_team_buttons_say_what_they_do():
    """Un lecteur d'écran annonçait « bouton A », sans dire de quel match il
    s'agit ni ce que le clic ferait."""
    page = pages.club_page(CLUB, [_match()], 0)
    for bouton in re.findall(r"<button[^>]*>", page):
        if "equipe" in bouton:
            assert "aria-label=" in bouton
            assert "ASKO" in bouton or "équipe" in bouton


def test_the_chosen_team_is_announced_as_pressed():
    page = pages.club_page(CLUB, [_match(equipe=0)], 0)
    presses = re.findall(r'aria-pressed="(\w+)"', page)
    assert presses.count("true") == 1
    assert presses.count("false") == 1


def test_the_button_says_a_second_click_removes_the_choice():
    page = pages.club_page(CLUB, [_match(equipe=0)], 0)
    assert "Retirer" in page


def test_the_progress_bar_is_announced():
    page = pages.status_page(_match(etat=JobState.PROCESSING, progres=0.62), 0)
    jauge = re.search(r'<div class="jauge"[^>]*>', page).group(0)
    assert 'role="progressbar"' in jauge
    assert 'aria-valuenow="62"' in jauge
    assert "aria-label=" in jauge


# --- clavier -----------------------------------------------------------------

def test_buttons_and_links_show_keyboard_focus():
    """Sur les seuls champs, un utilisateur au clavier perd sa position dès
    qu'il atteint un bouton."""
    for style in (pages.STYLE,):
        assert "button:focus-visible" in style
        assert "a:focus-visible" in style


def test_the_report_also_shows_focus():
    from football_analysis.report import STYLE

    assert "focus-visible" in STYLE


# --- téléphone ---------------------------------------------------------------

def test_the_table_becomes_cards_on_a_narrow_screen():
    """Cinq colonnes sur un téléphone imposent un défilement horizontal."""
    assert "@media (max-width:620px)" in pages.STYLE
    assert "table, tbody, tr, td { display:block" in pages.STYLE


def test_each_cell_carries_its_column_label():
    """Empilée, une ligne perd son en-tête : sans l'intitulé repris devant
    la valeur, on ne sait plus ce qu'on lit."""
    page = pages.club_page(CLUB, [_match()], 0)
    for intitule in ("Déposé", "État", "Votre équipe"):
        assert f'data-champ="{intitule}"' in page


def test_the_form_helps_phone_keyboards():
    page = pages.upload_form()
    assert 'autocomplete="email"' in page
    assert 'inputmode="email"' in page
    assert 'inputmode="url"' in page
