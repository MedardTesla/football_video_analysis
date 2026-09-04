from __future__ import annotations

import json
import re
from datetime import date

import pytest

from football_analysis.report import ReportMeta, render, write

META = ReportMeta(match_name="US Exemple – AS Test", played_on=date(2026, 9, 3), duration_s=2700)

STATS = {
    "possession": {"0": 0.58, "1": 0.42},
    "players": [
        {"track_id": 4, "team": 0, "distance_m": 9812.4, "top_speed_ms": 8.3, "seconds_seen": 2650},
        {"track_id": 9, "team": 1, "distance_m": 8730.1, "top_speed_ms": 9.1, "seconds_seen": 2600},
    ],
}


def test_report_contains_key_figures():
    page = render(STATS, META)
    assert "US Exemple" in page
    assert "58%" in page and "42%" in page
    assert "9.8" in page  # 9812 m affichés en km


def test_report_escapes_club_names():
    """Les noms de clubs viennent d'une saisie utilisateur."""
    page = render(STATS, ReportMeta(match_name="<script>alert(1)</script>"))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_report_survives_empty_analysis():
    page = render({"possession": {}, "players": []}, META)
    assert "non calculable" in page
    assert "Aucun joueur suivi" in page


def test_missing_pitch_is_flagged_to_the_club():
    page = render({"possession": {}, "players": []}, META)
    assert "caméra placée trop bas" in page


def test_identity_fragmentation_is_flagged():
    stats = {
        "possession": {"0": 0.5, "1": 0.5},
        "players": [
            {"track_id": i, "team": i % 2, "distance_m": 100.0,
             "top_speed_ms": 5.0, "seconds_seen": 2000}
            for i in range(40)
        ],
    }
    assert "40 identités pour 22 joueurs" in render(stats, META)


def test_players_without_team_do_not_crash():
    stats = {"possession": {}, "players": [
        {"track_id": 1, "team": None, "distance_m": 10.0,
         "top_speed_ms": 1.0, "seconds_seen": 90}
    ]}
    assert "Non attribué" in render(stats, META)


def test_write_produces_a_single_file(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(STATS))
    output = write(stats_path, tmp_path / "report.html", META)
    page = output.read_text(encoding="utf-8")
    assert output.exists()
    assert page.lstrip().startswith("<!doctype html>")


def test_only_fonts_are_fetched_from_the_network(tmp_path):
    """Le rapport doit rester lisible hors ligne.

    Les polices web sont la seule ressource distante tolérée : elles
    dégradent sur la pile de repli. Toute autre ressource externe (image,
    script, CSS) rendrait le rapport cassé sans connexion.
    """
    page = render(STATS, META)
    urls = re.findall(r'https?://[^"\s]+', page)
    assert urls, "polices attendues"
    for url in urls:
        assert url.startswith(
            ("https://fonts.googleapis.com", "https://fonts.gstatic.com")
        ), url


def test_every_font_family_has_a_local_fallback():
    """Sans repli local, le rapport s'affiche en Times hors ligne."""
    page = render(STATS, META)
    for declaration in re.findall(r"font(?:-family)?\s*:[^;{}]+", page):
        if '"' not in declaration:
            continue
        families = [f.strip().strip('"') for f in declaration.split(":", 1)[1].split(",")]
        webfonts = {"Barlow Condensed", "IBM Plex Sans", "IBM Plex Mono"}
        assert any(f not in webfonts for f in families), declaration


def test_coverage_is_shown_before_the_figures():
    """Le club doit savoir sur quelle portion du match portent les chiffres."""
    stats = dict(STATS, coverage=0.47, unmeasured_seconds=2900, measured_seconds=2500)
    page = render(stats, META)
    assert "47%" in page
    assert page.index("du match analysé") < page.index("Possession")


def test_low_coverage_warns_against_individual_figures():
    stats = dict(STATS, coverage=0.42, unmeasured_seconds=3100)
    page = render(stats, META)
    assert "cov--bad" in page
    assert "ne pas se fier aux chiffres individuels" in page


def test_good_coverage_is_not_alarming():
    stats = dict(STATS, coverage=0.94, unmeasured_seconds=320)
    page = render(stats, META)
    assert "cov--ok" in page
    assert "fiables" in page


def test_low_coverage_explains_the_remedy():
    stats = dict(STATS, coverage=0.47, unmeasured_seconds=2900)
    page = render(stats, META)
    assert "plus haut et plus reculé" in page


def test_distances_are_not_presented_as_a_floor():
    """Deux effets opposés : la couverture les diminue, l'imprécision de
    localisation les augmente — une erreur de 5 m multiplie par seize la
    longueur d'un pas. Les annoncer comme un plancher serait faux."""
    page = render(dict(STATS, coverage=0.47, unmeasured_seconds=2900), META)
    assert "plancher" not in page
    assert "sens contraire" in page
    assert "ordres de grandeur" in page


def test_report_without_coverage_still_renders():
    """Les anciens fichiers de statistiques n'ont pas ce champ."""
    page = render(STATS, META)
    assert "du match analysé" not in page
    assert "9.8" in page


def test_territorial_control_is_shown_when_available():
    """La statistique qui survit à une localisation imprécise."""
    stats = dict(STATS, control={"0": 0.56, "1": 0.44}, control_seconds=4800)
    page = render(stats, META)
    assert "Contrôle du terrain" in page
    assert "56%" in page and "44%" in page


def test_control_is_presented_apart_from_possession():
    """Deux notions distinctes : on peut avoir le ballon sans occuper
    le terrain."""
    stats = dict(STATS, control={"0": 0.56, "1": 0.44})
    page = render(stats, META)
    assert page.index("Possession") < page.index("Contrôle du terrain")
    assert "sans occuper le terrain" in page


def test_control_is_recommended_over_distances_with_figures():
    """L'écart mesuré : 2 points contre un facteur seize."""
    page = render(dict(STATS, control={"0": 0.56, "1": 0.44}), META)
    assert "deux points" in page
    assert "seize" in page


def test_a_report_without_control_omits_the_section():
    page = render(STATS, META)
    assert "Contrôle du terrain" not in page
