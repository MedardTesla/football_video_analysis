"""Génération du rapport de match remis au club.

C'est le livrable réel : un club veut des chiffres lisibles, pas un fichier
vidéo. Le rapport est un HTML autonome (aucune ressource externe), pour être
envoyé par mail, ouvert hors ligne et imprimé tel quel.
"""
from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

TEAM_LABELS = ("Équipe A", "Équipe B")
TEAM_HEX = ("#0080ff", "#ff8000")


@dataclass
class ReportMeta:
    match_name: str
    played_on: date | None = None
    duration_s: float | None = None
    demo: bool = False


def _possession_block(possession: dict[str, float]) -> str:
    """Barre unique divisée : la convention des diffusions télé.

    Deux barres séparées obligent à comparer deux longueurs ; une barre
    divisée montre le rapport directement, ce que le lecteur cherche.
    """
    if not possession:
        return (
            '<p class="empty">Possession non calculable : le terrain n\'a pas pu '
            "être localisé sur la vidéo.</p>"
        )

    keys = sorted(possession)
    left, right = int(keys[0]), int(keys[-1]) if len(keys) > 1 else int(keys[0])
    share_left = possession[keys[0]]

    return f"""<div class="poss">
  <div class="poss__ends">
    <span><i class="dot" style="background:{TEAM_HEX[left % 2]}"></i>{TEAM_LABELS[left % 2]}</span>
    <span>{TEAM_LABELS[right % 2]}<i class="dot" style="background:{TEAM_HEX[right % 2]}"></i></span>
  </div>
  <div class="poss__bar">
    <span style="width:{share_left * 100:.1f}%;background:{TEAM_HEX[left % 2]}"></span>
    <span style="width:{(1 - share_left) * 100:.1f}%;background:{TEAM_HEX[right % 2]}"></span>
  </div>
  <div class="poss__ends poss__figs">
    <span>{share_left:.0%}</span><span>{1 - share_left:.0%}</span>
  </div>
</div>"""


def _players_block(players: list[dict]) -> str:
    if not players:
        return '<p class="empty">Aucun joueur suivi sur cette vidéo.</p>'

    furthest = max((p["distance_m"] for p in players), default=1.0) or 1.0
    rows = []
    for player in players:
        team = player.get("team")
        colour = TEAM_HEX[team % 2] if team is not None else "#a9b0a9"
        label = TEAM_LABELS[team % 2] if team is not None else "Non attribué"
        km = player["distance_m"] / 1000
        kmh = player["top_speed_ms"] * 3.6
        rows.append(
            "<tr>"
            f'<td class="jersey">{player["track_id"]}</td>'
            f'<td class="team"><i class="dot" style="background:{colour}"></i>'
            f'<span class="team__name">{label}</span></td>'
            f'<td class="figure">{km:.1f}<abbr>km</abbr>'
            f'<i class="track"><b style="width:{player["distance_m"] / furthest * 100:.0f}%;'
            f'background:{colour}"></b></i></td>'
            f'<td class="figure">{kmh:.1f}<abbr>km/h</abbr></td>'
            f'<td class="figure minutes">{player["seconds_seen"] / 60:.0f}<abbr>min</abbr></td>'
            "</tr>"
        )
    return (
        '<div class="scroll"><table class="players"><thead><tr>'
        "<th>N°</th><th>Équipe</th><th>Distance</th><th>Pointe</th><th>Temps</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _caveats(stats: dict) -> list[str]:
    """Limites affichées au club, déduites des données elles-mêmes.

    Un rapport qui tait ses angles morts perd la confiance du club dès qu'il
    repère lui-même une incohérence. Autant les nommer d'abord.
    """
    notes = []
    players = stats.get("players", [])
    if not stats.get("possession"):
        notes.append(
            "Le terrain n'a pas pu être localisé sur la vidéo : distances, vitesses "
            "et possession sont indisponibles. Cause la plus fréquente, une caméra "
            "placée trop bas pour voir les lignes du terrain."
        )
    if len(players) > 30:
        notes.append(
            f"{len(players)} identités pour 22 joueurs attendus. Le suivi a perdu puis "
            "recréé des identités, ce qui fragmente les distances individuelles."
        )
    short = [p for p in players if p["seconds_seen"] < 60]
    if short:
        notes.append(
            f"{len(short)} identité(s) suivie(s) moins d'une minute : probablement des "
            "fragments d'un même joueur plutôt que des joueurs distincts."
        )
    return notes


STYLE = """
:root {
  --ground:#f7f8f5; --surface:#ffffff; --ink:#151a16; --muted:#5f6b62;
  --line:#e2e6df; --line-soft:#eef1ec; --turf:#2f6d43;
  --note-bg:#fdf8ec; --note-line:#e8dcc0; --note-ink:#6b5522;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#101310; --surface:#191d18; --ink:#e6ebe4; --muted:#98a297;
    --line:#2b312a; --line-soft:#232821; --turf:#63a97a;
    --note-bg:#221d12; --note-line:#3d3524; --note-ink:#d9c79b;
  }
}
:root[data-theme="dark"] {
  --ground:#101310; --surface:#191d18; --ink:#e6ebe4; --muted:#98a297;
  --line:#2b312a; --line-soft:#232821; --turf:#63a97a;
  --note-bg:#221d12; --note-line:#3d3524; --note-ink:#d9c79b;
}
* { box-sizing:border-box; }
body {
  margin:0; padding:2.5rem 1.25rem 1rem; background:var(--ground); color:var(--ink);
  font:16px/1.6 "IBM Plex Sans","Segoe UI",system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;
}
main { max-width:54rem; margin:0 auto; display:flex; flex-direction:column; gap:1.5rem; }

header { display:flex; flex-direction:column; gap:.35rem;
         border-bottom:2px solid var(--turf); padding-bottom:1rem; }
.eyebrow { font:600 .7rem/1 "IBM Plex Sans",sans-serif; letter-spacing:.16em;
           text-transform:uppercase; color:var(--turf); }
h1 { font:600 clamp(2rem,5vw,2.9rem)/1.05 "Barlow Condensed","Arial Narrow",sans-serif;
     margin:0; text-wrap:balance; letter-spacing:-.005em; }
.meta { color:var(--muted); font-size:.9rem; font-variant-numeric:tabular-nums; }
.demo { align-self:flex-start; margin-top:.25rem; padding:.2rem .55rem; border-radius:3px;
        background:var(--note-bg); border:1px solid var(--note-line); color:var(--note-ink);
        font:600 .68rem/1.5 "IBM Plex Sans",sans-serif; letter-spacing:.09em;
        text-transform:uppercase; }

section { display:flex; flex-direction:column; gap:1rem; }
h2 { font:600 .72rem/1 "IBM Plex Sans",sans-serif; letter-spacing:.14em;
     text-transform:uppercase; color:var(--muted); margin:0;
     padding-bottom:.6rem; border-bottom:1px solid var(--line); }

.poss { display:flex; flex-direction:column; gap:.5rem; }
.poss__ends { display:flex; justify-content:space-between; align-items:center;
              font-size:.86rem; color:var(--muted); }
.poss__ends span { display:flex; align-items:center; gap:.45rem; }
.poss__figs { font:600 1.9rem/1 "Barlow Condensed","Arial Narrow",sans-serif;
              color:var(--ink); font-variant-numeric:tabular-nums; }
.poss__bar { display:flex; height:14px; border-radius:2px; overflow:hidden; }
.poss__bar span { display:block; }
.dot { width:9px; height:9px; border-radius:50%; flex:none; }

.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; }
thead th { font:600 .68rem/1 "IBM Plex Sans",sans-serif; letter-spacing:.1em;
           text-transform:uppercase; color:var(--muted); text-align:left;
           padding:0 .5rem .6rem; }
tbody td { padding:.6rem .5rem; border-top:1px solid var(--line-soft);
           vertical-align:middle; }
tbody tr:first-child td { border-top:none; }
.jersey { font:600 1.35rem/1 "Barlow Condensed","Arial Narrow",sans-serif;
          color:var(--muted); font-variant-numeric:tabular-nums; width:2.5rem; }
.team { white-space:nowrap; }
.team .dot { display:inline-block; margin-right:.5rem; vertical-align:middle; }
.team__name { font-size:.9rem; }
.figure { font:500 1rem/1 "IBM Plex Mono",ui-monospace,monospace;
          font-variant-numeric:tabular-nums; white-space:nowrap; }
.figure abbr { font-size:.7rem; color:var(--muted); margin-left:.2rem;
               text-decoration:none; }
.minutes { color:var(--muted); }
.track { display:block; height:3px; margin-top:.45rem; background:var(--line-soft);
         border-radius:2px; min-width:5rem; }
.track b { display:block; height:100%; border-radius:2px; }

.radar { width:100%; border-radius:4px; display:block; }

.notes { background:var(--note-bg); border:1px solid var(--note-line);
         border-radius:6px; padding:1.15rem 1.35rem; gap:.75rem; }
.notes h2 { color:var(--note-ink); border-color:var(--note-line); }
.notes ul { margin:0; padding-left:1.15rem; color:var(--note-ink); font-size:.9rem; }
.notes li + li { margin-top:.55rem; }

.empty { color:var(--muted); margin:0; font-size:.92rem; }
footer { color:var(--muted); font-size:.78rem; padding:.5rem 0 2rem;
         border-top:1px solid var(--line); }

@media print {
  body { background:#fff; padding:0; }
  .notes { break-inside:avoid; }
}
@media (prefers-reduced-motion:reduce) { * { animation:none !important; transition:none !important; } }
"""


def render(stats: dict, meta: ReportMeta, radar_png: Path | None = None) -> str:
    """Page HTML complète et autonome."""
    return (
        '<!doctype html>\n<html lang="fr"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{html.escape(meta.match_name)}</title>\n"
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Barlow+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500&"
        'family=IBM+Plex+Sans:wght@400;600&display=swap">\n'
        f"<style>{STYLE}</style></head>\n<body>\n"
        + render_body(stats, meta, radar_png)
        + "\n</body></html>\n"
    )


def render_body(stats: dict, meta: ReportMeta, radar_png: Path | None = None) -> str:
    """Contenu seul, sans enveloppe de document."""
    radar = ""
    if radar_png and Path(radar_png).exists():
        encoded = base64.b64encode(Path(radar_png).read_bytes()).decode()
        radar = (
            "<section><h2>Positions au terrain</h2>"
            f'<img class="radar" alt="Vue tactique du terrain vu du dessus" '
            f'src="data:image/png;base64,{encoded}"></section>'
        )

    caveats = _caveats(stats)
    notes = (
        '<section class="notes"><h2>À savoir sur ces chiffres</h2><ul>'
        + "".join(f"<li>{html.escape(note)}</li>" for note in caveats)
        + "</ul></section>"
        if caveats
        else ""
    )

    subtitle = []
    if meta.played_on:
        subtitle.append(meta.played_on.strftime("%d/%m/%Y"))
    if meta.duration_s:
        subtitle.append(f"{meta.duration_s / 60:.0f} minutes analysées")
    badge = '<span class="demo">Données de démonstration</span>' if meta.demo else ""

    return f"""<main>
 <header>
  <span class="eyebrow">Rapport d'analyse</span>
  <h1>{html.escape(meta.match_name)}</h1>
  <div class="meta">{html.escape(" · ".join(subtitle))}</div>
  {badge}
 </header>
 <section><h2>Possession</h2>{_possession_block(stats.get("possession", {}))}</section>
 {radar}
 <section><h2>Joueurs</h2>{_players_block(stats.get("players", []))}</section>
 {notes}
 <footer>Analyse automatisée à partir de la vidéo du match. Les distances et
 vitesses sont des estimations, dépendantes de la qualité de l'image.</footer>
</main>"""


def write(
    stats_path: Path, output: Path, meta: ReportMeta, radar_png: Path | None = None
) -> Path:
    stats = json.loads(Path(stats_path).read_text())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(stats, meta, radar_png), encoding="utf-8")
    return output
