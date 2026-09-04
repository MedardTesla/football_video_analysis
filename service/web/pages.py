"""Pages du service, rendues côté serveur.

Trois écrans, pas davantage : déposer, attendre, lire. Un club de village n'a
ni compte, ni tableau de bord, ni envie d'apprendre une interface — il veut
déposer une vidéo et recevoir des chiffres.

La palette et les polices sont celles du rapport (`football_analysis.report`),
pour que le dépôt et le résultat se lisent comme un seul produit.
"""
from __future__ import annotations

import html

from ..jobs import Club, Job, JobState
from ..season import Season

STYLE = """
:root {
  --ground:#f7f8f5; --surface:#ffffff; --ink:#151a16; --muted:#5f6b62;
  --line:#e2e6df; --turf:#2f6d43; --turf-ink:#ffffff;
  --note-bg:#fdf8ec; --note-line:#e8dcc0; --note-ink:#6b5522;
  --bad-bg:#fcf2f1; --bad-line:#eccecb; --bad-ink:#8c3a30;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#101310; --surface:#191d18; --ink:#e6ebe4; --muted:#98a297;
    --line:#2b312a; --turf:#63a97a; --turf-ink:#101310;
    --note-bg:#221d12; --note-line:#3d3524; --note-ink:#d9c79b;
    --bad-bg:#241614; --bad-line:#422a26; --bad-ink:#e0a79e;
  }
}
:root[data-theme="dark"] {
  --ground:#101310; --surface:#191d18; --ink:#e6ebe4; --muted:#98a297;
  --line:#2b312a; --turf:#63a97a; --turf-ink:#101310;
  --note-bg:#221d12; --note-line:#3d3524; --note-ink:#d9c79b;
  --bad-bg:#241614; --bad-line:#422a26; --bad-ink:#e0a79e;
}
* { box-sizing:border-box; }
body { margin:0; padding:3rem 1.25rem; background:var(--ground); color:var(--ink);
       font:16px/1.6 "IBM Plex Sans","Segoe UI",system-ui,sans-serif; }
main { max-width:34rem; margin:0 auto; display:flex; flex-direction:column; gap:1.5rem; }
header { border-bottom:2px solid var(--turf); padding-bottom:1rem;
         display:flex; flex-direction:column; gap:.3rem; }
.eyebrow { font:600 .7rem/1 "IBM Plex Sans",sans-serif; letter-spacing:.16em;
           text-transform:uppercase; color:var(--turf); }
h1 { font:600 clamp(1.8rem,5vw,2.5rem)/1.05 "Barlow Condensed","Arial Narrow",sans-serif;
     margin:0; text-wrap:balance; }
p { margin:0; }
.lede { color:var(--muted); }
form { display:flex; flex-direction:column; gap:1.1rem; }
label { display:flex; flex-direction:column; gap:.35rem; font-size:.85rem;
        font-weight:600; }
input[type=text], input[type=file] {
  font:inherit; font-weight:400; padding:.65rem .75rem; border-radius:5px;
  border:1px solid var(--line); background:var(--surface); color:var(--ink); }
input[type=file] { padding:.55rem; }
input:focus-visible { outline:2px solid var(--turf); outline-offset:1px; }
button { font:600 1rem/1 "IBM Plex Sans",sans-serif; padding:.85rem 1.2rem;
         border:none; border-radius:5px; background:var(--turf);
         color:var(--turf-ink); cursor:pointer; }
button:hover { filter:brightness(1.08); }
.hint { font-size:.8rem; color:var(--muted); font-weight:400; }
.option { font-weight:400; font-size:.75rem; color:var(--muted);
          text-transform:uppercase; letter-spacing:.06em; margin-left:.4rem; }
.panel { background:var(--surface); border:1px solid var(--line);
         border-radius:8px; padding:1.25rem 1.4rem;
         display:flex; flex-direction:column; gap:.8rem; }
.lien { font:.85rem/1.5 "IBM Plex Mono",ui-monospace,monospace;
        background:var(--ground); border:1px solid var(--line); border-radius:5px;
        padding:.6rem .75rem; word-break:break-all; }
.etat { display:flex; align-items:center; gap:.9rem; }
.pastille { width:11px; height:11px; border-radius:50%; flex:none;
            background:var(--turf); }
.pastille--attente { background:var(--note-ink); }
.pastille--echec { background:var(--bad-ink); }
.etat strong { font-size:1.05rem; }
.jauge { height:8px; border-radius:4px; background:var(--line); overflow:hidden; }
.jauge span { display:block; height:100%; background:var(--turf); border-radius:4px;
              transition:width .4s ease; }
@media (prefers-reduced-motion:reduce) { .jauge span { transition:none; } }
.note { background:var(--note-bg); border:1px solid var(--note-line);
        color:var(--note-ink); border-radius:6px; padding:.9rem 1.1rem;
        font-size:.88rem; }
.erreur { background:var(--bad-bg); border:1px solid var(--bad-line);
          color:var(--bad-ink); border-radius:6px; padding:.9rem 1.1rem;
          font-size:.88rem; }
.actions { display:flex; gap:.7rem; flex-wrap:wrap; }
.actions a { font:600 .9rem/1 "IBM Plex Sans",sans-serif; text-decoration:none;
             padding:.75rem 1.1rem; border-radius:5px; }
.principal { background:var(--turf); color:var(--turf-ink); }
.secondaire { background:var(--surface); color:var(--ink);
              border:1px solid var(--line); }
footer { color:var(--muted); font-size:.78rem; border-top:1px solid var(--line);
         padding-top:1rem; }
@media (prefers-reduced-motion:reduce) { * { animation:none !important; } }
"""

# Mêmes teintes que le rapport, pour qu'une courbe et un rapport se lisent
# comme un seul produit.
TEAM_HEX_A = "#0080ff"
TEAM_HEX_B = "#d4622a"

LIBELLES = {
    JobState.QUEUED: ("En attente", "pastille--attente"),
    JobState.PROCESSING: ("Analyse en cours", ""),
    JobState.DONE: ("Analyse terminée", ""),
    JobState.FAILED: ("Analyse impossible", "pastille--echec"),
}


def _document(titre: str, corps: str, tete: str = "") -> str:
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(titre)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{STYLE}</style>{tete}</head>
<body><main>{corps}</main></body></html>"""


def _attente(en_attente: int) -> str:
    if en_attente <= 0:
        return "L'analyse démarre immédiatement."
    if en_attente == 1:
        return "Un match est en cours de traitement avant celui-ci."
    return f"{en_attente} matchs sont en attente avant celui-ci."


def _champs_formulaire(club: Club | None) -> str:
    """Champ « club » masqué et pré-rempli quand on vient de son espace.

    Le jeton voyage en champ caché plutôt qu'en paramètre d'adresse : il
    n'apparaît alors ni dans l'historique du navigateur ni dans les journaux
    du serveur, alors que le formulaire est parfois rempli sur un poste
    partagé au club.
    """
    if club is None:
        return """  <label>Club
   <input type="text" name="club" required maxlength="80" placeholder="US Valmont">
  </label>"""
    return (
        f'  <input type="hidden" name="club_id" value="{html.escape(club.id)}">\n'
        f'  <input type="hidden" name="club_token" value="{html.escape(club.token)}">\n'
        f'  <input type="hidden" name="club" value="{html.escape(club.name)}">'
    )


def upload_form(erreur: str | None = None, club: Club | None = None) -> str:
    alerte = f'<p class="erreur">{html.escape(erreur)}</p>' if erreur else ""
    titre = "Déposer un match" if club else "Déposer une vidéo"
    corps = f"""
 <header>
  <span class="eyebrow">{html.escape(club.name) if club else "Analyse de match"}</span>
  <h1>{titre}</h1>
 </header>
 <p class="lede">Vous recevez un lien à conserver. L'analyse dure environ une
 heure ; vous pouvez fermer cette page.</p>
 {alerte}
 <form method="post" action="/matches" enctype="multipart/form-data">
{_champs_formulaire(club)}
  <label>Match
   <input type="text" name="match_name" required maxlength="120"
          placeholder="US Valmont – AS Beaupré, 30 août">
  </label>
  <label>Vidéo
   <input type="file" name="video" accept="video/*" required>
   <span class="hint">MP4, MOV, AVI ou MKV. 8 Go maximum, soit environ 2 h en 1080p.</span>
  </label>
  <label>Adresse e-mail <span class="option">facultatif</span>
   <input type="email" name="contact" maxlength="120" placeholder="entraineur@club.fr">
   <span class="hint">Pour être prévenu quand le rapport est prêt. Le message
   contient le lien d'accès : vérifiez l'adresse, toute personne qui reçoit ce
   lien peut lire le rapport.</span>
  </label>
  <button type="submit">Envoyer la vidéo</button>
 </form>
 <p class="note">Filmez depuis un point haut et reculé : cela double la part du
 match réellement analysable, bien plus que n'importe quel réglage de notre côté.</p>
 <footer>La vidéo est supprimée de nos serveurs dès le rapport produit.</footer>"""
    return _document(titre, corps)


def _selecteur_equipe(club: Club, job: Job) -> str:
    """Deux boutons : laquelle des deux équipes du rapport est celle du club.

    Formulaire et non lien : désigner une équipe modifie un état, et un lien
    cliqué par un aspirateur de pages le changerait à l'insu du club.
    """
    boutons = []
    for equipe, libelle in ((0, "A"), (1, "B")):
        actif = " actif" if job.our_team == equipe else ""
        boutons.append(
            f'<form method="post" action="{html.escape(club.public_url)}'
            f'/match/{html.escape(job.id)}/equipe">'
            f'<input type="hidden" name="team" value="{equipe}">'
            f'<button class="equipe{actif}" type="submit">{libelle}</button></form>'
        )
    return f'<div class="equipes">{"".join(boutons)}</div>'


ETIQUETTES_COURTES = {
    JobState.QUEUED: "En attente",
    JobState.PROCESSING: "En cours",
    JobState.DONE: "Prêt",
    JobState.FAILED: "Échec",
}


def _courbe(points: list, valeur, couleur: str, titre: str) -> str:
    """Courbe d'une statistique d'équipe, match après match.

    SVG écrit à la main : une bibliothèque de graphiques pèserait plus lourd
    que toute la page, pour une courbe de quelques points lue sur un
    téléphone.
    """
    valeurs = [(i, valeur(p)) for i, p in enumerate(points) if valeur(p) is not None]
    if len(valeurs) < 2:
        return ""

    L, H, MARGE = 600, 130, 18
    pas = (L - 2 * MARGE) / max(len(points) - 1, 1)
    coords = [
        (MARGE + i * pas, H - MARGE - v * (H - 2 * MARGE))
        for i, v in valeurs
    ]
    ligne = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    pastilles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{couleur}"/>' for x, y in coords
    )
    moyenne = sum(v for _, v in valeurs) / len(valeurs)
    y_moy = H - MARGE - moyenne * (H - 2 * MARGE)

    return f"""<figure class="courbe">
 <figcaption>{titre} <b>{moyenne:.0%}</b> en moyenne</figcaption>
 <svg viewBox="0 0 {L} {H}" role="img" aria-label="{titre} sur {len(valeurs)} matchs">
  <line x1="{MARGE}" y1="{H - MARGE - 0.5 * (H - 2 * MARGE):.1f}"
        x2="{L - MARGE}" y2="{H - MARGE - 0.5 * (H - 2 * MARGE):.1f}"
        class="mediane"/>
  <line x1="{MARGE}" y1="{y_moy:.1f}" x2="{L - MARGE}" y2="{y_moy:.1f}"
        class="moyenne" stroke="{couleur}"/>
  <polyline points="{ligne}" fill="none" stroke="{couleur}" stroke-width="2.5"
            stroke-linejoin="round" stroke-linecap="round"/>
  {pastilles}
 </svg>
 <div class="courbe__legende"><span>50 % = équilibre</span>
  <span>{points[0].date} → {points[-1].date}</span></div>
</figure>"""


def _tendance(saison: Season, total_matchs: int) -> str:
    if not saison.usable:
        manquants = total_matchs - len(saison.points)
        if manquants > 0:
            return (
                '<section class="panel"><h2>Tendance de la saison</h2>'
                "<p class='lede'>Désignez votre équipe sur au moins deux matchs "
                "analysés pour voir la tendance. Les libellés « équipe A » et "
                "« équipe B » d'un rapport sont attribués automatiquement : rien "
                "ne garantit qu'ils désignent la même équipe d'un match à "
                "l'autre.</p></section>"
            )
        return ""

    return f"""<section class="panel">
 <h2>Tendance de la saison</h2>
 <p class="lede">Sur {len(saison.points)} matchs où votre équipe est désignée.
 Les distances individuelles ne sont pas agrégées : leur imprécision se
 cumulerait au lieu de se compenser.</p>
 {_courbe(saison.points, lambda p: p.possession, TEAM_HEX_A, "Possession")}
 {_courbe(saison.points, lambda p: p.control, TEAM_HEX_B, "Contrôle du terrain")}
</section>"""


def club_page(
    club: Club, matchs: list[Job], en_attente: int, saison: Season | None = None
) -> str:
    """Espace du club : tous ses matchs derrière un seul lien.

    Sans cette page, un club qui analyse dix matchs conserve dix liens. La
    valeur d'un tel produit tient pourtant dans la tendance d'une saison, pas
    dans un match isolé.
    """
    if matchs:
        lignes = []
        for m in matchs:
            libelle = ETIQUETTES_COURTES[m.state]
            classe = LIBELLES[m.state][1]
            date = m.created_at[:10]
            if m.state is JobState.DONE:
                lien = f'{html.escape(m.public_url)}/rapport'
                action = f'<a href="{lien}">Voir le rapport</a>'
                notre = _selecteur_equipe(club, m)
            else:
                action = f'<a href="{html.escape(m.public_url)}">Suivre</a>'
                notre = '<span class="muet">—</span>'
            lignes.append(
                f"<tr><td>{html.escape(m.match_name)}</td>"
                f'<td class="date">{date}</td>'
                f'<td><span class="pastille {classe}"></span>{libelle}</td>'
                f"<td>{notre}</td>"
                f'<td class="action">{action}</td></tr>'
            )
        table = (
            '<div class="scroll"><table><thead><tr><th>Match</th><th>Déposé</th>'
            "<th>État</th><th>Votre équipe</th><th></th></tr></thead><tbody>"
            + "".join(lignes)
            + "</tbody></table></div>"
        )
    else:
        table = "<p class='lede'>Aucun match analysé pour l'instant.</p>"

    corps = f"""
 <header>
  <span class="eyebrow">Espace du club</span>
  <h1>{html.escape(club.name)}</h1>
 </header>
 {_tendance(saison, len(matchs)) if saison else ""}
 <div class="panel">{table}</div>
 <div class="actions">
  <a class="principal" href="{html.escape(club.public_url)}/deposer">Déposer un match</a>
 </div>
 <footer>Conservez l'adresse de cette page : elle donne accès à tous vos
 rapports, et n'est envoyée nulle part ailleurs.</footer>"""
    return _document(club.name, corps)


def upload_done(job: Job, en_attente: int, club: Club | None = None) -> str:
    if job.contact:
        avis = (f"<p>Vous serez prévenu à <strong>{html.escape(job.contact)}</strong> "
                "dès que le rapport sera prêt. Conservez tout de même ce lien.</p>")
    else:
        avis = ("<p><strong>Conservez ce lien.</strong> C'est le seul moyen de "
                "retrouver votre rapport — il ne vous sera pas renvoyé.</p>")

    espace = ""
    if club is not None:
        espace = f"""
 <div class="panel">
  <p><strong>L'espace de votre club</strong> réunit tous vos matchs derrière une
  seule adresse. C'est celle à conserver sur la durée.</p>
  <p class="lien">{html.escape(club.public_url)}</p>
 </div>"""
    corps = f"""
 <header>
  <span class="eyebrow">Vidéo reçue</span>
  <h1>{html.escape(job.match_name)}</h1>
 </header>
 <div class="panel">
  {avis}
  <p class="lien">{html.escape(job.public_url)}</p>
 </div>
 {espace}
 <p class="lede">{_attente(en_attente)}</p>
 <div class="actions">
  <a class="principal" href="{html.escape(job.public_url)}">Suivre l'analyse</a>
  <a class="secondaire" href="/">Déposer un autre match</a>
 </div>"""
    return _document(f"{job.match_name} — vidéo reçue", corps)


def status_page(job: Job, en_attente: int) -> str:
    libelle, classe = LIBELLES[job.state]

    if job.state is JobState.DONE:
        detail = "<p class='lede'>Votre rapport est prêt.</p>"
        actions = f"""<div class="actions">
   <a class="principal" href="{html.escape(job.public_url)}/rapport">Voir le rapport</a>
   <a class="secondaire" href="{html.escape(job.public_url)}/video">Vidéo annotée</a>
  </div>"""
    elif job.state is JobState.FAILED:
        detail = f"<p class='erreur'>{html.escape(job.error or 'Cause inconnue.')}</p>"
        actions = '<div class="actions"><a class="secondaire" href="/">Déposer à nouveau</a></div>'
    elif job.state is JobState.PROCESSING:
        pourcent = int(job.progress * 100)
        detail = (
            f'<div class="jauge"><span style="width:{pourcent}%"></span></div>'
            f"<p class='lede'>{pourcent} % analysé. Comptez environ une heure au total. "
            "Cette page se met à jour seule.</p>"
        )
        actions = ""
    else:
        detail = f"<p class='lede'>{_attente(en_attente)} Cette page se met à jour seule.</p>"
        actions = ""

    # Le rafraîchissement s'arrête de lui-même une fois l'état terminal :
    # laisser tourner une requête toutes les dix secondes sur un rapport
    # consulté longtemps n'apporterait rien. Tant que l'analyse tourne, la
    # jauge est mise à jour sans recharger la page — recharger ferait
    # clignoter l'écran toutes les dix secondes pendant une heure.
    script = "" if job.state.terminal else f"""
<script>
 const jauge = document.querySelector(".jauge span");
 setInterval(async () => {{
   try {{
     const r = await fetch("{job.public_url}/etat");
     if (!r.ok) return;
     const etat = await r.json();
     if (etat.terminal) return location.reload();
     if (jauge && etat.state === "processing") {{
       jauge.style.width = Math.round(etat.progress * 100) + "%";
     }} else if (!jauge && etat.state === "processing") {{
       location.reload();   // passage de « en attente » à « en cours »
     }}
   }} catch (e) {{ /* hors ligne : on réessaiera */ }}
 }}, 10000);
</script>"""

    corps = f"""
 <header>
  <span class="eyebrow">{html.escape(job.club)}</span>
  <h1>{html.escape(job.match_name)}</h1>
 </header>
 <div class="panel">
  <div class="etat"><span class="pastille {classe}"></span><strong>{libelle}</strong></div>
  {detail}
 </div>
 {actions}
 <footer>Conservez l'adresse de cette page : elle est le seul accès à votre rapport.</footer>"""
    return _document(job.match_name, corps, script)
