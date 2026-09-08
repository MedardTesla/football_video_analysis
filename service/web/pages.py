"""Pages du service, rendues côté serveur.

Trois écrans, pas davantage : déposer, attendre, lire. Un club de village n'a
ni compte, ni tableau de bord, ni envie d'apprendre une interface — il veut
déposer une vidéo et recevoir des chiffres.

La palette et les polices sont celles du rapport (`football_analysis.report`),
pour que le dépôt et le résultat se lisent comme un seul produit.
"""
from __future__ import annotations

import html
from collections.abc import Sequence

from ..jobs import Club, Job, JobState
from ..season import Season
from ..settings import PRODUIT

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
input[type=text], input[type=file], input[type=email], input[type=date],
input[type=url] {
  font:inherit; font-weight:400; padding:.65rem .75rem; border-radius:5px;
  border:1px solid var(--line); background:var(--surface); color:var(--ink); }
input[type=file] { padding:.55rem; }
/* Sur les seuls champs, un utilisateur au clavier perd sa position dès
   qu'il atteint un bouton ou un lien. */
input:focus-visible, button:focus-visible, a:focus-visible,
summary:focus-visible { outline:2px solid var(--turf); outline-offset:2px;
                        border-radius:3px; }
button { font:600 1rem/1 "IBM Plex Sans",sans-serif; padding:.85rem 1.2rem;
         border:none; border-radius:5px; background:var(--turf);
         color:var(--turf-ink); cursor:pointer; }
button:hover { filter:brightness(1.08); }
.hint { font-size:.8rem; color:var(--muted); font-weight:400; }
fieldset { border:1px solid var(--line); border-radius:6px; padding:1rem 1.1rem;
           margin:0; display:flex; flex-direction:column; gap:.9rem; }
legend { font-size:.85rem; font-weight:600; padding:0 .4rem; }
.ou { text-align:center; font-size:.78rem; color:var(--muted);
      text-transform:uppercase; letter-spacing:.1em; }
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

/* --- tableaux (espace du club, nommage) ------------------------------- */
.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:.92rem; }
thead th { text-align:left; font-size:.72rem; text-transform:uppercase;
           letter-spacing:.09em; color:var(--muted); padding:0 .5rem .55rem;
           font-weight:600; }
tbody td { padding:.6rem .5rem; border-top:1px solid var(--line);
           vertical-align:middle; }
tbody tr:first-child td { border-top:none; }
td.date { color:var(--muted); font-variant-numeric:tabular-nums;
          white-space:nowrap; }
td.action { text-align:right; white-space:nowrap; }
td.action a { color:var(--turf); font-weight:600; text-decoration:none; }
td.action a:hover { text-decoration:underline; }
td.jersey { font:600 1.2rem/1 "Barlow Condensed","Arial Narrow",sans-serif;
            color:var(--muted); font-variant-numeric:tabular-nums; width:2.2rem; }
td input[type=text] { width:100%; }
tbody .pastille { display:inline-block; margin-right:.45rem; vertical-align:middle; }
.muet { color:var(--muted); }

/* --- désignation de l'équipe ------------------------------------------ */
.equipes { display:flex; gap:.3rem; }
.equipes form { display:inline; }
button.equipe { font:600 .8rem/1 "IBM Plex Sans",sans-serif; padding:.35rem .6rem;
                border:1px solid var(--line); border-radius:4px;
                background:var(--ground); color:var(--muted); cursor:pointer; }
button.equipe:hover { border-color:var(--turf); color:var(--ink); }
button.equipe.actif { background:var(--turf); border-color:var(--turf);
                      color:var(--turf-ink); }
button.discret { background:none; border:none; color:var(--muted);
                 font:400 .82rem/1 "IBM Plex Sans",sans-serif; padding:.4rem 0;
                 cursor:pointer; text-decoration:underline; }
button.discret:hover { color:var(--bad-ink); }

/* --- tendance de saison ------------------------------------------------ */
section.panel { display:flex; flex-direction:column; gap:.9rem; }
section.panel h2 { font:600 .72rem/1 "IBM Plex Sans",sans-serif;
                   letter-spacing:.14em; text-transform:uppercase;
                   color:var(--muted); margin:0; }
.courbe { margin:0; display:flex; flex-direction:column; gap:.4rem; }
.courbe + .courbe { margin-top:1.2rem; }
.courbe figcaption { font-size:.85rem; color:var(--muted); }
.courbe figcaption b { color:var(--ink); font-size:1rem; }
.courbe svg { width:100%; height:auto; display:block; }
.courbe .mediane { stroke:var(--line); stroke-width:1; }
.courbe .moyenne { stroke-width:1; stroke-dasharray:4 4; opacity:.55; }
.courbe__legende { display:flex; justify-content:space-between;
                   font-size:.75rem; color:var(--muted);
                   font-variant-numeric:tabular-nums; }

/* --- page d'accueil ----------------------------------------------------
   Seule page sans jeton dans l'adresse, donc seule page qu'un visiteur
   atteint sans rien savoir du service. */
.marque { font:600 clamp(2.6rem,9vw,3.6rem)/1 "Barlow Condensed","Arial Narrow",sans-serif;
          margin:0; letter-spacing:.005em; }
.promesse { font-size:1.05rem; }
ol.etapes { margin:0; padding-left:1.2rem; display:flex; flex-direction:column;
            gap:.7rem; }
ol.etapes::marker, ol.etapes li::marker { color:var(--turf); font-weight:600; }
ol.etapes b { display:block; }
ul.recu { margin:0; padding-left:1.1rem; display:flex; flex-direction:column;
          gap:.5rem; }
ul.recu li::marker { color:var(--turf); }
dl.franchise { margin:0; font-size:.92rem; }
dl.franchise dt { font-weight:600; }
dl.franchise dd { margin:.15rem 0 .85rem; color:var(--muted); }
dl.franchise dd:last-of-type { margin-bottom:0; }
h2.depot { font:600 clamp(1.5rem,4vw,1.9rem)/1.1 "Barlow Condensed","Arial Narrow",sans-serif;
           margin:.5rem 0 0; padding-top:1.4rem; border-top:2px solid var(--turf); }

/* --- téléphone ---------------------------------------------------------
   Sous 620 px, cinq colonnes imposent un défilement horizontal. Chaque
   ligne devient une fiche, l'intitulé de colonne étant repris devant la
   valeur — empilée, la ligne n'a plus d'en-tête pour se lire. */
@media (max-width:620px) {
  table, tbody, tr, td { display:block; width:100%; }
  thead { position:absolute; width:1px; height:1px; overflow:hidden;
          clip:rect(0 0 0 0); white-space:nowrap; }
  tbody tr { border:1px solid var(--line); border-radius:6px;
             padding:.7rem .85rem; margin-bottom:.7rem; }
  tbody tr:first-child td { border-top:none; }
  tbody td { border:none; padding:.28rem 0; display:flex;
             justify-content:space-between; align-items:center; gap:1rem; }
  tbody td::before { content:attr(data-champ); color:var(--muted);
                     font-size:.75rem; text-transform:uppercase;
                     letter-spacing:.07em; }
  tbody td.titre { font-weight:600; padding-bottom:.5rem; }
  tbody td.titre::before, tbody td.action::before { content:none; }
  tbody td.action { justify-content:flex-end; padding-top:.5rem; }
}
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


def _document(titre: str, corps: str, tete: str = "",
              indexable: bool = False) -> str:
    """`indexable` n'est vrai que pour la page d'accueil.

    Toutes les autres adresses portent un jeton d'accès. Indexées, elles
    deviendraient publiques : un club qui colle son lien sur un forum
    exposerait ses rapports à quiconque cherche le nom de son club. Le refus
    est donc le défaut, et l'exception s'écrit à l'appel.
    """
    robots = ("index, follow" if indexable else "noindex, nofollow")
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="{robots}">
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
   <input type="text" name="club" required maxlength="80" placeholder="US Valmont"
          autocomplete="organization">
  </label>"""
    return (
        f'  <input type="hidden" name="club_id" value="{html.escape(club.id)}">\n'
        f'  <input type="hidden" name="club_token" value="{html.escape(club.token)}">\n'
        f'  <input type="hidden" name="club" value="{html.escape(club.name)}">'
    )


def admin_page(clubs: list[Club], matchs: list[Job], sante: dict) -> str:
    """Vue d'exploitation.

    Sert d'abord au support : un club qui perd son lien n'a aucun recours, et
    sans cette page il faudrait interroger la base à la main pour le lui
    renvoyer. Les jetons y figurent donc en clair — c'est le seul écran du
    service où c'est le cas, et il est protégé par un jeton distinct.
    """
    par_club: dict[str, list[Job]] = {}
    for m in matchs:
        par_club.setdefault(m.club_id, []).append(m)

    blocs = []
    for club in clubs:
        siens = par_club.get(club.id, [])
        lignes = "".join(
            f'<tr><td data-champ="Match">{html.escape(m.match_name)}</td>'
            f'<td data-champ="Déposé" class="date">{m.created_at[:10]}</td>'
            f'<td data-champ="État"><span class="pastille {LIBELLES[m.state][1]}"></span>'
            f'{ETIQUETTES_COURTES[m.state]}</td>'
            f'<td data-champ="Erreur" class="date">{html.escape(m.error or "—")}</td>'
            f'<td class="action"><a href="{html.escape(m.public_url)}">ouvrir</a></td></tr>'
            for m in siens
        ) or '<tr><td colspan="5" class="muet">aucun match</td></tr>'
        blocs.append(f"""<section class="panel">
  <h2>{html.escape(club.name)} — {len(siens)} match(s)</h2>
  <p class="lien">{html.escape(club.public_url)}</p>
  <div class="scroll"><table><thead><tr><th>Match</th><th>Déposé</th>
   <th>État</th><th>Erreur</th><th></th></tr></thead>
   <tbody>{lignes}</tbody></table></div>
 </section>""")

    alerte = ""
    if sante.get("bloques"):
        alerte = (f'<p class="erreur">{sante["bloques"]} match(s) bloqué(s) : '
                  "le worker est probablement arrêté.</p>")
    elif sante.get("sature"):
        alerte = '<p class="erreur">Disque presque plein : les dépôts sont refusés.</p>'

    corps = f"""
 <header>
  <span class="eyebrow">Exploitation</span>
  <h1>{len(clubs)} clubs, {len(matchs)} matchs</h1>
  <div class="meta">file : {sante.get('en_attente', 0)} en attente ·
   disque libre : {sante.get('disque_libre_go', 0)} Go</div>
 </header>
 {alerte}
 {"".join(blocs) or "<p class='lede'>Aucun club pour l'instant.</p>"}
 <footer>Cette page montre les liens privés des clubs. Ne pas la partager.</footer>"""
    return _document("Exploitation", corps)


def error_page(code: int, message: str) -> str:
    """Page d'erreur lisible par un club.

    Sans elle, une adresse mal recopiée affiche la réponse JSON brute de
    l'API — illisible, et qui donne l'impression d'un service en panne plutôt
    que d'un lien erroné.
    """
    if code == 404:
        titre, aide = "Page introuvable", (
            "Le lien est peut-être incomplet : ces adresses sont longues et se "
            "coupent souvent lorsqu'on les recopie à la main. Vérifiez-le, ou "
            "reprenez celui reçu au dépôt."
        )
    elif code == 409:
        titre, aide = "Analyse en cours", (
            "Ce rapport n'est pas encore prêt. Ouvrez le lien de suivi pour "
            "connaître l'avancement."
        )
    else:
        titre, aide = "Une erreur est survenue", (
            "Réessayez dans quelques instants. Si cela se reproduit, "
            "signalez-le en indiquant l'adresse utilisée."
        )

    corps = f"""
 <header>
  <span class="eyebrow">Erreur {code}</span>
  <h1>{titre}</h1>
 </header>
 <p class="lede">{html.escape(message)}</p>
 <p class="lede">{aide}</p>
 <div class="actions"><a class="secondaire" href="/">Déposer une vidéo</a></div>"""
    return _document(titre, corps)


def _bloc_depot(erreur: str | None = None, club: Club | None = None) -> str:
    """Le formulaire seul, partagé par la page d'accueil et le dépôt rattaché.

    Dupliquer ce balisage les ferait diverger : un champ ajouté d'un côté
    manquerait de l'autre, et aucun test ne le verrait — les deux pages
    rendraient un formulaire valide, simplement différent.
    """
    alerte = f'<p class="erreur">{html.escape(erreur)}</p>' if erreur else ""
    return f"""{alerte}
 <form method="post" action="/matches" enctype="multipart/form-data">
{_champs_formulaire(club)}
  <label>Match
   <input type="text" name="match_name" required maxlength="120"
          placeholder="US Valmont – AS Beaupré, 30 août">
  </label>
  <label>Date du match <span class="option">facultatif</span>
   <input type="date" name="played_on" max="2100-12-31">
   <span class="hint">Celle de la rencontre, pas celle du dépôt. Laissée vide,
   aucune date n'apparaît sur le rapport.</span>
  </label>
  <fieldset>
   <legend>La vidéo</legend>
   <label>Lien vers la vidéo
    <input type="url" name="source_url" inputmode="url"
           placeholder="https://www.youtube.com/watch?v=...">
    <span class="hint">Le plus simple si votre match est déjà en ligne : le
    lien part en une seconde, nous téléchargeons depuis nos serveurs.</span>
   </label>
   <p class="ou">ou</p>
   <label>Fichier vidéo
    <input type="file" name="video" accept="video/*">
    <span class="hint">MP4, MOV, AVI ou MKV. 8 Go maximum, soit environ 2 h en
    1080p. Comptez du temps sur une connexion mobile.</span>
   </label>
  </fieldset>
  <label>Adresse e-mail <span class="option">facultatif</span>
   <input type="email" name="contact" maxlength="120" placeholder="entraineur@club.fr"
          autocomplete="email" inputmode="email">
   <span class="hint">Pour être prévenu quand le rapport est prêt. Le message
   contient le lien d'accès : vérifiez l'adresse, toute personne qui reçoit ce
   lien peut lire le rapport.</span>
  </label>
  <button type="submit">Envoyer la vidéo</button>
 </form>"""


def upload_form(erreur: str | None = None, club: Club | None = None) -> str:
    titre = "Déposer un match" if club else "Déposer une vidéo"
    corps = f"""
 <header>
  <span class="eyebrow">{html.escape(club.name) if club else "Analyse de match"}</span>
  <h1>{titre}</h1>
 </header>
 <p class="lede">Vous recevez un lien à conserver. L'analyse dure environ une
 heure ; vous pouvez fermer cette page.</p>
 {_bloc_depot(erreur, club)}
 <p class="note">Filmez depuis un point haut et reculé : cela double la part du
 match réellement analysable, bien plus que n'importe quel réglage de notre côté.</p>
 <footer>La vidéo est supprimée de nos serveurs dès le rapport produit.</footer>"""
    return _document(titre, corps)


def home_page(erreur: str | None = None) -> str:
    """Page d'accueil : la seule qu'un visiteur atteint sans rien savoir.

    Elle porte le formulaire en bas plutôt que derrière un lien. Un club de
    village décide en une page ou pas du tout, et un clic de plus entre la
    promesse et le champ suffit à le perdre.

    L'argumentaire ne dit que du mesuré. Un rapport d'analyse automatique est
    cru sur parole : promettre ici ce que le rapport ne tient pas se paierait
    à la première lecture.
    """
    description = ("Déposez la vidéo de votre match, recevez possession, "
                   "contrôle du terrain et distance parcourue par joueur. "
                   "Sans compte ni logiciel.")
    corps = f"""
 <header>
  <span class="eyebrow">Analyse vidéo de match</span>
  <h1 class="marque">{html.escape(PRODUIT)}</h1>
 </header>
 <p class="promesse">Vous filmez le match, vous déposez la vidéo. Une heure
 plus tard, vous savez ce que la rencontre a réellement produit : la
 possession, le contrôle du terrain, et les kilomètres de chaque joueur.</p>
 <div class="actions"><a class="principal" href="#deposer">Déposer un match</a></div>

 <section class="panel">
  <h2>Ce que vous recevez</h2>
  <ul class="recu">
   <li>Un <b>rapport</b> lisible sur téléphone et imprimable : possession,
   contrôle du terrain, distance et vitesse de pointe joueur par joueur.</li>
   <li>La <b>vidéo annotée</b>, chaque joueur suivi et rattaché à son équipe.</li>
   <li>Une <b>vue du dessus</b> du placement des deux équipes.</li>
   <li>Le <b>relevé en tableur</b>, pour les clubs qui tiennent leurs chiffres.</li>
   <li>Un <b>espace de club</b> : tous vos matchs derrière un lien, et la
   tendance de la saison match après match.</li>
  </ul>
 </section>

 <section class="panel">
  <h2>Comment ça marche</h2>
  <ol class="etapes">
   <li><b>Vous déposez.</b> Le lien de votre match s'il est déjà en ligne,
   sinon le fichier. Ni compte à créer, ni logiciel à installer.</li>
   <li><b>Nous analysons.</b> Environ une heure. Fermez la page : vous gardez
   un lien, et un e-mail vous prévient si vous en laissez un.</li>
   <li><b>Vous nommez vos joueurs.</b> Le rapport remplace alors les numéros
   par les noms de votre effectif.</li>
  </ol>
 </section>

 <p class="note"><b>Ce qui pèse le plus, c'est votre caméra.</b> Mesuré sur
 deux matchs de la même équipe : selon le seul point de vue, la part du match
 réellement analysable passe de 47 % à plus de 92 %. Filmez d'un point haut et
 reculé, en un plan large, sans zoom brusque. Aucun réglage de notre côté ne
 rattrape cela.</p>

 <section class="panel">
  <h2>Ce que nous ne promettons pas</h2>
  <dl class="franchise">
   <dt>Un match mesuré de bout en bout.</dt>
   <dd>Chaque rapport affiche d'abord la part du match qu'il a réellement
   mesurée. Les totaux sont des planchers, jamais extrapolés : une distance
   affichée est toujours inférieure ou égale à la réalité.</dd>
   <dt>Des distances au mètre près.</dt>
   <dd>Ce sont des ordres de grandeur, et le rapport le dit. Nous préférons
   l'écrire plutôt que d'afficher une décimale qui ferait sérieux.</dd>
   <dt>De garder vos images.</dt>
   <dd>La vidéo est supprimée de nos serveurs dès le rapport produit. Ce sont
   les images de votre club.</dd>
   <dt>Un compte et un mot de passe.</dt>
   <dd>Votre lien est votre clé : conservez-le, il est le seul accès. Toute
   personne à qui vous le donnez peut lire le rapport.</dd>
  </dl>
 </section>

 <h2 class="depot" id="deposer">Déposer un match</h2>
 <p class="lede">Vous recevez un lien à conserver. L'analyse dure environ une
 heure ; vous pouvez fermer cette page.</p>
 {_bloc_depot(erreur)}
 <footer>Le service est gratuit pendant sa mise au point.
 La vidéo est supprimée de nos serveurs dès le rapport produit.</footer>"""
    return _document(
        f"{PRODUIT} — analyse vidéo de match", corps,
        tete=f'\n<meta name="description" content="{html.escape(description)}">',
        indexable=True,
    )


def _selecteur_equipe(club: Club, job: Job) -> str:
    """Deux boutons : laquelle des deux équipes du rapport est celle du club.

    Formulaire et non lien : désigner une équipe modifie un état, et un lien
    cliqué par un aspirateur de pages le changerait à l'insu du club.
    """
    boutons = []
    for equipe, libelle in ((0, "A"), (1, "B")):
        choisie = job.our_team == equipe
        actif = " actif" if choisie else ""
        # Un lecteur d'écran annoncerait « bouton A » : ni de quel match il
        # s'agit, ni ce que le clic ferait.
        titre = (f"{'Retirer' if choisie else 'Désigner'} l'équipe {libelle} "
                 f"comme celle du club pour {job.match_name}")
        boutons.append(
            f'<form method="post" action="{html.escape(club.public_url)}'
            f'/match/{html.escape(job.id)}/equipe">'
            f'<input type="hidden" name="team" value="{equipe}">'
            f'<button class="equipe{actif}" type="submit"'
            f' aria-pressed="{"true" if choisie else "false"}"'
            f' aria-label="{html.escape(titre)}">{libelle}</button></form>'
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
                f'<tr><td data-champ="Match" class="titre">'
                f"{html.escape(m.match_name)}</td>"
                f'<td data-champ="Déposé" class="date">{date}</td>'
                f'<td data-champ="État"><span class="pastille {classe}"></span>{libelle}</td>'
                f'<td data-champ="Votre équipe">{notre}</td>'
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


# Une piste vue moins que cette part du temps mesuré n'est pas proposée au
# nommage : c'est un fragment, pas un joueur, et l'entraîneur ne saurait pas
# lequel désigner.
PART_MINIMALE = 0.10


def nommables(stats: dict, part_minimale: float = PART_MINIMALE) -> list[dict]:
    """Pistes assez durables pour qu'un entraîneur les reconnaisse."""
    joueurs = stats.get("players") or []
    if not joueurs:
        return []
    plus_longue = max(j["seconds_seen"] for j in joueurs)
    seuil = plus_longue * part_minimale
    retenus = [j for j in joueurs if j["seconds_seen"] >= seuil]
    return sorted(retenus, key=lambda j: -j["seconds_seen"])


def naming_page(job: Job, joueurs: list[dict], effectif: Sequence[str] = ()) -> str:
    """Formulaire de nommage des joueurs d'un match.

    `effectif` est la liste des noms déjà saisis par le club. Elle sert de
    suggestions à la frappe, jamais de pré-remplissage : aucune piste n'est
    reliée d'un match au suivant, et un nom placé d'office serait une
    affirmation que rien ne soutient.
    """
    if effectif:
        liste = ('<datalist id="effectif">'
                 + "".join(f'<option value="{html.escape(n)}">' for n in effectif)
                 + "</datalist>")
        suggestions = ' list="effectif"'
        rappel = ("<p class='lede'>Les noms déjà donnés par le club sont proposés "
                  "dès les premières lettres. Ils ne sont pas placés d'avance : "
                  "rien ne relie une piste d'un match à celle du suivant, et "
                  "c'est vous qui reconnaissez le joueur.</p>")
    else:
        liste = suggestions = rappel = ""

    if not joueurs:
        lignes = ("<p class='lede'>Aucune piste assez suivie pour être nommée "
                  "sur ce match.</p>")
    else:
        rangs = []
        for j in joueurs:
            equipe = j.get("team")
            couleur = TEAM_HEX_A if equipe == 0 else TEAM_HEX_B if equipe == 1 else "#a9b0a9"
            valeur = html.escape(job.player_names.get(str(j["track_id"]), ""))
            rangs.append(f"""<tr>
   <td class="jersey">{j["track_id"]}</td>
   <td><span class="pastille" style="background:{couleur}"></span></td>
   <td class="date">{j["distance_m"] / 1000:.1f} km · {j["seconds_seen"] / 60:.0f} min</td>
   <td><input type="text" name="nom_{j["track_id"]}" value="{valeur}"
              maxlength="60" placeholder="Nom du joueur"{suggestions}></td>
  </tr>""")
        lignes = ('<div class="scroll"><table><thead><tr><th>N°</th><th></th>'
                  "<th>Relevé</th><th>Nom</th></tr></thead><tbody>"
                  + "".join(rangs) + "</tbody></table></div>")

    corps = f"""
 <header>
  <span class="eyebrow">{html.escape(job.club)}</span>
  <h1>Nommer les joueurs</h1>
 </header>
 <p class="lede">Les numéros sont attribués automatiquement et ne
 correspondent pas aux maillots. Repérez chaque joueur dans la vidéo annotée,
 puis inscrivez son nom ici : il remplacera le numéro dans le rapport.</p>
 {rappel}
 <form method="post" action="{html.escape(job.public_url)}/joueurs">
  {liste}
  <div class="panel">{lignes}</div>
  <button type="submit">Enregistrer les noms</button>
 </form>
 <div class="actions">
  <a class="secondaire" href="{html.escape(job.public_url)}/video">Vidéo annotée</a>
  <a class="secondaire" href="{html.escape(job.public_url)}/rapport">Rapport</a>
 </div>"""
    return _document(f"Nommer — {job.match_name}", corps)


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
   <a class="secondaire" href="{html.escape(job.public_url)}/joueurs">Nommer les joueurs</a>
   <a class="secondaire" href="{html.escape(job.public_url)}/releve.csv">Tableur</a>
  </div>
  <form method="post" action="{html.escape(job.public_url)}/supprimer"
        onsubmit="return confirm('Supprimer ce match et son rapport ? Cette action est définitive.')">
   <button class="discret" type="submit">Supprimer ce match</button>
  </form>"""
    elif job.state is JobState.FAILED:
        detail = f"<p class='erreur'>{html.escape(job.error or 'Cause inconnue.')}</p>"
        actions = '<div class="actions"><a class="secondaire" href="/">Déposer à nouveau</a></div>'
    elif job.state is JobState.PROCESSING:
        pourcent = int(job.progress * 100)
        detail = (
            f'<div class="jauge" role="progressbar" aria-valuemin="0"'
            f' aria-valuemax="100" aria-valuenow="{pourcent}"'
            f' aria-label="Avancement de l\'analyse">'
            f'<span style="width:{pourcent}%"></span></div>'
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
