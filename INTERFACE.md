# Interface : forme du produit, écrans, design

## Un site web, pas une application

Le club ouvre une adresse, dépose sa vidéo, reçoit un lien. Rien à installer,
aucun compte à créer.

Ce choix n'est pas neutre, il découle de trois contraintes du marché visé :

- **Le traitement demande un GPU.** Une application de bureau obligerait chaque
  club à posséder une carte graphique à plusieurs centaines d'euros. Le calcul
  reste donc chez nous, et le club n'envoie qu'une vidéo.
- **Une analyse dure une heure.** Le club dépose puis ferme la page. Une
  application qui doit rester ouverte pendant une heure serait abandonnée.
- **Les clubs visés n'ont pas d'informaticien.** Une installation, une mise à
  jour ou un pilote manquant suffiraient à perdre le client.

Une application mobile viendra peut-être, mais elle ne changerait rien au
calcul : elle ne serait qu'une autre façon d'atteindre le même service.

## Les quatre écrans

| Écran | Adresse | Rôle |
|---|---|---|
| Accueil | `/` | présentation du service, puis le dépôt en bas de page |
| Dépôt | `/` (bas de page) | club, match, date, vidéo ou lien, e-mail |
| Confirmation | après envoi | montre le lien privé, rappelle de le garder |
| Suivi | `/m/{id}/{jeton}` | état, jauge de progression, mise à jour seule |
| Rapport | `/m/{id}/{jeton}/rapport` | le livrable, chiffres et vue tactique |
| Nommage | `/m/{id}/{jeton}/joueurs` | associer un nom à chaque piste |
| Espace du club | `/c/{id}/{jeton}` | tous les matchs du club, un seul lien |
| Dépôt rattaché | `/c/{id}/{jeton}/deposer` | le match rejoint l'espace |

Plus la vidéo annotée en téléchargement, et `/sante` pour la supervision.

Le code est dans `service/web/pages.py` pour les trois premiers, et dans
`football_analysis/report.py` pour le rapport. Tout est rendu côté serveur :
pas de framework, pas d'étape de compilation, une page qui s'affiche même sur
un téléphone d'entrée de gamme et une connexion lente.

## Le système visuel

Un seul système pour les quatre écrans, afin que le dépôt et le rapport se
lisent comme un même produit.

**Couleurs.** Blanc cassé biaisé vert pelouse en fond (`#f7f8f5`), encre
`#151a16`, vert terrain `#2f6d43` pour les accents. Les équipes prennent un
bleu `#0080ff` et un orange brûlé `#d4622a`, distinguables en niveaux de gris
pour les clubs qui impriment. Treize à quinze jetons de couleur, jamais de
valeur codée en dur dans un composant.

**Typographie.** Barlow Condensed pour les titres et les chiffres — c'est la
lettre des maillots et des tableaux d'affichage. IBM Plex Sans pour le texte,
IBM Plex Mono pour les mesures, qui doivent s'aligner en colonnes.

**Thème clair et sombre.** Les deux sont définis, y compris l'état par défaut
où le navigateur n'annonce rien.

**Conventions du football.** La possession et le contrôle s'affichent en barre
unique divisée, comme à la télévision. Deux barres séparées obligeraient à
comparer deux longueurs ; une barre divisée montre le rapport directement.

## La tendance de saison

L'espace du club trace la possession et le contrôle du terrain match après
match, en SVG écrit à la main — une bibliothèque de graphiques pèserait plus
lourd que la page entière pour six points lus sur un téléphone.

Une difficulté a dû être résolue avant de pouvoir tracer quoi que ce soit.
Dans un rapport, « équipe A » et « équipe B » sont des étiquettes issues d'un
regroupement automatique : rien ne garantit que l'équipe A d'un match soit la
même que celle du match suivant. Comparer ces chiffres d'une rencontre à
l'autre aurait produit une courbe qui s'inverse au hasard.

Le club désigne donc son équipe, d'un bouton, sur chaque match analysé. Les
matchs sans désignation sont exclus de la tendance plutôt qu'inclus au hasard,
et la page explique pourquoi. Recliquer sur l'équipe déjà choisie l'efface —
seul moyen de corriger une erreur.

Les distances individuelles ne sont pas agrégées : à la précision actuelle,
leur imprécision se cumulerait au lieu de se compenser.

## Déposer par lien plutôt que par fichier

Un match de 90 minutes pèse plusieurs gigaoctets. Le téléverser depuis une
connexion mobile — la norme du marché visé — prend des heures et échoue à la
moindre coupure, sans reprise possible.

Beaucoup de clubs publient déjà leurs matchs sur une plateforme vidéo. Le
formulaire accepte donc un lien : il part en une seconde, et le téléchargement
se fait depuis nos serveurs, sur une liaison stable.

Le téléchargement a lieu dans le worker et non au dépôt : il peut durer, et
bloquer une requête HTTP pendant ce temps la ferait expirer.

Comme c'est le service qui télécharge, c'est le service qu'on peut faire
pointer n'importe où. L'adresse est donc résolue et vérifiée avant tout
téléchargement : un lien vers `169.254.169.254` exposerait les identifiants
d'accès de la machine chez la plupart des hébergeurs. C'est l'adresse obtenue
qui décide, pas le nom — un nom public peut pointer vers le réseau interne.

## Les noms de joueurs

Les numéros affichés viennent du traqueur et ne correspondent pas aux
maillots : un entraîneur ne reconnaît pas « joueur 17 ». Il repère chaque
joueur dans la vidéo annotée, saisit son nom, et le rapport est réécrit.

Seules les pistes durables sont proposées — au moins un dixième du temps de la
plus longue. Nommer un fragment de trois secondes ajouterait du bruit au lieu
d'en retirer, et l'entraîneur ne saurait pas lequel désigner.

Le numéro reste affiché à côté du nom : c'est lui qui figure dans la vidéo
annotée, et le club doit pouvoir faire le lien.

**L'effectif du club est proposé à la frappe.** Les noms déjà saisis sur les
matchs précédents du même espace de club reviennent en suggestions dès les
premières lettres. Ressaisir vingt noms à chaque match est le premier motif
d'abandon du nommage, alors qu'un effectif change de deux ou trois joueurs
par saison.

Ils ne sont **jamais placés d'office**. Rien ne relie une piste d'un match à
celle du suivant : le numéro 4 d'aujourd'hui n'est pas celui de la semaine
dernière. Pré-remplir affirmerait un lien que rien ne soutient, et le club
signerait sans le savoir un rapport attribuant à un joueur la course d'un
autre. La suggestion supprime la frappe, pas la reconnaissance.

L'effectif ne franchit pas la frontière d'un espace de club, et un match
déposé seul n'en a aucun : deux dépôts séparés n'ont pas de lien démontrable,
et les rapprocher sur la seule ressemblance du nom de club livrerait
l'effectif d'un club à un autre.

## Ce que l'interface dit, et pourquoi

L'enjeu n'est pas décoratif. Un rapport d'analyse automatique est cru sur
parole : l'interface doit donc empêcher un club de se tromper.

- **La date affichée est celle du match**, saisie par le club, jamais celle de
  l'analyse. Un club dépose souvent une rencontre jouée des semaines plus tôt,
  et le rapport affichait la date du jour. Sans saisie, aucune date n'est
  montrée : mieux vaut rien qu'une date fausse.
- **La couverture s'affiche avant les chiffres.** Un entraîneur doit savoir sur
  quelle part du match ils portent avant de les lire, pas après.
- **Les distances sont annoncées comme des ordres de grandeur.** Mesuré : une
  erreur de 5 m sur la position multiplie par seize la longueur d'un pas de
  course, quand elle ne décale le contrôle du terrain que de deux points.
- **Les erreurs parlent au club, pas au développeur.** Jamais de « CUDA out of
  memory » ni de chemin interne, toujours une cause et une action. Une adresse
  mal recopiée affiche une page, pas la réponse JSON de l'API — et suggère la
  cause la plus fréquente, un lien coupé à la copie.
- **Aucune page n'est indexable.** Les adresses portent un jeton d'accès :
  indexées, elles deviendraient publiques. `noindex` sur chaque page et un
  `robots.txt` qui interdit tout le site.
- **Un match peut être supprimé** par le club lui-même, fichiers compris. Sans
  ce bouton, une vidéo déposée par erreur imposerait de nous écrire.
- **Une page en attente n'affiche pas de jauge à 0 %**, qui laisserait croire à
  un blocage alors que rien n'a commencé.

## La page d'accueil

`/` présentait directement le formulaire. Un visiteur qui arrivait sans rien
savoir tombait sur un champ « Nom du club » et repartait.

La page présente maintenant le service, et **porte le formulaire en bas d'elle-
même** plutôt que derrière un lien : un club de village décide en une page ou
pas du tout, et un clic entre la promesse et le champ suffit à le perdre. Le
formulaire n'est écrit qu'une fois (`_bloc_depot`) et sert aussi au dépôt
rattaché à un espace de club — dupliqué, il aurait divergé sans que rien ne le
signale.

Trois choix de fond :

- **L'argumentaire ne dit que du mesuré.** Le 47 % → 92 % selon la captation
  vient d'`ANALYSE_TERRAIN.md`. Un rapport d'analyse automatique est cru sur
  parole : promettre ici ce que le rapport ne tient pas se paierait à la
  première lecture.
- **Une section « ce que nous ne promettons pas ».** Couverture partielle,
  distances données en ordre de grandeur, lien unique sans compte. Un club qui
  découvre seul une limite doute de tout le reste ; annoncée d'avance, elle
  devient une preuve de sérieux.
- **C'est la seule page indexable du service.** Toutes les autres portent un
  jeton d'accès et gardent leur `noindex` ; `robots.txt` n'autorise que la
  racine exacte. `_document` refuse l'indexation par défaut, l'accueil est
  l'exception écrite à l'appel — l'inverse aurait fini par exposer une page à
  jeton.

Le nom du produit vit dans `settings.PRODUIT`, en un seul endroit et
surchargeable par variable d'environnement. Il est **provisoire**.

## Ce qui n'existe pas encore

Assumé pour l'instant, à traiter quand un client le demandera :

- **Pas d'identité visuelle arrêtée.** Le nom est provisoire et il n'y a pas
  de logo. La page d'accueil existe, la marque non.
- **Aucune personnalisation par club.** Couleurs de maillot, logo sur le
  rapport.
- **Pas de facturation** : le service est gratuit et ouvert à quiconque a
  l'adresse.

## Ordre proposé

1. ~~**Report des noms d'un match sur l'autre.**~~ Fait, sous la seule forme
   honnête : l'effectif du club est suggéré à la frappe. Le report complet
   demanderait de relier les pistes entre matchs, ce que rien ne permet —
   ni le traqueur, ni l'apparence, les maillots étant identiques.
2. **Identité visuelle**, quand le produit aura un nom.
3. **Facturation**, quand le modèle économique sera arrêté.
