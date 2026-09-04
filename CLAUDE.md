# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
.venv/bin/python -m football_analysis.cli input_video/match.mp4 -o output_video/out.mp4
.venv/bin/python -m pytest tests -q          # suite complète
.venv/bin/python -m pytest tests/test_analytics.py::test_ball_rejects_impossible_jump
```

Un venv existe déjà en `.venv/` avec numpy, opencv, supervision, trackers et pytest —
soit de quoi faire tourner les tests. Les dépendances lourdes (`torch`,
`transformers`, `ultralytics`, `umap-learn`) ne sont **pas** installées : les tests
les évitent par des doublures, mais le pipeline réel en a besoin
(`pip install -r requirements.txt`).

### Runtime directories are required but not in git

`main.py` reads and writes paths that do not exist in a fresh clone and are not gitignored:

- `input_video/08fd33_4.mp4` — source clip
- `models/best_telecharger.pt` — YOLO weights produced by `training/footbal_training_yolo_v5.ipynb`
- `stubs/track_stubs.pkl` — cached tracking results
- `output_video/output_video.avi` — annotated result

Create these directories (and supply the video + weights) before running, or `main.py` fails.

### Entraînement

```bash
python -m training.train_detection --data datasets/players/data.yaml
python -m training.train_keypoints --write-config datasets/pitch   # génère data.yaml
python -m training.train_keypoints --data datasets/pitch/data.yaml
```

Pour le modèle de points clés, `training/train_keypoints_colab.ipynb` fait tout sur
GPU gratuit : téléchargement du dataset, vérification de l'ordre des points, correction
des chemins du `data.yaml` livré par Roboflow (ses chemins relatifs ne se résolvent pas
sous Ultralytics), entraînement, puis un test d'acceptation exprimé en centimètres
d'erreur au sol plutôt qu'en mAP.

Le notebook **duplique** la géométrie du terrain, Colab n'ayant pas le dépôt.
`tests/test_colab_notebook.py` vérifie que cette copie n'a pas dérivé de
`pitch/geometry.py` — une divergence donnerait un modèle entraîné contre une autre
géométrie, sans erreur visible.

Le notebook historique `training/footbal_training_yolo_v5.ipynb` pulls the Roboflow `football-players-detection-3zvbc` dataset,
then `shutil.move`s `train/`, `valid/`, `test/` one level deeper — which is why the repo has the doubled
`training/football-players-detection-1/football-players-detection-1/` path. That layout is what makes the
relative paths in `data.yaml` resolve; don't "flatten" it. Training runs via
`yolo task=detect mode=train model=yolov5.pt data=.../data.yaml epochs=10 imgsz=640`.

Dataset classes (`data.yaml`): `ball`, `goalkeeper`, `player`, `referee` (nc=4).

## Architecture

Tout le code vit dans le package `football_analysis/`. L'ancien prototype
(`main.py`, `trackers/`, `utils/`) a été supprimé : le dossier `trackers/` masquait
le paquet PyPI du même nom, dont dépend désormais le suivi.

```
config.py              tous les chemins et seuils réglables (rien en dur ailleurs)
video/io.py            lecture/écriture en flux (générateurs, pas de liste de frames)
detection/detector.py  YOLOv8 + NMS agnostique de classe, par lots
tracking/tracker.py    BoT-SORT (paquet `trackers`) ; le ballon en est exclu
analytics/stats.py     distance, vitesse, possession -> JSON
report.py              rapport HTML remis au club (le livrable réel)
pipeline.py            orchestration bout en bout
cli.py                 point d'entrée
teams/classifier.py    SigLIP (768-D) -> UMAP (3-D) -> K-Means (2 clusters)
pitch/geometry.py      32 sommets du terrain en cm — contrat avec le modèle pose
pitch/view.py          homographie bidirectionnelle + lissage fenêtre glissante
analytics/ball.py      rejet des sauts > 5 m, puis moyenne glissante
analytics/voronoi.py   zones de contrôle par équipe, sur grille du terrain
```

Décisions structurantes, non évidentes à la lecture d'un seul fichier :

- **La vidéo livrée est en H.264, 1280 px de large.** Mesuré sur des images
  réelles : le codec mp4v d'OpenCV en 1080p donne 3,3 Go pour un match de 90
  minutes, intéléchargeable sur la connexion mobile qui est la norme du marché
  visé ; H.264 en 720 p donne 0,38 Go, et les numéros de maillot restent nets.
  OpenCV ne sait pas encoder en H.264 dans les distributions courantes — `avc1`
  échoue silencieusement et laisse un fichier vide — d'où le passage par
  ffmpeg, avec repli sur mp4v s'il est absent : un fichier lourd vaut mieux
  qu'une analyse de quarante minutes perdue.
- **Tout est en flux.** `video/io.py` expose des générateurs. Le prototype chargeait
  la vidéo entière en mémoire, ce qui plafonnait la durée traitable.
- **Le pipeline sous-échantillonne** à `ProcessingConfig.sample_fps` (12 fps,
  mesuré sur extrait réel : 15 identités à 25 fps, 14 à 12,5, puis 20 à 5 et 22
  à 2 — descendre fragmente les pistes sans économie proportionnelle). Mesuré : à 11 km parcourus par match, descendre de 25 à 5 fps ne
  sous-estime la distance que de 0,3 %. Ce n'est pas la mesure qui fixe cette
  cadence mais le suivi, qui a besoin de recouvrement entre images. Deux
  conséquences faciles à manquer : la vidéo annotée s'écrit à la cadence réduite
  pour se lire à la bonne vitesse, et `TrackingConfig.frame_rate` doit suivre —
  sinon les pistes expirent `stride` fois trop tard.
- **Le ballon ne passe pas par le traqueur.** Trop petit et trop rapide pour que
  l'IoU entre deux frames soit non nul. Sa position vient des détections brutes, filtrée
  par `BallTrajectory` dans l'espace terrain — c'est là qu'une contrainte physique
  ("pas plus de 5 m par frame") est exprimable.
- **Les gardiens gardent leur classe.** L'ancien prototype les réécrivait en
  `player` avant le suivi. On ne peut alors plus appliquer l'heuristique de
  proximité au centroïde qui décide de leur équipe, puisque SigLIP les classe
  arbitrairement (maillot différent).
- **Le classifieur d'équipes s'ajuste une seule fois** sur un échantillon de frames
  du début de match. Réajuster UMAP par frame donne des clusters qui permutent d'une
  frame à l'autre.
- **K-Means tourne avec `n_teams + 1` clusters.** Les deux plus peuplés sont les
  équipes, le reste est `UNASSIGNED`. Sans ce cluster supplémentaire, les arbitres
  sont assignés de force à une équipe — vérifié sur match réel. Un rejet par
  distance au centroïde a été essayé avant et n'écartait rien : présents à
  l'ajustement, les arbitres tombent dans la dispersion normale.
- **Le rapport est régénérable sans la vidéo.** `report.write` ne relit que le
  JSON de statistiques, ce qui permet à l'API de le réécrire quand le club
  nomme ses joueurs. `report.py` ne dépend d'aucune bibliothèque de calcul :
  l'importer depuis l'API ne casse pas la séparation vérifiée par
  `test_the_api_does_not_load_the_machine_learning_stack`.
- **Les pistes sont recollées avant publication.** `tracking/stitching.py`
  rattache deux segments d'un même joueur — pas de chevauchement temporel,
  écart bref, déplacement physiquement possible dans l'espace du terrain, même
  équipe. Sans cela, un joueur perdu puis retrouvé figure deux fois avec la
  moitié de sa distance chacune. `Stitcher.observe` attend l'index d'**image
  traitée**, jamais celui de la source : mélanger les deux unités double
  silencieusement tous les écarts et empêche tout recollement.
- **Le trajet entre deux segments n'est jamais comblé.** Il n'a pas été
  observé ; l'inventer fabriquerait la donnée que le recollement doit rendre
  crédible.
- **Une détection sans identifiant de piste ne fait pas une identité.** BoT-SORT
  rend -1 tant qu'une piste n'est pas confirmée ; `MatchStats.update_player`
  refuse désormais les identifiants négatifs. Les accepter fusionnait tous les
  joueurs non suivis en une identité unique, cumulant leurs distances : sur un
  extrait de 24 s elle affichait 123,5 s de présence et la plus grande distance
  du match. Filtrer avec `tracking.tracker.is_tracked`.
- **`UNASSIGNED` vaut -1, ne jamais l'utiliser comme index.** `-1 % 2` vaut 1 en
  Python : un joueur non attribué serait peint aux couleurs de l'équipe B. Passer
  par `annotators.team_color`.
- **L'ordre des 32 keypoints** de `pitch/geometry.py` est le contrat avec les poids
  YOLOv8-pose. Le modifier invalide tout modèle déjà entraîné. Vérifié identique au
  dataset public `football-field-detection-f07vi` v15 (CC BY 4.0), y compris
  `flip_idx`, sans qu'aucune valeur n'ait été recopiée.
- **Les dimensions du terrain sont mesurées, pas conventionnelles.** 105 × 68 m avec
  les cotes de la loi du jeu minimisent l'erreur de reprojection sur 228 images
  annotées (0,398 % contre 0,960 % pour la convention Roboflow 120 × 70). Utiliser
  cette dernière gonflerait toute distance de 14 %. Les dimensions restent réglables
  par club — un terrain réel fait 100 à 110 m.
- **Le détecteur est derrière un protocole** (`detection/base.py`). Ultralytics est
  en AGPL-3.0 ; la migration vers un modèle Apache-2.0 doit se limiter à une
  nouvelle classe, sans toucher au pipeline.
- **`sv.ByteTrack` est déprécié** (supprimé en supervision 0.31). On utilise le
  paquet `trackers`, avec BoT-SORT et compensation du mouvement caméra activée —
  sans CMC, un panoramique déplace toutes les boîtes et fait perdre les identités.
- **La confiance de boîte du modèle de terrain n'est pas un indicateur de
  qualité.** Elle varie de 0,05 à 0,89 sur des images où les points restent bons.
  `instance_confidence` est donc à 0,02 et le filtrage se fait sur les points.
  Le seuil par défaut d'Ultralytics divisait par deux le taux d'images exploitables.
- **Le pipeline dégrade proprement.** Une frame sans keypoints exploitables réutilise
  la dernière homographie valide ; sans homographie du tout, la frame est annotée
  mais n'alimente ni le radar ni les statistiques spatiales. C'est le cas normal en
  caméra basse, pas une erreur.
- **`flip_index` est calculé, pas écrit à la main.** YOLO-pose a besoin de
  `flip_idx` pour l'augmentation `fliplr` : sans lui, l'image est mirorée mais pas
  les labels, et le coin haut gauche garde l'indice du coin haut droit. Panne
  silencieuse — l'entraînement converge quand même. `tests/test_training_config.py`
  vérifie la cohérence entre le data.yaml et la géométrie.
- **L'homographie se périme.** `HomographyCache` la refuse au-delà de
  `homography_max_age_s` (2 s par défaut). Réutiliser indéfiniment la dernière
  valide était le comportement d'origine ; sur vidéo réelle, des trous de 12
  minutes ont été mesurés, pendant lesquels la caméra panoramique plusieurs fois
  et les positions projetées dérivent silencieusement. Les frames concernées sont
  marquées non mesurées, et `MatchStats.mark_unmeasured` oublie les dernières
  positions connues — sans quoi la reprise compterait le trajet du trou entier
  comme une course.
- **Les totaux sont plancher, jamais extrapolés.** `coverage` dit quelle part du
  match a été mesurée. Une distance affichée est toujours inférieure ou égale à
  la réalité, ce que le rapport indique au club.
- **Le rapport ne cache pas ses angles morts.** `report._caveats` déduit des données
  elles-mêmes ce qui doit être signalé au club (terrain non localisé, identités
  fragmentées). Un club qui repère seul une incohérence perd confiance dans le reste.
- **Mosaic doit être à 0** à l'entraînement du modèle pose : cette augmentation colle
  plusieurs images ensemble et apprend au modèle à chercher plusieurs terrains.

## Couche de livraison (`service/`)

```
jobs.py      états d'un match, stockage SQLite, réservation atomique
storage.py   dépôt des vidéos, formats et taille acceptés
worker.py    consomme la file, produit le rapport
api.py       trois écrans : déposer, suivre, lire
web/pages.py rendu serveur, palette et polices communes au rapport
```

Lancement : `uvicorn service.api:app` d'un côté, `python -m service.run_worker`
de l'autre. Ils ne partagent que `FA_DATA_ROOT` — l'API tient sur une petite
machine, le worker a besoin d'un GPU qu'on veut pouvoir éteindre. Déploiement
en deux images distinctes : `Dockerfile` (cibles `api` et `worker`),
`docker-compose.yml`, détails dans `DEPLOIEMENT.md`.

**L'API ne doit jamais charger torch ni OpenCV.** `service/settings.py` porte
les constantes partagées et `worker.py` importe le pipeline tardivement, ce qui
permet une image API de quelques dizaines de mégaoctets au lieu de plusieurs
gigaoctets. Deux tests le vérifient en sous-processus ; y ajouter un import
direct du pipeline dans `api.py` ou en tête de `worker.py` les casse.

Décisions structurantes :

- **Pas de comptes.** Le lien porte un jeton de 24 octets ; `authenticate`
  compare avec `secrets.compare_digest`, un `==` laissant deviner le jeton
  caractère par caractère au chronomètre. Un jeton faux et un match inexistant
  renvoient le même 404 : les distinguer confirmerait qu'un match existe.
- **`claim_next` est atomique** — `UPDATE ... WHERE state='queued'` — pour que
  deux workers ne traitent pas le même match. Ne jamais s'en servir pour tester
  si la file est vide : utiliser `pending_count`, sinon le job suivant reste
  bloqué en « en cours ».
- **La vidéo source est supprimée** dès le rapport produit : poste de stockage
  dominant, et ce sont les images du club.
- **Les erreurs sont traduites** par `worker._message_lisible`. Un club ne doit
  jamais lire « CUDA out of memory » ni un chemin interne.
- **La notification ne peut pas faire échouer une analyse.** `worker._prevenir`
  avale toute exception d'envoi : le rapport existe et le lien fonctionne, seul
  l'avis manque. Le contact est facultatif, et une adresse invalide est refusée
  au dépôt plutôt qu'ignorée — sinon le club attendrait un message qui ne
  viendrait jamais. `notify.Notifier` est un protocole : un envoi WhatsApp se
  branchera sans toucher au reste.
- **Les matchs abandonnés sont repris.** Un worker tué laisse un match en « en
  cours » pour toujours ; `reclaim_stale` le remet en file après 15 min sans
  nouvelle. Les écritures de progression servent de battement de cœur, ce qui
  évite de reprendre un match qui avance encore. `attempts` plafonne les
  reprises : un fichier qui fait planter le worker bloquerait sinon la file
  derrière lui indéfiniment.
- **`/sante` répond 503 si des matchs sont bloqués ou si le disque sature**,
  pour qu'une sonde externe détecte la panne sans lire le corps de la réponse.
- **L'espace disque est protégé sur trois fronts** : réserve de 5 Go vérifiée
  avant *et pendant* l'écriture d'un dépôt, rétention de 90 jours purgée par le
  worker. Le dépôt étant ouvert sans compte, un disque plein arrêterait le
  service pour tous les clubs — y compris les analyses en cours.

## Current state vs. README

`README.md` (français) décrit le système visé — équipes, homographie, vitesses.
`ROADMAP.md` liste ce qui est réellement implémenté étape par étape, ainsi que les
bloquants juridiques à trancher avant tout entraînement destiné à un produit
commercial : Ultralytics est en AGPL-3.0 (clause réseau) et le dataset DFL Bundesliga
est sous conditions de compétition.

Aucun modèle entraîné n'est présent dans le dépôt ; `data/README.md` dit quoi placer
où. Rien ne tourne de bout en bout tant que `models/` est vide.
