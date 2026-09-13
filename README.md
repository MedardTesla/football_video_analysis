# Analyse vidéo de match de football

Un club dépose la vidéo de son match, reçoit un lien, et retrouve un rapport
chiffré une heure plus tard. Pas de compte à créer, pas de logiciel à installer.

Conçu pour des clubs qui filment eux-mêmes, avec une caméra unique.

## Ce que le rapport contient

- **Possession** — quelle équipe a le ballon, dans le temps
- **Contrôle du terrain** — quelle équipe occupe l'espace, par diagramme de Voronoï
- **Joueurs** — distance parcourue, pointe de vitesse, temps de jeu
- **Vue tactique** — positions projetées sur un terrain vu du dessus
- **Vidéo annotée** — joueurs identifiés, équipes colorées, ballon suivi

Chaque rapport affiche la **part du match réellement analysée** avant les
chiffres, et signale ce qui les limite. Un club doit savoir sur quoi il lit.

## État

| Étape | État |
|---|---|
| Détection joueurs, gardiens, arbitres, ballon | code prêt, **poids à entraîner** |
| Suivi BoT-SORT avec compensation caméra | fait |
| Équipes par SigLIP + UMAP + K-Means | fait, validé sur match réel |
| Points clés du terrain (32 repères) | modèle entraîné, **5 m d'erreur** |
| Homographie et péremption | fait |
| Possession, contrôle, distances | fait |
| Rapport et vidéo annotée | fait |
| Service de dépôt et de livraison | fait |
| Déploiement Docker | fait |

Le pipeline tourne de bout en bout sur de la vraie vidéo. Les deux limites
connues sont chiffrées dans `ANALYSE_TERRAIN.md`.

## Ce qui décide de la qualité

**La caméra, avant tout le reste.** Deux matchs de la même équipe, analysés par
le même code :

| Captation | Images exploitables |
|---|---|
| Tribune haute, 1080p | **83 %** |
| Bord de touche, 720p | **0 %** |

Aucun modèle ne rattrape cet écart. Filmer d'un point haut et reculé, en 1080p,
vaut plus que des mois d'ingénierie.

**Ce qui résiste à l'imprécision, et ce qui n'y résiste pas.** Avec 5 m
d'erreur de localisation :

| Statistique | Effet |
|---|---|
| Contrôle du terrain | décalé de 2 points |
| Distance parcourue | pas de course **multiplié par 16** |

D'où la position tenue par le rapport : les statistiques d'équipe sont
exploitables aujourd'hui, les distances individuelles sont des ordres de
grandeur.

## Démarrer

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
```

Les modèles ne sont pas versionnés. Deux notebooks Colab les produisent sur GPU
gratuit — voir `data/README.md`. Une fois `models/` rempli :

```bash
python -m football_analysis.cli match.mp4 -o resultat.mp4
```

Pour le service complet, voir `DEPLOIEMENT.md`.

## Organisation

```
football_analysis/   pipeline d'analyse
service/             dépôt, file d'attente, livraison des rapports
training/            notebooks d'entraînement des deux modèles
tests/               173 tests
```

`INTERFACE.md` décrit la forme du produit, ses écrans et son système visuel.
`CLAUDE.md` documente les décisions non évidentes et les pièges rencontrés.
`ANALYSE_TERRAIN.md` contient les mesures faites sur vidéos réelles.
`ROADMAP.md` liste ce qui reste, dont les questions de licence à trancher
avant toute commercialisation.

## Licence

Voir `LICENSE`. Attention : Ultralytics est sous AGPL-3.0, ce qui contraint
l'exploitation commerciale d'un service en ligne bâti dessus. Détails dans
`ROADMAP.md`.
