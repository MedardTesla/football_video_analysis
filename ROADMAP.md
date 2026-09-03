# Roadmap — de prototype à produit

## État réel du code

| # | Étape | État |
|---|-------|------|
| 1 | Détection YOLOv8 | Fait — `detection/detector.py`, **poids à fournir** |
| 2 | Suivi BoT-SORT (+CMC) | Fait — `tracking/tracker.py`, testé sur doublures |
| 3 | Équipes SigLIP + UMAP + K-Means | Structure prête — `teams/classifier.py`, non testé |
| 4 | Points clés terrain (YOLOv8-pose, 32 kp) | Géométrie + script d'entraînement prêts, **modèle à entraîner** |
| 5 | Homographie bidirectionnelle | Fait — `pitch/view.py` (+ lissage fenêtre glissante) |
| 6 | Voronoï / trajectoire ballon | Fait — `analytics/` |
| 7 | Optimisation vitesse | Non commencé — non bloquant en traitement asynchrone |
| 8 | Pipeline + CLI + statistiques JSON | Fait — `pipeline.py`, `cli.py`, `analytics/stats.py` |

Aucun modèle entraîné n'est présent dans le dépôt. Rien n'est exécutable de bout en
bout tant que `models/` n'est pas rempli.

## Bloquants pour la commercialisation

Ce sont des contraintes juridiques, pas techniques. Elles se règlent en amont, sinon
elles se règlent en justice.

**1. Ultralytics est en AGPL-3.0.** Vendre une analyse produite par YOLOv8 via un
service en ligne déclenche la clause réseau de l'AGPL : tu dois publier l'intégralité
du code source de ton service sous AGPL. Trois sorties :
- licence entreprise Ultralytics (payante, annuelle) ;
- remplacer le détecteur par un modèle sous Apache-2.0 (RF-DETR, RT-DETR, YOLOX) ;
- assumer l'open source complet.
À trancher **avant** d'investir dans l'entraînement, car cela change les poids.

**2. Le dataset DFL Bundesliga (Kaggle)** est publié sous des conditions de
compétition. Un modèle entraîné dessus n'est pas librement commercialisable. Pour un
produit, il faut un jeu de données propre — idéalement filmé chez tes clubs pilotes,
ce qui règle en même temps le problème d'angle de caméra ci-dessous.

**3. Les images de match** appartiennent au club ou au diffuseur. Pour de petits
clubs filmant eux-mêmes, c'est simple, mais le contrat doit le dire.

## Les limites que tu as identifiées, par ordre d'impact réel

1. **Angle de caméra** — c'est *le* risque produit. Un petit club filme depuis une
   tribune basse ou un trépied en bord de touche, pas depuis la position de
   diffusion. Un modèle robuste en vue TV peut être inutilisable sur ces images.
   À vérifier sur de vraies vidéos de club **avant** tout autre travail.
2. **Vitesse (1 FPS)** — non bloquant si l'analyse est asynchrone : le club dépose sa
   vidéo, reçoit son rapport plus tard. Le temps réel n'est nécessaire que pour du
   direct, qui n'est probablement pas le premier besoin.
3. **Occlusions** — partiellement traité : BoT-SORT avec compensation de mouvement
   caméra remplace ByteTrack. La ré-identification par apparence améliorerait encore les
   statistiques individuelles. Ça compte pour « distance parcourue par joueur »,
   moins pour les statistiques d'équipe.
4. **Ballon aérien / 3D** — dernier de la liste : ça dégrade le radar, ça ne casse
   rien.

## Ordre de travail proposé

- ~~**Phase 0 — valider le besoin**~~ : fait sur deux matchs, voir
  `ANALYSE_TERRAIN.md`. Résultat déterminant : la disponibilité de l'homographie
  passe de 47 % à plus de 92 % selon la seule captation, à code identique.
  **Écrire un cahier des charges de captation avant tout autre travail** — c'est
  le levier le plus rentable du projet.
- **Phase 1 — pipeline bout en bout** sur une vidéo, sortie = vidéo annotée + radar.
- **Phase 2 — entraîner le modèle de keypoints terrain** (YOLOv8x-pose, mosaic à 0 :
  l'augmentation mosaic colle plusieurs terrains sur une image et apprend au modèle à
  en chercher plusieurs).
- ~~**Phase 3 — rapport livrable**~~ : fait (`report.py`), imprimable et lisible
  hors ligne. Démo : https://claude.ai/code/artifact/13840721-26c5-4d07-a58f-1d28977428ed
- **Phase 4 — industrialisation** : file d'attente, stockage, facturation.
- **Phase 5 — optimisation** (ONNX/TensorRT), seulement si le coût GPU le justifie.
