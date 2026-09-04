# Modèles et données

Rien de lourd n'est versionné (voir `.gitignore`). Les deux modèles s'obtiennent
avec les notebooks de `training/`, sur **Colab ou Kaggle** — GPU gratuit, aucun
matériel requis. Les notebooks détectent la plateforme et s'adaptent seuls.

Kaggle offre 30 h de GPU par semaine, contre un quota variable et sans préavis
chez Colab : c'est le choix par défaut si un entraînement doit être relancé.
Sur Kaggle, l'archive d'images du test facultatif s'ajoute comme *Dataset*
depuis le panneau de droite, et les poids se récupèrent dans l'onglet *Output*
après un *Save Version*.

| Modèle | Notebook | Durée |
|---|---|---|
| `pitch_keypoints.pt` | `train_keypoints_colab.ipynb` | 1 h 30 à 2 h 30 |
| `player_detection.pt` | `train_detection_colab.ipynb` | 2 h à 3 h |

Chacun contient un garde-fou qui arrête l'exécution si le dataset ne respecte
plus la convention attendue — ordre des 32 points clés pour l'un, ordre des
quatre classes pour l'autre. Ces deux divergences produiraient des modèles qui
convergent normalement en étant inutilisables.

À placer manuellement :

```
models/player_detection.pt    # YOLOv8 — 4 classes : ball, goalkeeper, player, referee
models/pitch_keypoints.pt     # YOLOv8-pose — 32 points clés (voir pitch/geometry.py)
input_video/                  # vidéos sources
```

L'ordre des 32 keypoints dans `football_analysis/pitch/geometry.py` est le contrat
entre le modèle pose et le code. Le modifier invalide tous les poids entraînés.
