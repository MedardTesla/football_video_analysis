# Modèles et données

Rien de lourd n'est versionné (voir `.gitignore`). À placer manuellement :

```
models/player_detection.pt    # YOLOv8 — 4 classes : ball, goalkeeper, player, referee
models/pitch_keypoints.pt     # YOLOv8-pose — 32 points clés (voir pitch/geometry.py)
input_video/                  # vidéos sources
```

L'ordre des 32 keypoints dans `football_analysis/pitch/geometry.py` est le contrat
entre le modèle pose et le code. Le modifier invalide tous les poids entraînés.
