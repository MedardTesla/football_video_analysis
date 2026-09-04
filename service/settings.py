"""Réglages partagés entre l'API et le worker.

Module volontairement sans dépendance lourde. L'API n'a besoin ni de torch ni
d'OpenCV : elle reçoit des fichiers et sert des pages. Faire transiter une
constante par `worker.py` — qui importe le pipeline — obligerait à installer
plusieurs gigaoctets de bibliothèques de calcul sur la machine qui sert le
formulaire de dépôt.
"""
from __future__ import annotations

import os
from pathlib import Path

# Racine des données partagées entre l'API et le worker : base SQLite et
# vidéos. Les deux processus peuvent tourner sur des machines différentes à
# condition de voir le même chemin.
DATA_ROOT = Path(os.environ.get("FA_DATA_ROOT", "data/service"))

# Un match sans nouvelle pendant ce délai est considéré abandonné. Large à
# dessein : la phase de calibrage du classifieur d'équipes ne remonte aucune
# progression et peut durer plusieurs minutes sur une longue vidéo.
STALE_SECONDS = 15 * 60
