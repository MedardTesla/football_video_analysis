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

# Adresse publique du service, pour que les liens envoyés par message soient
# cliquables. Vide, le message montre un chemin relatif — inutilisable, mais
# le club a de toute façon reçu le lien complet à l'écran.
BASE_URL = os.environ.get("FA_BASE_URL", "").rstrip("/")


# Longueurs maximales des champs libres. Le `maxlength` d'un formulaire est
# une aide à la saisie, pas une contrainte : rien n'empêche d'envoyer la
# requête directement. Sans coupe côté serveur, un nom de 50 000 caractères
# est stocké puis renvoyé sur chaque page.
LONGUEUR_CLUB = 80
LONGUEUR_MATCH = 120
LONGUEUR_CONTACT = 120
LONGUEUR_LIEN = 500

# Un club ne peut pas avoir plus de matchs en attente que cela. Limite de
# produit autant que garde-fou : un club qui dépose sa saison entière d'un
# coup monopoliserait la file, et ses propres rapports arriveraient plus tard.
FILE_MAX_PAR_CLUB = 3

# Plafond global de la file. Au-delà, le service refuse poliment plutôt que
# d'accepter des matchs qu'il ne traitera pas avant des jours.
FILE_MAX = 50


# Jeton d'accès à la page d'exploitation. Vide, la page est inaccessible —
# c'est le défaut : mieux vaut pas d'administration qu'une administration
# ouverte à tous.
ADMIN_TOKEN = os.environ.get("FA_ADMIN_TOKEN", "")
