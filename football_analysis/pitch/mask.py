"""Masque de la pelouse, pour écarter tout ce qui n'est pas sur le terrain.

Sur une caméra de club placée en bord de touche, le détecteur trouve autant de
personnes hors jeu que de joueurs : entraîneurs debout, remplaçants sur le
banc, spectateurs dans les gradins. Les compter fausse la possession, le
Voronoï et le nombre de joueurs.

Filtrer par homographie serait plus propre, mais l'homographie n'est pas
toujours calculable — et quand elle l'est, elle a déjà besoin d'un terrain
correctement identifié. Ce masque ne dépend que de la couleur, donc il
fonctionne sur toutes les frames, y compris celles où les lignes sont hors
champ.
"""
from __future__ import annotations

import cv2
import numpy as np

# Teinte du gazon en HSV OpenCV (H sur 0-179). La plage est large : le même
# terrain passe du vert-jaune en plein soleil au vert sombre à l'ombre.
TURF_LOWER = np.array([30, 40, 40], dtype=np.uint8)
TURF_UPPER = np.array([90, 255, 255], dtype=np.uint8)


def turf_mask(frame: np.ndarray, close_px: int = 25) -> np.ndarray:
    """Masque binaire de la surface de jeu (255 = pelouse).

    La fermeture morphologique rebouche les joueurs, les lignes blanches et
    les ombres, qui découpent sinon la pelouse en fragments. Seule la plus
    grande composante est conservée : les bandes d'herbe derrière les gradins
    ne font pas partie du terrain.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TURF_LOWER, TURF_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def playable_area(frame: np.ndarray) -> np.ndarray:
    """Masque rempli : la pelouse et tout ce qui se trouve dessus.

    `turf_mask` laisse des trous là où se tiennent les joueurs. Remplir
    l'enveloppe évite qu'un joueur au centre d'un attroupement soit rejeté
    parce qu'il masque lui-même l'herbe sous ses pieds.
    """
    mask = turf_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return filled


def on_pitch(mask: np.ndarray, boxes: np.ndarray, tolerance_px: int = 6) -> np.ndarray:
    """Booléens : le point d'appui de chaque boîte est-il sur le terrain ?

    On teste le milieu du bord inférieur — les pieds — et non le centre de la
    boîte : un entraîneur debout au bord de touche a le torse au-dessus de la
    pelouse tout en ayant les pieds à côté.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)

    height, width = mask.shape[:2]
    xs = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2).astype(int), 0, width - 1)
    ys = np.clip((boxes[:, 3] - tolerance_px).astype(int), 0, height - 1)
    return mask[ys, xs] > 0
