"""Détection des lignes du terrain, et estimation de l'exploitabilité.

AVERTISSEMENT — ce module fournit des primitives de mesure, PAS un verdict.

Il a été écrit pour estimer à moindre coût la proportion de frames offrant
assez de structure pour une homographie. Cette tentative a échoué : selon le
réglage des seuils, le compte de lignes passait de 0 à 23 sur la même image,
sans réglage intermédiaire qui distingue les vraies lignes de terrain des
poteaux de but, des ombres et des maillots clairs. Sur une pelouse en plein
soleil, les lignes ne sont que légèrement plus contrastées que l'herbe.

La mesure a finalement été obtenue par jugement visuel sur 36 images
échantillonnées (voir ANALYSE_TERRAIN.md). N'utilisez pas ces fonctions comme
filtre de qualité sans les avoir calibrées sur vos propres vidéos.

Deux pièges dominent :

- Un stade est plein de lignes droites qui ne sont pas le terrain : grillages,
  murs de gradins, poteaux, toits d'abris. Tout est donc restreint au masque
  de pelouse.
- Les lignes du terrain sont des structures *fines et claires* sur fond
  sombre. Un seuil sur la luminosité attrape aussi les maillots blancs et les
  reflets ; un chapeau haut-de-forme morphologique ne retient que ce qui est
  fin, ce qui écarte les grandes surfaces claires.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .mask import turf_mask


MIN_LINE_LENGTH = 120


def line_pixels(frame: np.ndarray, turf: np.ndarray, thickness: int = 15) -> np.ndarray:
    """Masque des pixels appartenant à une ligne de terrain.

    `thickness` est l'épaisseur maximale, en pixels, d'une structure à
    retenir. Une contrainte de couleur ("les lignes sont blanches") a été
    essayée puis retirée : sur une pelouse en plein soleil, les lignes ne sont
    que légèrement plus claires que l'herbe, et le filtre supprimait de vraies
    lignes de touche. La sélection par longueur, plus bas, écarte bien mieux
    les maillots — un maillot fait 40 px de large, une ligne bien plus.
    """
    # Éroder la pelouse : ses bords sont eux-mêmes des transitions claires,
    # qui produiraient des lignes fantômes tout autour du terrain.
    eroded = cv2.erode(turf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    if eroded.sum() == 0:
        return eroded

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)

    # Seuil calculé sur la pelouse seule. Calculé sur toute l'image, les
    # poteaux de but et les gradins en plein soleil écrasent la statistique
    # et le seuil monte trop haut pour les lignes, bien moins contrastées.
    threshold, _ = cv2.threshold(
        tophat[eroded > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thin = (tophat > threshold).astype(np.uint8) * 255
    return cv2.bitwise_and(thin, eroded)


def _angle(segment: np.ndarray) -> float:
    """Orientation en degrés dans [0, 180)."""
    x1, y1, x2, y2 = segment
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)


def _offset(segment: np.ndarray) -> float:
    """Distance signée de l'origine à la droite portant le segment."""
    x1, y1, x2, y2 = segment
    theta = np.radians(_angle(segment))
    return float(x1 * -np.sin(theta) + y1 * np.cos(theta))


def merge_segments(
    segments: np.ndarray, angle_tol: float = 8.0, offset_tol: float = 30.0
) -> list[np.ndarray]:
    """Fusionne les segments qui décrivent la même droite.

    HoughLinesP découpe une ligne de touche en dizaines de segments dès qu'un
    joueur la traverse. Sans fusion, on compterait ces morceaux comme autant
    de lignes distinctes et on surestimerait massivement la structure visible.
    """
    kept: list[np.ndarray] = []
    for segment in segments:
        angle, offset = _angle(segment), _offset(segment)
        for other in kept:
            delta = abs(angle - _angle(other))
            delta = min(delta, 180.0 - delta)
            if delta < angle_tol and abs(offset - _offset(other)) < offset_tol:
                break
        else:
            kept.append(segment)
    return kept


def intersections(lines: list[np.ndarray], shape: tuple[int, int]) -> list[tuple[float, float]]:
    """Points d'intersection tombant dans l'image.

    Ce sont eux qui comptent : une homographie a besoin de correspondances de
    points, pas de lignes. Deux lignes de touche parallèles n'en donnent
    aucune, d'où l'importance d'avoir plusieurs orientations.
    """
    height, width = shape[:2]
    points = []
    for i, a in enumerate(lines):
        for b in lines[i + 1 :]:
            d = (a[2] - a[0]) * (b[3] - b[1]) - (a[3] - a[1]) * (b[2] - b[0])
            if abs(d) < 1e-6:
                continue
            t = ((b[0] - a[0]) * (b[3] - b[1]) - (b[1] - a[1]) * (b[2] - b[0])) / d
            x, y = a[0] + t * (a[2] - a[0]), a[1] + t * (a[3] - a[1])
            if 0 <= x < width and 0 <= y < height:
                points.append((float(x), float(y)))
    return points


@dataclass
class LineEvidence:
    """Compte brut. Aucune interprétation : voir l'avertissement en tête."""

    n_lines: int
    n_orientations: int
    n_intersections: int


def evaluate(frame: np.ndarray, turf: np.ndarray | None = None) -> LineEvidence:
    """Mesure la structure de terrain visible sur une frame."""
    turf = turf_mask(frame) if turf is None else turf
    if turf.sum() == 0:
        return LineEvidence(0, 0, 0)

    pixels = line_pixels(frame, turf)
    raw = cv2.HoughLinesP(
        pixels, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=MIN_LINE_LENGTH, maxLineGap=40,
    )
    if raw is None:
        return LineEvidence(0, 0, 0)

    lines = merge_segments(raw.reshape(-1, 4))
    # Familles d'orientation par pas de 20°, pour distinguer les lignes de
    # touche des lignes de surface sans compter deux fois un léger biais.
    families = {int(_angle(line) // 20) for line in lines}
    return LineEvidence(
        n_lines=len(lines),
        n_orientations=len(families),
        n_intersections=len(intersections(lines, frame.shape)),
    )
