"""Rendu : annotations sur la vidéo et radar 2D vu du dessus."""
from __future__ import annotations

import cv2
import numpy as np

from ..pitch.geometry import PITCH

# BGR. Les deux équipes doivent rester distinguables en niveaux de gris,
# pour les clubs qui impriment les rapports.
TEAM_COLORS = ((255, 128, 0), (0, 128, 255))   # bleu, orange
REFEREE_COLOR = (0, 255, 255)
BALL_COLOR = (255, 255, 255)
UNASSIGNED_COLOR = (170, 170, 170)


def team_color(team: int) -> tuple[int, int, int]:
    """Couleur d'une équipe, ou gris si non attribuée.

    Indexer TEAM_COLORS directement serait piégeux : en Python, -1 % 2 vaut 1,
    donc un joueur non attribué serait peint aux couleurs de l'équipe B sans
    que rien ne le signale.
    """
    return TEAM_COLORS[team] if team in (0, 1) else UNASSIGNED_COLOR


def draw_ellipse(
    frame: np.ndarray, bbox: np.ndarray, color: tuple[int, int, int], label: str | None = None
) -> np.ndarray:
    """Arc au sol sous le joueur, avec étiquette optionnelle."""
    x1, _, x2, y2 = (int(v) for v in bbox)
    x_center = (x1 + x2) // 2
    width = x2 - x1

    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(max(width, 1), max(int(0.35 * width), 1)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    if label is None:
        return frame

    w, h = 40, 20
    top_left = (x_center - w // 2, y2 + 5)
    bottom_right = (x_center + w // 2, y2 + 5 + h)
    cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)
    (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(
        frame,
        label,
        (x_center - text_w // 2, y2 + 5 + h - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_ball_marker(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Triangle au-dessus du ballon. Couleur distincte de celles des équipes."""
    x1, y1, x2, _ = (int(v) for v in bbox)
    x, y = (x1 + x2) // 2, y1
    points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
    cv2.drawContours(frame, [points], 0, BALL_COLOR, cv2.FILLED)
    cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
    return frame


def draw_pitch(scale: float = 0.1, padding: int = 50) -> np.ndarray:
    """Fond du radar : terrain vu du dessus, en pixels (1 px = 10 cm par défaut)."""
    height = int(PITCH.width * scale) + 2 * padding
    width = int(PITCH.length * scale) + 2 * padding
    pitch = np.full((height, width, 3), (40, 110, 40), dtype=np.uint8)

    vertices = PITCH.vertices
    for start, end in PITCH.edges:
        p1 = vertices[start - 1]
        p2 = vertices[end - 1]
        cv2.line(
            pitch,
            (int(p1[0] * scale) + padding, int(p1[1] * scale) + padding),
            (int(p2[0] * scale) + padding, int(p2[1] * scale) + padding),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.circle(
        pitch,
        (int(PITCH.length / 2 * scale) + padding, int(PITCH.width / 2 * scale) + padding),
        int(PITCH.centre_circle_radius * scale),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return pitch


def draw_on_pitch(
    pitch: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    scale: float = 0.1,
    padding: int = 50,
    radius: int = 8,
) -> np.ndarray:
    """Place des positions terrain (cm) sur le radar."""
    for x, y in points:
        cv2.circle(
            pitch,
            (int(x * scale) + padding, int(y * scale) + padding),
            radius,
            color,
            cv2.FILLED,
            cv2.LINE_AA,
        )
        cv2.circle(
            pitch,
            (int(x * scale) + padding, int(y * scale) + padding),
            radius,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return pitch


def overlay_radar(
    frame: np.ndarray,
    radar: np.ndarray,
    opacity: float = 0.75,
    width_ratio: float = 0.26,
    margin: int = 24,
) -> np.ndarray:
    """Incruste le radar en bas à droite.

    Placé au centre et occupant 40 % de la largeur, il masquait le jeu :
    l'entraîneur regarde la vidéo pour revoir une action, pas le radar. En bas
    à droite et à 26 %, il reste lisible sans couvrir le centre du terrain,
    où se trouve presque toujours le ballon.
    """
    echelle = (frame.shape[1] * width_ratio) / radar.shape[1]
    petit = cv2.resize(
        radar, (int(radar.shape[1] * echelle), int(radar.shape[0] * echelle))
    )
    h, w = petit.shape[:2]
    x = frame.shape[1] - w - margin
    y = frame.shape[0] - h - margin
    if y < 0 or x < 0:
        return frame
    region = frame[y : y + h, x : x + w]
    cv2.addWeighted(petit, opacity, region, 1 - opacity, 0, region)
    return frame
