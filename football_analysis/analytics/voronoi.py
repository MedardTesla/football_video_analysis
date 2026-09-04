"""Diagramme de Voronoï : zone de contrôle de chaque équipe.

Calculé sur une grille du terrain plutôt qu'avec scipy.spatial.Voronoi : la
grille se rend directement en image de radar et donne gratuitement la surface
contrôlée (en comptant les cellules), qui est la statistique réellement utile.
"""
from __future__ import annotations

import numpy as np

from ..pitch.geometry import PITCH


def control_map(
    team_a: np.ndarray, team_b: np.ndarray, resolution_cm: int = 50
) -> np.ndarray:
    """Rend une grille (H, W) : 0 = contrôlé par A, 1 = par B.

    `team_a` / `team_b` : positions terrain (cm), formes (N, 2) et (M, 2).
    """
    if team_a.size == 0 or team_b.size == 0:
        raise ValueError("les deux équipes doivent avoir au moins un joueur")

    xs = np.arange(0, PITCH.length, resolution_cm)
    ys = np.arange(0, PITCH.width, resolution_cm)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

    # Distance au joueur le plus proche de chaque équipe.
    d_a = np.linalg.norm(grid[:, None, :] - team_a[None, :, :], axis=-1).min(axis=1)
    d_b = np.linalg.norm(grid[:, None, :] - team_b[None, :, :], axis=-1).min(axis=1)

    return (d_b < d_a).astype(np.uint8).reshape(len(ys), len(xs))


def control_share(
    team_a: np.ndarray, team_b: np.ndarray, resolution_cm: int = 50
) -> tuple[float, float]:
    """Part du terrain contrôlée par chaque équipe, en fraction de 1."""
    grid = control_map(team_a, team_b, resolution_cm=resolution_cm)
    share_b = float(grid.mean())
    return 1.0 - share_b, share_b
