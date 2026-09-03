"""Géométrie d'un terrain de football, en centimètres réels.

Les dimensions ci-dessous ne sont pas une convention : elles ont été ajustées
sur les 228 images annotées du dataset public de points clés, en minimisant
l'erreur de reprojection de l'homographie. L'optimum tombe exactement sur le
terrain FIFA standard et les cotes de la loi du jeu.

    105 x 68 m, surface de 16,50 x 40,32 m : 0,398 % d'erreur
    120 x 70 m, surface de 20,15 x 41,00 m : 0,960 %

La seconde ligne est la convention diffusée par les exemples Roboflow. La
suivre aurait gonflé toute distance mesurée de 14 % — un terrain de 120 m au
lieu de 105. Pour un produit qui vend « distance parcourue », c'est une erreur
systématique inacceptable.

Les dimensions restent réglables : un terrain réel fait entre 100 et 110 m de
long. Un club qui mesure le sien supprime cette incertitude ; sans mesure,
elle vaut environ +/- 5 % sur les distances.

Les 32 sommets sont l'espace cible de l'homographie : le modèle YOLOv8-pose
prédit ces mêmes points dans l'image, et `findHomography` relie les deux.
L'ordre des sommets est le contrat entre le modèle entraîné et ce fichier —
le changer invalide tous les poids déjà entraînés.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SoccerPitch:
    length: int = 10500          # ligne de touche (FIFA : 100 à 110 m)
    width: int = 6800            # ligne de but (FIFA : 64 à 75 m)
    penalty_box_length: int = 1650   # loi du jeu : 16,50 m
    penalty_box_width: int = 4032    # loi du jeu : 40,32 m
    goal_box_length: int = 550
    goal_box_width: int = 1832
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> list[tuple[int, int]]:
        L, W = self.length, self.width
        pb_l, pb_w = self.penalty_box_length, self.penalty_box_width
        gb_l, gb_w = self.goal_box_length, self.goal_box_width
        r, spot = self.centre_circle_radius, self.penalty_spot_distance
        return [
            (0, 0),                              # 1  coin haut gauche
            (0, (W - pb_w) // 2),                # 2
            (0, (W - gb_w) // 2),                # 3
            (0, (W + gb_w) // 2),                # 4
            (0, (W + pb_w) // 2),                # 5
            (0, W),                              # 6  coin bas gauche
            (gb_l, (W - gb_w) // 2),             # 7
            (gb_l, (W + gb_w) // 2),             # 8
            (spot, W // 2),                      # 9  point de penalty gauche
            (pb_l, (W - pb_w) // 2),             # 10
            (pb_l, (W - gb_w) // 2),             # 11
            (pb_l, (W + gb_w) // 2),             # 12
            (pb_l, (W + pb_w) // 2),             # 13
            (L // 2, 0),                         # 14 ligne médiane haut
            (L // 2, W // 2 - r),                # 15 rond central haut
            (L // 2, W // 2 + r),                # 16 rond central bas
            (L // 2, W),                         # 17 ligne médiane bas
            (L - pb_l, (W - pb_w) // 2),         # 18
            (L - pb_l, (W - gb_w) // 2),         # 19
            (L - pb_l, (W + gb_w) // 2),         # 20
            (L - pb_l, (W + pb_w) // 2),         # 21
            (L - spot, W // 2),                  # 22 point de penalty droit
            (L - gb_l, (W - gb_w) // 2),         # 23
            (L - gb_l, (W + gb_w) // 2),         # 24
            (L, 0),                              # 25 coin haut droit
            (L, (W - pb_w) // 2),                # 26
            (L, (W - gb_w) // 2),                # 27
            (L, (W + gb_w) // 2),                # 28
            (L, (W + pb_w) // 2),                # 29
            (L, W),                              # 30 coin bas droit
            (L // 2 - r, W // 2),                # 31 rond central gauche
            (L // 2 + r, W // 2),                # 32 rond central droit
        ]

    @property
    def flip_index(self) -> list[int]:
        """Correspondance des sommets sous symétrie horizontale (0-based).

        YOLO-pose a besoin de `flip_idx` pour l'augmentation `fliplr` : quand
        l'image est mirrorée, le coin haut gauche devient le coin haut droit.
        Sans cette table, `fliplr` apprend au modèle des labels faux — panne
        silencieuse, le modèle converge quand même mais prédit n'importe quoi.

        Calculé depuis la géométrie plutôt qu'écrit à la main : une table
        fausse ne se voit qu'après plusieurs heures d'entraînement.
        """
        position = {vertex: i for i, vertex in enumerate(self.vertices)}
        return [position[(self.length - x, y)] for x, y in self.vertices]

    # Arêtes en indices 1-based, pour dessiner le radar.
    edges: tuple[tuple[int, int], ...] = field(
        default=(
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 17), (17, 16), (16, 15),
            (15, 14), (14, 1), (2, 10), (10, 11), (11, 12), (12, 13), (13, 5),
            (3, 7), (7, 8), (8, 4), (18, 21), (18, 19), (19, 20), (20, 21),
            (23, 24), (26, 27), (27, 28), (28, 29), (25, 26), (29, 30),
            (30, 17), (25, 14), (19, 23), (24, 20),
        ),
        repr=False,
    )


PITCH = SoccerPitch()
