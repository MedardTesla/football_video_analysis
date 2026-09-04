"""Recoller les pistes d'un même joueur.

Le traqueur perd un joueur dès qu'il sort du cadre, qu'un autre le masque
longuement ou que la caméra balaie trop vite. Il lui attribue alors un nouvel
identifiant. Sur un extrait réel, 35 identités ont été produites pour 22
joueurs — chacune ne portant qu'une fraction de la distance parcourue.

Le recollement se fait après coup, dans l'espace du terrain : deux segments
appartiennent au même joueur s'ils ne se chevauchent pas dans le temps, si
l'écart est bref, et si la distance à parcourir entre la fin de l'un et le
début de l'autre reste physiquement possible.

Aucune distance n'est ajoutée pour combler l'intervalle : ce trajet n'a pas
été observé, et l'inventer reviendrait à fabriquer la mesure que le
recollement doit justement rendre crédible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Vitesse au-delà de laquelle un déplacement n'est plus une course. Le record
# du monde tourne autour de 12 m/s ; on garde une marge pour l'imprécision de
# localisation, qui gonfle les écarts mesurés.
VITESSE_MAX_MS = 14.0

# Au-delà, deux segments peuvent appartenir à deux joueurs différents passés
# au même endroit. Trois secondes couvrent une occlusion ou une sortie de
# cadre brève, sans autoriser un aller-retour de milieu de terrain.
ECART_MAX_S = 3.0


@dataclass
class Segment:
    """Une piste continue, réduite à ses extrémités."""

    track_id: int
    first_frame: int
    last_frame: int
    first_xy: np.ndarray
    last_xy: np.ndarray
    team: int | None = None

    @property
    def duration_frames(self) -> int:
        return self.last_frame - self.first_frame + 1


@dataclass
class Stitcher:
    """Accumule les extrémités des pistes puis propose un recollement."""

    segments: dict[int, Segment] = field(default_factory=dict)

    def observe(
        self, track_id: int, frame: int, xy: np.ndarray, team: int | None
    ) -> None:
        """`frame` compte les images **traitées**, pas celles de la source.

        Mélanger les deux unités fausse silencieusement tous les écarts : avec
        un sous-échantillonnage à une image sur deux, les intervalles
        paraissent deux fois plus longs et plus rien ne se recolle.
        """
        segment = self.segments.get(track_id)
        if segment is None:
            self.segments[track_id] = Segment(
                track_id=track_id, first_frame=frame, last_frame=frame,
                first_xy=np.array(xy, dtype=float), last_xy=np.array(xy, dtype=float),
                team=team,
            )
            return
        segment.last_frame = frame
        segment.last_xy = np.array(xy, dtype=float)
        if segment.team is None:
            segment.team = team

    def mapping(
        self, fps: float, vitesse_max: float = VITESSE_MAX_MS,
        ecart_max_s: float = ECART_MAX_S,
    ) -> dict[int, int]:
        """Identifiant d'origine -> identifiant conservé.

        Chaînage transitif : si A se recolle à B et B à C, les trois portent
        l'identifiant de A. Un joueur peut être perdu plusieurs fois.
        """
        ordre = sorted(self.segments.values(), key=lambda s: s.first_frame)
        canonique: dict[int, int] = {s.track_id: s.track_id for s in ordre}
        # Fin de chaîne : le dernier segment rattaché à chaque identifiant.
        queue: dict[int, Segment] = {}

        for segment in ordre:
            meilleur, meilleur_cout = None, None
            for racine, precedent in queue.items():
                if segment.team is not None and precedent.team is not None:
                    if segment.team != precedent.team:
                        continue
                ecart_frames = segment.first_frame - precedent.last_frame
                if ecart_frames <= 0:
                    # Chevauchement : deux joueurs visibles en même temps.
                    continue
                ecart_s = ecart_frames / fps
                if ecart_s > ecart_max_s:
                    continue
                distance_cm = float(np.linalg.norm(segment.first_xy - precedent.last_xy))
                if distance_cm / 100 > vitesse_max * ecart_s:
                    continue
                # À contrainte respectée, le plus proche dans l'espace gagne.
                if meilleur_cout is None or distance_cm < meilleur_cout:
                    meilleur, meilleur_cout = racine, distance_cm

            if meilleur is None:
                queue[segment.track_id] = segment
            else:
                canonique[segment.track_id] = meilleur
                queue[meilleur] = segment

        return canonique
