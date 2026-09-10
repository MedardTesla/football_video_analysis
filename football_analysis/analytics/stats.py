"""Statistiques de match, en coordonnées terrain.

C'est le vrai livrable pour un club : des chiffres, pas une vidéo annotée.
Tout est calculé en centimètres puis converti, pour éviter des conversions
dispersées dans le code.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

CM_PER_M = 100.0
# Au-delà, c'est une erreur de suivi (identités permutées), pas un sprint. Un
# footballeur d'élite culmine vers 10 m/s (36 km/h) ; le record du monde du
# 100 m, tenu sur dix mètres, atteint 12 m/s. Retenir 12 laissait donc passer
# des valeurs qu'aucun joueur n'atteint.
MAX_PLAUSIBLE_SPEED_MS = 10.0

# Durée sur laquelle la vitesse de pointe est mesurée. Une pointe déduite de
# deux images consécutives ne mesure pas une course mais le bruit : à 12,5 fps
# une image dure 0,08 s, et une erreur de position de 96 cm — ordinaire pour
# une homographie — suffit à produire 12 m/s. Le maximum de milliers d'écarts
# bruités vient alors se coller juste sous le plafond, et toutes les pointes du
# rapport se ressemblent. Mesuré sur un extrait réel avant ce correctif :
# 22 joueurs sur 28 au-dessus de 40 km/h.
#
# Une demi-seconde impose de couvrir réellement du terrain. La mesure porte sur
# le déplacement net entre les deux bouts de la fenêtre : une course en courbe
# est donc sous-estimée, ce qui est le bon compromis pour une pointe — elle ne
# peut jamais être surestimée par un tremblement isolé.
SPEED_WINDOW_S = 0.5


def _max_connu(a: float | None, b: float | None) -> float | None:
    """Maximum de deux valeurs dont l'une peut être inconnue."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


@dataclass
class PlayerStats:
    track_id: int
    team: int | None = None
    distance_m: float = 0.0
    # `None` tant qu'aucune fenêtre complète n'a été observée : une pointe
    # inconnue n'est pas une pointe nulle, et le rapport doit pouvoir les
    # distinguer. Une piste trop fragmentée n'en produit jamais.
    top_speed_ms: float | None = None
    frames_seen: int = 0


@dataclass
class MatchStats:
    """Accumule les statistiques frame par frame."""

    fps: float
    players: dict[int, PlayerStats] = field(default_factory=dict)
    possession_frames: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    measured_frames: int = 0
    unmeasured_frames: int = 0
    control_frames: int = 0
    control_sum: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    _last_xy: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _positions: dict[int, deque] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Nombre de positions couvrant SPEED_WINDOW_S : n positions décrivent
        # n-1 intervalles, d'où le +1. Deux au minimum, sans quoi aucune
        # vitesse n'est calculable.
        self._fenetre = max(2, int(round(self.fps * SPEED_WINDOW_S)) + 1)

    def update_player(self, track_id: int, xy: np.ndarray, team: int | None) -> None:
        """`xy` : position terrain en cm.

        Un identifiant négatif signale une détection non associée par le
        traqueur. L'accepter fusionnerait tous les joueurs non suivis en une
        identité unique, qui cumulerait leurs distances et paraîtrait vue plus
        longtemps que ne dure le match.
        """
        if track_id < 0:
            raise ValueError(
                f"identifiant de piste invalide : {track_id}. "
                "Filtrer les détections non suivies avant d'accumuler."
            )
        stats = self.players.setdefault(track_id, PlayerStats(track_id=track_id))
        stats.frames_seen += 1
        if team is not None:
            stats.team = team

        previous = self._last_xy.get(track_id)
        self._last_xy[track_id] = xy
        positions = self._positions.setdefault(track_id, deque(maxlen=self._fenetre))
        positions.append(xy)
        if previous is None:
            return

        step_m = float(np.linalg.norm(xy - previous)) / CM_PER_M
        if step_m * self.fps > MAX_PLAUSIBLE_SPEED_MS:
            # Saut d'identité : on ignore ce pas plutôt que de gonfler la
            # distance totale du joueur. La fenêtre est vidée par la même
            # occasion, la trajectoire n'étant plus continue.
            positions.clear()
            positions.append(xy)
            return
        stats.distance_m += step_m
        vitesse = self._vitesse_fenetre(positions)
        if vitesse is not None:
            stats.top_speed_ms = _max_connu(stats.top_speed_ms, vitesse)

    def _vitesse_fenetre(self, positions: deque) -> float | None:
        """Vitesse soutenue sur la fenêtre, ou `None` si elle est incomplète.

        Rendre une valeur sur une fenêtre partielle rouvrirait la porte au
        bruit : c'est exactement la mesure sur deux images que l'on remplace.
        """
        if len(positions) < self._fenetre:
            return None
        ecart_m = float(np.linalg.norm(positions[-1] - positions[0])) / CM_PER_M
        duree_s = (len(positions) - 1) / self.fps
        return ecart_m / duree_s if duree_s else None

    def merge_identities(self, mapping: dict[int, int]) -> None:
        """Fusionne les pistes recollées en une seule identité.

        Les distances s'additionnent ; l'intervalle non observé entre deux
        segments n'est pas comblé, faute d'avoir été mesuré. La vitesse de
        pointe est le maximum, pas la moyenne : c'est une pointe.
        """
        fusionnes: dict[int, PlayerStats] = {}
        for track_id, stats in self.players.items():
            racine = mapping.get(track_id, track_id)
            garde = fusionnes.get(racine)
            if garde is None:
                fusionnes[racine] = PlayerStats(
                    track_id=racine, team=stats.team,
                    distance_m=stats.distance_m, top_speed_ms=stats.top_speed_ms,
                    frames_seen=stats.frames_seen,
                )
                continue
            garde.distance_m += stats.distance_m
            garde.top_speed_ms = _max_connu(garde.top_speed_ms, stats.top_speed_ms)
            garde.frames_seen += stats.frames_seen
            if garde.team is None:
                garde.team = stats.team
        self.players = fusionnes

    def mark_measured(self) -> None:
        self.measured_frames += 1

    def mark_unmeasured(self) -> None:
        """Frame sans homographie exploitable.

        Les dernières positions connues sont oubliées : à la reprise, le
        joueur aura bougé pendant tout l'intervalle, et compter ce saut comme
        une course lui attribuerait une distance qu'il n'a pas parcourue —
        ou, pire, une distance plausible mais fausse.
        """
        self.unmeasured_frames += 1
        self._last_xy.clear()
        # Même raison pour les fenêtres de vitesse : à la reprise, les deux
        # bouts encadreraient l'intervalle non observé et fabriqueraient une
        # pointe à partir d'un trajet que personne n'a vu.
        self._positions.clear()

    @property
    def coverage(self) -> float:
        """Fraction du match réellement mesurée."""
        total = self.measured_frames + self.unmeasured_frames
        return self.measured_frames / total if total else 0.0

    def update_possession(self, team: int | None) -> None:
        if team is not None:
            self.possession_frames[team] += 1

    def update_control(self, shares: dict[int, float]) -> None:
        """Part du terrain contrôlée par chaque équipe sur cette frame.

        Contrairement aux distances individuelles, cette statistique reste
        valable malgré une localisation imprécise : déplacer tous les joueurs
        de quelques mètres ne change presque pas le partage du terrain, alors
        que cela fausse chaque pas de course.
        """
        if not shares:
            return
        self.control_frames += 1
        for team, part in shares.items():
            self.control_sum[team] += part

    def control_share(self) -> dict[int, float]:
        """Contrôle territorial moyen sur les frames mesurables."""
        if self.control_frames == 0:
            return {}
        return {t: s / self.control_frames for t, s in self.control_sum.items()}

    def possession_share(self) -> dict[int, float]:
        total = sum(self.possession_frames.values())
        if total == 0:
            return {}
        return {team: count / total for team, count in self.possession_frames.items()}

    def to_dict(self) -> dict:
        total = self.measured_frames + self.unmeasured_frames
        return {
            "coverage": round(self.coverage, 3),
            "measured_seconds": round(self.measured_frames / self.fps, 1),
            "unmeasured_seconds": round(self.unmeasured_frames / self.fps, 1),
            "total_seconds": round(total / self.fps, 1),
            "possession": self.possession_share(),
            "control": self.control_share(),
            "control_seconds": round(self.control_frames / self.fps, 1),
            "players": [
                {
                    "track_id": p.track_id,
                    "team": p.team,
                    "distance_m": round(p.distance_m, 1),
                    "top_speed_ms": (
                        round(p.top_speed_ms, 2) if p.top_speed_ms is not None else None
                    ),
                    "seconds_seen": round(p.frames_seen / self.fps, 1),
                }
                for p in sorted(self.players.values(), key=lambda s: -s.distance_m)
            ],
        }


def nearest_player_team(
    ball_xy: np.ndarray | None,
    players_xy: np.ndarray,
    teams: np.ndarray,
    max_distance_cm: float = 300.0,
) -> int | None:
    """Équipe du joueur le plus proche du ballon, si assez proche.

    Au-delà du seuil, personne n'a le ballon : le compter quand même
    fausserait la possession pendant les longues passes.
    """
    if ball_xy is None or len(players_xy) == 0:
        return None
    distances = np.linalg.norm(players_xy - ball_xy, axis=1)
    closest = int(np.argmin(distances))
    if distances[closest] > max_distance_cm:
        return None
    return int(teams[closest])
