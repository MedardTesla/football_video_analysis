"""Statistiques de match, en coordonnées terrain.

C'est le vrai livrable pour un club : des chiffres, pas une vidéo annotée.
Tout est calculé en centimètres puis converti, pour éviter des conversions
dispersées dans le code.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

CM_PER_M = 100.0
# Au-delà, c'est une erreur de suivi (identités permutées), pas un sprint :
# le record du monde est à ~12 m/s.
MAX_PLAUSIBLE_SPEED_MS = 12.0


@dataclass
class PlayerStats:
    track_id: int
    team: int | None = None
    distance_m: float = 0.0
    top_speed_ms: float = 0.0
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
        if previous is None:
            return

        step_m = float(np.linalg.norm(xy - previous)) / CM_PER_M
        speed = step_m * self.fps
        if speed > MAX_PLAUSIBLE_SPEED_MS:
            # Saut d'identité : on ignore ce pas plutôt que de gonfler
            # la distance totale du joueur.
            return
        stats.distance_m += step_m
        stats.top_speed_ms = max(stats.top_speed_ms, speed)

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
            garde.top_speed_ms = max(garde.top_speed_ms, stats.top_speed_ms)
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
                    "top_speed_ms": round(p.top_speed_ms, 2),
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
