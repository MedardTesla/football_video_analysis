"""Tendance d'un club sur plusieurs matchs.

Un match isolé dit peu de choses. Ce qu'un entraîneur cherche, c'est de savoir
si son équipe progresse — c'est aussi ce qui justifie un abonnement plutôt
qu'un paiement à l'acte.

Seules les statistiques d'équipe sont agrégées. Les distances individuelles ne
le sont pas : à la précision de localisation actuelle, une erreur de 5 m
multiplie par seize la longueur d'un pas de course, et cumuler ce bruit sur
une saison ne le corrigerait pas, seulement le masquerait.
"""
from __future__ import annotations

from dataclasses import dataclass

from .jobs import Job, JobState


@dataclass
class MatchPoint:
    """Un match réduit à ce qui se compare d'une rencontre à l'autre."""

    match_name: str
    date: str
    possession: float | None
    control: float | None
    coverage: float | None


@dataclass
class Season:
    points: list[MatchPoint]

    @property
    def usable(self) -> bool:
        """Deux matchs au minimum : une tendance ne se lit pas sur un point."""
        return len(self.points) >= 2

    def _moyenne(self, champ: str) -> float | None:
        valeurs = [getattr(p, champ) for p in self.points]
        valeurs = [v for v in valeurs if v is not None]
        return sum(valeurs) / len(valeurs) if valeurs else None

    @property
    def possession_moyenne(self) -> float | None:
        return self._moyenne("possession")

    @property
    def control_moyen(self) -> float | None:
        return self._moyenne("control")


def _part(stats: dict, cle: str, equipe: int) -> float | None:
    """Part de l'équipe désignée, les clés JSON étant des chaînes."""
    valeurs = stats.get(cle) or {}
    for k, v in valeurs.items():
        if int(k) == equipe:
            return float(v)
    return None


def build(matchs: list[Job]) -> Season:
    """Construit la tendance à partir des matchs exploitables.

    Un match sans équipe désignée est écarté : son « équipe A » peut être
    l'adversaire, et l'inclure inverserait la courbe sans prévenir.
    """
    points = []
    for job in matchs:
        if job.state is not JobState.DONE or not job.stats or job.our_team is None:
            continue
        points.append(
            MatchPoint(
                match_name=job.match_name,
                date=job.created_at[:10],
                possession=_part(job.stats, "possession", job.our_team),
                control=_part(job.stats, "control", job.our_team),
                coverage=job.stats.get("coverage"),
            )
        )
    # Du plus ancien au plus récent : une tendance se lit dans ce sens.
    points.reverse()
    return Season(points)
