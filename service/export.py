"""Export des relevés d'un match en tableur.

Beaucoup de clubs tiennent déjà leurs statistiques à la main. Leur imposer de
recopier un rapport reviendrait à leur demander du travail en plus pour un
service censé leur en épargner.

Le fichier est en CSV avec séparateur point-virgule et en-tête BOM : c'est ce
qu'attend Excel en configuration française, et sans quoi les accents
s'affichent en caractères illisibles et tout tient dans une seule colonne.
"""
from __future__ import annotations

import csv
import io


def players_csv(stats: dict, names: dict[str, str] | None = None) -> str:
    """Un joueur par ligne, dans l'ordre du rapport."""
    names = names or {}
    tampon = io.StringIO()
    ecrivain = csv.writer(tampon, delimiter=";", lineterminator="\r\n")
    ecrivain.writerow(
        ["Numero", "Nom", "Equipe", "Distance_km", "Vitesse_max_kmh", "Temps_min"]
    )
    for joueur in stats.get("players") or []:
        equipe = joueur.get("team")
        ecrivain.writerow([
            joueur["track_id"],
            names.get(str(joueur["track_id"]), ""),
            {0: "A", 1: "B"}.get(equipe, ""),
            f"{joueur['distance_m'] / 1000:.2f}".replace(".", ","),
            f"{joueur['top_speed_ms'] * 3.6:.1f}".replace(".", ","),
            f"{joueur['seconds_seen'] / 60:.0f}",
        ])
    # BOM : sans lui Excel lit le fichier en latin-1 et casse les accents.
    return "\ufeff" + tampon.getvalue()
