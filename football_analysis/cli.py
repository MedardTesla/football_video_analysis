"""Point d'entrée en ligne de commande."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from dataclasses import replace

from .config import Config
from .pipeline import run
from .report import ReportMeta, write as write_report


def _afficher_progression():
    """Barre d'avancement sur une seule ligne.

    Une analyse dure des dizaines de minutes : sans retour, rien ne distingue
    un traitement en cours d'un processus figé.
    """
    debut = time.monotonic()

    def afficher(fraction: float) -> None:
        ecoule = time.monotonic() - debut
        restant = ecoule * (1 - fraction) / fraction if fraction > 0.01 else None
        pleine = int(fraction * 30)
        barre = "█" * pleine + "·" * (30 - pleine)
        fin = f" | reste ~{restant / 60:.0f} min" if restant else ""
        sys.stderr.write(f"\r  {barre} {fraction:5.1%}{fin}   ")
        sys.stderr.flush()
        if fraction >= 1.0:
            sys.stderr.write("\n")

    return afficher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="football-analysis",
        description="Analyse une vidéo de match : suivi, équipes, radar, statistiques.",
    )
    parser.add_argument("video", type=Path, help="vidéo source")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output_video/analysis.mp4")
    )
    parser.add_argument(
        "--no-radar", action="store_true", help="ne pas incruster le radar 2D"
    )
    parser.add_argument(
        "--match-name", default=None, help="titre affiché sur le rapport du club"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="ne pas générer le rapport HTML"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="images analysées par seconde (défaut 12 ; au-delà, coût doublé "
             "sans gain mesuré)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="ne pas afficher la progression"
    )
    args = parser.parse_args(argv)

    config = Config()
    if args.fps is not None:
        config.processing = replace(config.processing, sample_fps=args.fps)

    result = run(
        args.video, args.output, config,
        with_radar=not args.no_radar,
        on_progress=None if args.quiet else _afficher_progression(),
    )
    print(f"vidéo        : {result.video_path}")
    print(f"statistiques : {result.stats_path}")

    if not args.no_report:
        report = write_report(
            result.stats_path,
            args.output.with_suffix(".html"),
            ReportMeta(
                match_name=args.match_name or args.video.stem,
                played_on=date.today(),
            ),
            radar_png=result.radar_path,
        )
        print(f"rapport      : {report}")

    for team, share in sorted(result.stats["possession"].items()):
        print(f"  possession équipe {team} : {share:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
