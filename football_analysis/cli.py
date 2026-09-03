"""Point d'entrée en ligne de commande."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .config import Config
from .pipeline import run
from .report import ReportMeta, write as write_report


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
    args = parser.parse_args(argv)

    result = run(args.video, args.output, Config(), with_radar=not args.no_radar)
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
