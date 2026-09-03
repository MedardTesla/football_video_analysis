"""Entraînement du détecteur joueurs / gardiens / arbitres / ballon.

L'entrée est étirée en 1280x1280 depuis du 1920x1080. C'est délibéré : le
ballon ne fait que quelques pixels sur une vue de diffusion, et le
redimensionnement classique en 640 le fait disparaître. La déformation
d'aspect n'est pas un problème — le modèle apprend la géométrie déformée, et
l'inférence applique le même étirement.

Usage :
    python -m training.train_detection --data datasets/players/data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path


def train(
    data: Path,
    model: str = "yolov8x.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 4,
    project: str = "runs/detect",
) -> None:
    from ultralytics import YOLO

    YOLO(model).train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        # La mosaïque aide ici, contrairement au modèle de pose : elle
        # multiplie les contextes de joueurs. Coupée en fin d'entraînement
        # pour que le modèle finisse sur des images réalistes.
        mosaic=1.0,
        close_mosaic=10,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolov8x.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args(argv)

    train(args.data, args.model, args.epochs, args.imgsz, args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
