"""Entraînement du modèle de points clés du terrain (YOLOv8-pose, 32 keypoints).

Deux réglages d'augmentation sont critiques et n'ont pas de valeur par défaut
correcte pour cette tâche :

- `mosaic=0` : l'augmentation mosaïque colle quatre images en une. Le modèle
  apprend alors à chercher plusieurs terrains sur une même image, ce qui est
  exactement ce qu'on ne veut pas d'un détecteur de terrain unique.
- `flip_idx` : sans la table de symétrie, `fliplr` mirore l'image sans
  permuter les labels. Le coin haut gauche devient le coin haut droit mais
  garde son ancien indice. Panne silencieuse : l'entraînement converge et le
  modèle prédit n'importe quoi.

Usage :
    python -m training.train_keypoints --data datasets/pitch/data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from football_analysis.pitch.geometry import PITCH


def write_data_yaml(dataset_root: Path, output: Path | None = None) -> Path:
    """Génère le data.yaml de pose, cohérent avec `pitch/geometry.py`.

    `kpt_shape` est [32, 3] : x, y et visibilité. La visibilité est
    indispensable — sur une vue de diffusion, la moitié des points du terrain
    est hors champ, et il faut pouvoir les marquer absents plutôt qu'à zéro.
    """
    output = output or dataset_root / "data.yaml"
    config = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "kpt_shape": [len(PITCH.vertices), 3],
        "flip_idx": PITCH.flip_index,
        "names": {0: "pitch"},
    }
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    return output


def train(
    data: Path,
    model: str = "yolov8x-pose.pt",
    epochs: int = 500,
    imgsz: int = 640,
    batch: int = 8,
    project: str = "runs/pose",
) -> None:
    from ultralytics import YOLO

    YOLO(model).train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        # Voir le module : ces deux valeurs ne sont pas négociables.
        mosaic=0.0,
        fliplr=0.5,
        # Le terrain est un plan rigide : les déformations agressives
        # produisent des géométries qui n'existent pas dans la réalité.
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        scale=0.3,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="data.yaml de pose")
    parser.add_argument("--model", default="yolov8x-pose.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--write-config",
        type=Path,
        help="génère data.yaml depuis la racine du dataset, puis sort",
    )
    args = parser.parse_args(argv)

    if args.write_config:
        path = write_data_yaml(args.write_config)
        print(f"écrit : {path}")
        return 0

    train(args.data, args.model, args.epochs, args.imgsz, args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
