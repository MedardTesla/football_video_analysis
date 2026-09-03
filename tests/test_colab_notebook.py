"""Le notebook Colab duplique la géométrie du terrain.

Colab n'a pas le dépôt : le notebook doit être autonome. Cette duplication
peut donc dériver de `pitch/geometry.py`, et une divergence produirait un
modèle entraîné contre une géométrie différente de celle du pipeline — sans
aucune erreur visible. Ce test l'interdit.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from football_analysis.pitch.geometry import PITCH

NOTEBOOK = Path(__file__).resolve().parent.parent / "training" / "train_keypoints_colab.ipynb"


@pytest.fixture(scope="module")
def cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


@pytest.fixture(scope="module")
def code(cells):
    return "\n".join(src for src, kind in cells if kind == "code")


def test_every_code_cell_parses(cells):
    for src, kind in cells:
        if kind == "code" and not src.strip().startswith("%"):
            ast.parse(src)


def test_embedded_geometry_matches_the_source(code):
    """Les 32 sommets du notebook doivent être ceux de geometry.py."""
    namespace: dict = {}
    start = code.index("L, W = ")
    end = code.index("CIBLE = np.array")
    exec(code[start:end], namespace)
    assert namespace["VERTICES"] == PITCH.vertices


def test_embedded_dimensions_match_the_source(code):
    namespace: dict = {}
    start = code.index("L, W = ")
    exec(code[start : code.index("VERTICES = [")], namespace)
    assert namespace["L"] == PITCH.length
    assert namespace["W"] == PITCH.width
    assert namespace["PB_L"] == PITCH.penalty_box_length
    assert namespace["PB_W"] == PITCH.penalty_box_width
    assert namespace["GB_L"] == PITCH.goal_box_length
    assert namespace["GB_W"] == PITCH.goal_box_width
    assert namespace["R"] == PITCH.centre_circle_radius
    assert namespace["SPOT"] == PITCH.penalty_spot_distance


def test_expected_flip_index_matches_the_source(code):
    namespace: dict = {}
    start = code.index("ATTENDU = [")
    exec(code[start : code.index("yaml_path = ")], namespace)
    assert namespace["ATTENDU"] == PITCH.flip_index


def test_mosaic_is_disabled(code):
    """L'augmentation mosaïque apprend au modèle à chercher plusieurs
    terrains par image. C'est le réglage le plus facile à perdre."""
    assert "mosaic=0.0" in code


def test_rigid_plane_augmentations_are_disabled(code):
    for reglage in ("degrees=0.0", "shear=0.0", "perspective=0.0"):
        assert reglage in code, reglage


def test_horizontal_flip_stays_enabled(code):
    """Vraie symétrie du terrain, et flip_idx permute les labels."""
    assert "fliplr=0.5" in code


def test_the_key_is_never_written_in_clear(code):
    assert "getpass" in code
    assert "api_key='" not in code.replace(" ", "")


def test_confidence_thresholds_match_the_pipeline(code):
    from football_analysis.config import PitchConfig

    cfg = PitchConfig()
    assert f"SEUIL = {cfg.confidence}" in code
    assert f"MINI = {cfg.min_keypoints}" in code
