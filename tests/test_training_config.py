"""Le data.yaml de pose doit rester cohérent avec la géométrie du terrain.

Une incohérence ici ne se voit qu'après des heures d'entraînement.
"""
from __future__ import annotations

import yaml

from football_analysis.pitch.geometry import PITCH
from training.train_keypoints import write_data_yaml


def test_data_yaml_matches_geometry(tmp_path):
    config = yaml.safe_load(write_data_yaml(tmp_path).read_text())

    assert config["kpt_shape"] == [32, 3]
    assert config["flip_idx"] == PITCH.flip_index
    assert len(config["flip_idx"]) == len(PITCH.vertices)


def test_flip_idx_is_a_valid_permutation(tmp_path):
    """Ultralytics exige une permutation complète des indices."""
    config = yaml.safe_load(write_data_yaml(tmp_path).read_text())
    assert sorted(config["flip_idx"]) == list(range(32))
