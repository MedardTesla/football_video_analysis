"""Configuration centrale : chemins, seuils, constantes du pipeline.

Toute valeur réglable vit ici — aucun chemin ni seuil en dur dans les modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
INPUT_DIR = ROOT / "input_video"
OUTPUT_DIR = ROOT / "output_video"
CACHE_DIR = ROOT / "stubs"


@dataclass
class DetectionConfig:
    """Détection joueurs / arbitres / gardiens / ballon."""

    weights: Path = MODELS_DIR / "player_detection.pt"
    # Suréchantillonnage volontaire : le ballon ne fait qu'une douzaine de
    # pixels (mesuré à 13x12 px sur une source 720p, voir ANALYSE_TERRAIN.md).
    # Descendre sous la largeur native de la source le fait disparaître.
    imgsz: int = 1280
    confidence: float = 0.3
    # NMS agnostique de classe : un même joueur détecté à la fois comme
    # `player` et `goalkeeper` ne doit produire qu'une seule boîte.
    nms_iou: float = 0.5
    class_agnostic_nms: bool = True
    batch_size: int = 16


@dataclass
class TrackingConfig:
    """ByteTrack — joueurs, arbitres, gardiens. Le ballon en est exclu."""

    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 25


@dataclass
class TeamConfig:
    """Classification d'équipes : SigLIP -> UMAP -> K-Means."""

    siglip_model: str = "google/siglip-base-patch16-224"
    embedding_batch_size: int = 32
    umap_components: int = 3
    n_teams: int = 2
    # Cluster supplémentaire pour absorber arbitres et gardiens : sans lui,
    # ils sont assignés de force à une équipe. Voir teams/classifier.py.
    extra_clusters: int = 1
    # Nombre de frames échantillonnées pour ajuster le classifieur une fois
    # pour toutes en début de match.
    fit_stride: int = 30
    fit_max_frames: int = 40


@dataclass
class PitchConfig:
    """Détection des points clés du terrain (YOLOv8-pose, 32 keypoints)."""

    weights: Path = MODELS_DIR / "pitch_keypoints.pt"
    # Confiance minimale d'un point clé pour servir à l'homographie.
    confidence: float = 0.5
    # Confiance minimale de la boîte englobant le terrain, volontairement
    # basse. Le modèle ne connaît qu'une classe : il n'y a pas de faux positif
    # à craindre, et sa confiance de boîte s'effondre dès que le terrain est
    # partiellement hors champ — mesuré entre 0,05 et 0,89 sur des images où
    # les points clés restaient bons. Le seuil par défaut d'Ultralytics (0,25)
    # jetait l'instance entière, points compris.
    instance_confidence: float = 0.02
    # findHomography exige >= 4 correspondances ; on en demande plus pour
    # éviter les homographies dégénérées sur points quasi colinéaires.
    min_keypoints: int = 6
    # Lissage de la matrice d'homographie sur fenêtre glissante.
    homography_window: int = 5
    # Durée au-delà de laquelle une homographie sans repère frais est
    # considérée périmée. Deux secondes : sur une caméra qui suit le jeu,
    # c'est déjà un panoramique complet. Passé ce délai, les frames sont
    # marquées non mesurées plutôt que projetées au hasard.
    homography_max_age_s: float = 2.0
    # Écarte les personnes hors pelouse : staff, remplaçants, spectateurs.
    # Sur une caméra de bord de touche, ils représentent un tiers des
    # détections. Le masque est recalculé tous les N frames, la caméra
    # bougeant lentement devant une pelouse qui, elle, ne bouge pas.
    use_turf_mask: bool = True
    turf_mask_interval: int = 10


@dataclass
class BallConfig:
    """Nettoyage et lissage de la trajectoire du ballon."""

    # Un ballon ne parcourt pas plus de 5 m entre deux frames consécutives :
    # au-delà, c'est une fausse détection.
    max_displacement_cm: float = 500.0
    smoothing_window: int = 5


@dataclass
class ProcessingConfig:
    """Fréquence de traitement, principal levier sur le coût GPU."""

    # Images traitées par seconde de match. Traiter les 25 ou 50 images d'une
    # seconde n'apporte presque rien : à 11 km parcourus par match, descendre
    # de 25 à 5 fps ne sous-estime la distance que de 0,3 % (mesuré, voir
    # service/README.md). Ce n'est donc pas la mesure qui fixe cette valeur
    # mais le suivi, qui a besoin de recouvrement entre images consécutives
    # pour conserver les identités.
    #
    # 10 fps est un compromis prudent : deux fois plus de marge que le seuil
    # où la mesure commencerait à souffrir, et cinq fois moins de calcul qu'un
    # traitement intégral. `None` traite toutes les images.
    sample_fps: float | None = 10.0

    # Fréquence des remontées de progression, en images. Écrire en base à
    # chaque image saturerait SQLite pendant que l'API lit.
    progress_every: int = 250


@dataclass
class Config:
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    teams: TeamConfig = field(default_factory=TeamConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    ball: BallConfig = field(default_factory=BallConfig)


# Identifiants de classe du modèle de détection.
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
