"""Classification d'équipes : SigLIP -> UMAP -> K-Means.

La couleur moyenne des pixels d'un joueur recadré échoue en pratique : le fond
(pelouse, tribunes), la pose et l'éclairage variable selon la zone du terrain
dominent le signal du maillot. On passe donc par des embeddings appris.

Le classifieur s'ajuste UNE fois sur un échantillon de frames en début de
match, puis prédit par lot : refaire tourner UMAP à chaque frame donnerait des
clusters instables d'une frame à l'autre.
"""
from __future__ import annotations

import numpy as np


class TeamClassifier:
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        batch_size: int = 32,
        n_components: int = 3,
        n_teams: int = 2,
        device: str | None = None,
    ) -> None:
        import torch
        from sklearn.cluster import KMeans
        from transformers import AutoModel, AutoProcessor
        import umap

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        # UMAP plutôt que PCA : conserve les relations locales, ce qui sépare
        # mieux deux maillots de teintes proches.
        self.reducer = umap.UMAP(n_components=n_components)
        self.cluster = KMeans(n_clusters=n_teams, n_init=10)
        self._fitted = False

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embeddings 768-D des vignettes joueurs (BGR, telles que découpées)."""
        from PIL import Image

        vectors = []
        for start in range(0, len(crops), self.batch_size):
            batch = [
                Image.fromarray(crop[:, :, ::-1]) for crop in crops[start : start + self.batch_size]
            ]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                features = self.model.get_image_features(**inputs)
            vectors.append(features.cpu().numpy())
        return np.concatenate(vectors) if vectors else np.empty((0, 768))

    def fit(self, crops: list[np.ndarray]) -> "TeamClassifier":
        embeddings = self._embed(crops)
        projected = self.reducer.fit_transform(embeddings)
        self.cluster.fit(projected)
        self._fitted = True
        return self

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("appeler fit() sur un échantillon de frames d'abord")
        if not crops:
            return np.empty(0, dtype=int)
        projected = self.reducer.transform(self._embed(crops))
        return self.cluster.predict(projected)


def assign_goalkeeper(
    goalkeeper_xy: np.ndarray, players_xy: np.ndarray, team_ids: np.ndarray
) -> int:
    """Rattache un gardien à une équipe par proximité au centroïde.

    Le gardien porte un maillot différent : SigLIP le classera de façon
    arbitraire. On utilise donc la position — un gardien est toujours dans la
    moitié de terrain de son équipe, donc proche de son centroïde.
    """
    centroids = np.stack(
        [players_xy[team_ids == team].mean(axis=0) for team in (0, 1)]
    )
    distances = np.linalg.norm(centroids - goalkeeper_xy, axis=1)
    return int(np.argmin(distances))
