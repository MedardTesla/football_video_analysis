"""Classification d'équipes : SigLIP -> UMAP -> K-Means.

La couleur moyenne des pixels d'un joueur recadré échoue en pratique : le fond
(pelouse, tribunes), la pose et l'éclairage variable selon la zone du terrain
dominent le signal du maillot. On passe donc par des embeddings appris.

Le classifieur s'ajuste UNE fois sur un échantillon de frames en début de
match, puis prédit par lot : refaire tourner UMAP à chaque frame donnerait des
clusters instables d'une frame à l'autre.

K-Means à deux classes ne sait pas dire « ni l'un ni l'autre » : tout ce qu'on
lui donne atterrit dans une équipe. Vérifié sur match réel, les arbitres en
turquoise tombaient dans le cluster de l'équipe en rouge. Le détecteur est
censé les séparer en amont, mais une erreur de sa part polluerait alors
silencieusement les statistiques d'équipe.

On regroupe donc en `n_teams + 1` clusters et on ne retient que les deux plus
peuplés comme équipes. Sur un terrain il y a toujours un troisième groupe
vestimentaire — arbitres, et gardiens qui portent un maillot distinct.

Un rejet par distance au centroïde a été essayé d'abord, et abandonné : les
arbitres étant présents à l'ajustement, UMAP les plaçait dans la dispersion
normale et aucun n'était rejeté. Sur 507 vignettes réelles, le seuil retenu en
écartait exactement zéro.
"""

from __future__ import annotations

import numpy as np

# Rendu par `predict` quand une vignette n'appartient clairement à aucune
# des deux équipes : arbitre, gardien, ou détection erronée.
UNASSIGNED = -1


class TeamClassifier:
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        batch_size: int = 32,
        n_components: int = 3,
        n_teams: int = 2,
        device: str | None = None,
        extra_clusters: int = 1,
    ) -> None:
        import torch
        from sklearn.cluster import KMeans
        from transformers import AutoImageProcessor, AutoModel
        import umap

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._torch = torch
        # AutoImageProcessor et non AutoProcessor : ce dernier charge aussi le
        # tokenizer de SigLIP, qui exige SentencePiece et échoue à l'import.
        # On n'utilise que `get_image_features` — le volet texte est inutile.
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        # UMAP plutôt que PCA : conserve les relations locales, ce qui sépare
        # mieux deux maillots de teintes proches.
        self.reducer = umap.UMAP(n_components=n_components)
        self.n_teams = n_teams
        self.cluster = KMeans(n_clusters=n_teams + extra_clusters, n_init=10)
        # cluster K-Means -> identifiant d'équipe, ou UNASSIGNED.
        self._team_of_cluster: dict[int, int] = {}
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
            # transformers >= 5 renvoie un objet de sortie ; les versions
            # antérieures renvoyaient directement le tenseur.
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            vectors.append(features.cpu().numpy())
        return np.concatenate(vectors) if vectors else np.empty((0, 768))

    def fit(self, crops: list[np.ndarray]) -> "TeamClassifier":
        embeddings = self._embed(crops)
        projected = self.reducer.fit_transform(embeddings)
        self.cluster.fit(projected)

        # Les deux groupes les plus peuplés sont les équipes : sur un match,
        # 22 joueurs contre 3 officiels, l'ordre de taille est fiable.
        sizes = np.bincount(self.cluster.labels_, minlength=self.cluster.n_clusters)
        ranked = np.argsort(-sizes)
        self._team_of_cluster = {
            int(cluster): (team if team < self.n_teams else UNASSIGNED)
            for team, cluster in enumerate(ranked)
        }
        self._fitted = True
        return self

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("appeler fit() sur un échantillon de frames d'abord")
        if not crops:
            return np.empty(0, dtype=int)

        projected = self.reducer.transform(self._embed(crops))
        clusters = self.cluster.predict(projected)
        return np.array([self._team_of_cluster[int(c)] for c in clusters])


def assign_goalkeeper(
    goalkeeper_xy: np.ndarray, players_xy: np.ndarray, team_ids: np.ndarray
) -> int:
    """Rattache un gardien à une équipe par proximité au centroïde.

    Le gardien porte un maillot différent : SigLIP le classera de façon
    arbitraire. On utilise donc la position — un gardien est toujours dans la
    moitié de terrain de son équipe, donc proche de son centroïde.
    """
    centroids = []
    for team in (0, 1):
        members = players_xy[team_ids == team]
        if len(members) == 0:
            # Aucun joueur reconnu pour cette équipe sur la frame : on ne
            # peut pas décider, plutôt que de deviner.
            return UNASSIGNED
        centroids.append(members.mean(axis=0))
    distances = np.linalg.norm(np.stack(centroids) - goalkeeper_xy, axis=1)
    return int(np.argmin(distances))
