# Déploiement

Deux processus, deux machines possibles, un volume partagé.

| | Machine | Rôle |
|---|---|---|
| `api` | petite, sans GPU | formulaire de dépôt, pages de suivi, rapports |
| `worker` | avec GPU | analyse des matchs |

Ils ne communiquent que par `FA_DATA_ROOT` : base SQLite et vidéos. Le worker
peut donc être éteint quand la file est vide — c'est le poste de coût dominant.

## Lancer

```bash
docker compose up -d api
docker compose up -d worker      # nécessite nvidia-container-toolkit
```

Les poids sont **montés**, pas copiés dans l'image : ils changent à chaque
réentraînement et pèsent plus lourd que le code. Placer avant de démarrer :

```
models/pitch_keypoints.pt
models/player_detection.pt
```

## Pourquoi deux images

L'API ne charge ni torch ni OpenCV — un test le vérifie
(`test_the_api_does_not_load_the_machine_learning_stack`). Elle tient donc dans
une image `python:3.12-slim` de quelques dizaines de mégaoctets, là où le
worker en pèse plusieurs gigaoctets.

Cette séparation a coûté un découplage : `service/settings.py` porte les
constantes partagées, et le worker importe le pipeline tardivement. Sans cela,
une simple constante lue par `/sante` imposait toute la pile de calcul à la
machine qui sert le formulaire.

## Exploitation

```
FA_ADMIN_TOKEN=une-chaîne-longue-et-imprévisible
```

Donne accès à `/admin/{ce jeton}` : liste des clubs, de leurs matchs et de
leurs liens privés. Indispensable au support — un club qui perd son lien n'a
sinon aucun recours.

Sans cette variable, la page n'existe pas.

## Surveillance

`/sante` répond **503** dès qu'un match reste bloqué plus de quinze minutes
sans progresser — signe que le worker est mort. Une sonde externe suffit à
être prévenu avant le club.

Le `HEALTHCHECK` de l'image API interroge ce même point d'entrée : un
conteneur marqué `unhealthy` signale donc un worker en panne, pas une API en
panne. C'est délibéré — l'API seule ne sert à rien si rien n'est analysé.

## Sauvegarde

Le volume `donnees` contient `jobs.db`, les rapports et les vidéos annotées.
Les vidéos sources en sont effacées dès le rapport produit. Sauvegarder
`jobs.db` suffit à ne pas perdre les liens remis aux clubs — les perdre
rendrait tous les rapports inaccessibles, aucun club n'ayant de compte.
