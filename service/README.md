# Service d'analyse

Trois écrans : déposer une vidéo, suivre l'analyse, lire le rapport. Pas de
compte à créer — le club reçoit un lien porteur d'un jeton imprévisible.

## Lancer

```bash
# API : petite machine, pas de GPU
uvicorn service.api:app --host 0.0.0.0 --port 8000

# Worker : machine GPU, séparée
python -m service.run_worker
```

Les deux ne partagent que `FA_DATA_ROOT` (défaut `data/service/`), qui contient
la base SQLite et les vidéos. Cette séparation permet d'éteindre le GPU quand
la file est vide — c'est le poste de coût dominant.

## Ce qui est délibéré

**Pas d'authentification.** Un club de village ne créera pas de compte pour
trois matchs par saison. Le jeton du lien joue ce rôle et se remplace par une
vraie authentification le jour où un client la demande.

**La vidéo source est supprimée** dès le rapport produit. C'est le poste de
stockage dominant, et ce sont les images du club.

**Les messages d'erreur s'adressent à un club**, pas à un développeur : la
trace complète va dans les logs.

## Coût de traitement mesuré

| Échantillonnage | Frames sur 90 min | T4 | A10G |
|---|---|---|---|
| 25 fps | 135 000 | 188 min | 75 min |
| 10 fps | 54 000 | 75 min | 30 min |
| 5 fps | 27 000 | 38 min | 15 min |

Sous-échantillonner ne coûte presque rien en précision : à 11 km parcourus par
match, passer de 25 à 5 fps sous-estime la distance de **0,3 %**. Ce n'est pas
la mesure qui fixe la fréquence, c'est le suivi — il lui faut du recouvrement
entre images consécutives pour conserver les identités. Le plancher pratique
reste à déterminer une fois les modèles disponibles.
