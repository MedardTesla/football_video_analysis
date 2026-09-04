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

## Reprise après panne

Un worker tué — coupure de courant, machine GPU rendue — laisse un match en
« en cours ». Au démarrage et entre deux matchs, le worker remet en file ce qui
n'a plus donné signe de vie depuis 15 minutes. Les écritures de progression
servent de battement de cœur : un match qui avance n'est jamais repris.

`attempts` plafonne les reprises à trois. Sans ce plafond, un fichier qui fait
planter le worker serait relancé indéfiniment et bloquerait toute la file.

Surveillance : `/sante` répond **503** si des matchs sont bloqués. Une sonde
externe suffit alors à détecter un worker mort.

## Notification

Le club peut laisser une adresse au dépôt. Elle est facultative : sans elle, le
service fonctionne comme avant, le club revient sur son lien.

Configuration par variables d'environnement — sans elles, rien n'est envoyé et
le worker le signale au démarrage :

```
FA_BASE_URL=https://analyse.exemple.fr
FA_SMTP_HOST=smtp.exemple.fr
FA_SMTP_PORT=587
FA_SMTP_USER=...
FA_SMTP_PASSWORD=...
FA_SMTP_FROM=no-reply@exemple.fr
```

Deux points de conception :

- **Le message contient le lien privé.** Une adresse mal saisie l'envoie à un
  inconnu. Le formulaire l'écrit, et une adresse invalide est refusée plutôt
  qu'ignorée — l'ignorer ferait attendre au club un message qui ne viendrait
  jamais.
- **Un envoi raté n'annule pas une analyse réussie.** Le rapport existe, le
  lien fonctionne, seul l'avis manque. Perdre une analyse parce qu'un serveur
  SMTP est injoignable serait absurde.

Pour WhatsApp, plus pertinent en Afrique de l'Ouest, il suffira d'écrire une
classe respectant le protocole `Notifier` — le reste du service n'y touche pas.

## Protection du disque

Le dépôt est ouvert sans compte : n'importe qui peut envoyer 8 Go. Trois
garde-fous, parce qu'un disque plein arrête le service pour tous les clubs, y
compris ceux dont l'analyse est en cours et dont le travail serait perdu.

- **Réserve de 5 Go** : un dépôt est refusé si l'espace libre passe dessous,
  et le contrôle est refait *pendant* l'écriture — la taille annoncée par un
  client n'engage à rien, et plusieurs envois se partagent le même disque.
- **Rétention de 90 jours** : les dossiers de match plus anciens sont purgés
  par le worker, au démarrage et avant chaque analyse. La vidéo annotée est le
  seul poste qui grossit sans limite.
- **`/sante` répond 503** dès que la réserve est entamée, avec l'espace libre
  en gigaoctets.

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

Le pipeline échantillonne à 10 fps par défaut (`ProcessingConfig.sample_fps`),
soit deux fois plus de marge que le seuil où la mesure souffrirait.
