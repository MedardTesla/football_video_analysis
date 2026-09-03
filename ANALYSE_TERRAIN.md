# Phase 0 — Analyse d'une vidéo réelle

Source : ASKO vs AC Barracuda, D1 Lonato J23 (YouTube, `hwKSqtpk_a4`).
Méthode : 5 images échantillonnées sur les 2 h du match (7', 25', 45', 70', 90'),
passées dans un YOLOv8x générique COCO — aucun modèle spécialisé, donc les
chiffres ci-dessous sont un **plancher**, pas un plafond.

## Caractéristiques de la source

| | |
|---|---|
| Résolution | 1280×720, 30 fps |
| Durée | 7151 s (~2 h) |
| Caméra | Unique, en bord de touche, à hauteur d'homme. Panoramique et zoom. |
| Maillots | ASKO jaune/noir, Barracuda bordeaux. Arbitres en rouge. |

## Résultats mesurés

| Image | Personnes | Sur le terrain | Écartées | Hauteur médiane |
|---|---|---|---|---|
| 07' | 19 | 11 | 8 | 96 px |
| 25' | 18 | 12 | 6 | 137 px |
| 45' | 15 | 9 | 6 | 113 px |
| 70' | 19 | 11 | 8 | 114 px |
| 90' | 23 | 18 | 5 | 101 px |

Ballon détecté sur **1 image sur 5**, à 13×12 px.

## Ce qui fonctionne

**La détection des joueurs.** Un modèle générique, non entraîné sur du football,
trouve déjà tous les joueurs visibles, y compris à 30 px de haut. Un modèle
spécialisé fera nettement mieux.

**La classification d'équipes.** Jaune vif contre bordeaux, avec des boîtes de
100 px de haut en médiane : largement assez de pixels pour SigLIP. C'est le cas
favorable, pas le cas limite.

**Le suivi.** La caméra bouge en permanence — d'où l'importance de la
compensation de mouvement caméra déjà activée dans BoT-SORT.

## Ce qui pose problème

**Un tiers des personnes détectées ne sont pas des joueurs.** Entraîneurs
debout, remplaçants sur le banc, spectateurs dans les gradins. Traité par
`pitch/mask.py` : masque de pelouse par couleur, indépendant de l'homographie.
Limite résiduelle assumée — un remplaçant assis dans l'herbe reste compté.

**Le ballon.** 1 détection sur 5 images avec un modèle générique. À 720p il ne
fait qu'une douzaine de pixels. La possession en dépend directement.
Pistes : `imgsz` supérieur à la résolution native pour suréchantillonner, ou
inférence par tuiles sur la zone de jeu.

**L'homographie sera intermittente.** À 45' presque aucune ligne n'est visible ;
à 90' on voit surface de but, ligne de touche et rond central. Le pipeline
réutilise déjà la dernière homographie valide, mais sur cette source une part
des frames n'aura aucune projection exploitable.

**Le 720p invalide une hypothèse d'origine.** L'étirement 1920×1080 → 1280×1280
supposait du 1080p. Ici la source est déjà en 1280 de large : il n'y a plus de
suréchantillonnage, et le ballon reste à sa taille native.

## Conclusion

L'angle de caméra n'est pas rédhibitoire, contrairement au risque anticipé. Les
statistiques **par équipe** (possession territoriale, zones de contrôle) sont
atteignables. Les statistiques **individuelles** (distance, vitesse par joueur)
resteront fragiles tant que l'homographie est intermittente.

Prochaine étape utile : mesurer le taux de frames avec homographie exploitable
sur un match complet. C'est ce chiffre qui décide de ce qu'on peut vendre.
