# Phase 0 — Analyse de vidéos réelles

Deux matchs, deux dispositifs de captation, **la même équipe** (ASKO joue dans
les deux). L'écart entre les deux mesure donc la captation, pas le football.

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

## Mesure : combien de frames permettent une homographie ?

36 images échantillonnées régulièrement sur les 2 h, jugées visuellement sur un
critère unique : y voit-on au moins 4 repères de terrain identifiables
(intersections de lignes, coins de surface, arcs de cercle, cadre de but) ?

| | |
|---|---|
| Frames exploitables | **17/36 = 47 %** (erreur-type 8 %, soit ~31-64 %) |
| 1re période | 7/18 = 39 % |
| 2e période | 10/18 = 56 % |
| Plus longue série sans repère | 4 échantillons consécutifs, soit **~12 minutes** |

### Pourquoi le jugement visuel plutôt qu'un détecteur

Une mesure automatique par transformée de Hough a été tentée d'abord, puis
abandonnée. Selon le réglage des seuils, le compte de lignes passait de 0 à 23
sur la même image, sans réglage intermédiaire séparant les vraies lignes des
poteaux de but, des ombres et des maillots clairs — sur une pelouse en plein
soleil, une ligne n'est que légèrement plus contrastée que l'herbe. Calibrer ce
détecteur aurait demandé autant de travail que le modèle de points clés qu'il
devait justement éviter. Les primitives sont conservées dans `pitch/lines.py`,
sans verdict et avec un avertissement explicite.

### Ce que ce chiffre implique

Les ~47 % ne sont pas le problème principal : le pipeline réutilise déjà la
dernière homographie valide. **La série de 12 minutes, si.** Sur un tel
intervalle la caméra a panoramiqué et zoomé plusieurs fois ; l'homographie
conservée n'a plus aucun rapport avec l'image, et les positions projetées
dérivent sans que rien ne le signale.

Conséquence à implémenter : périmer l'homographie au bout de quelques secondes
sans repère, et marquer ces intervalles comme non mesurés plutôt que de
produire des distances fausses. Un rapport qui annonce « 23 minutes non
analysables » reste vendable ; un rapport qui invente 3 km de course, non.

### Autre constat

Une des 36 images n'est pas du football : à 97', le flux diffuse une boîte de
dialogue système (« REPAIR IMAGE DB FILE »). La chaîne de traitement doit
tolérer des frames sans terrain du tout — le masque de pelouse les rejette
déjà, mais rien ne le signale en aval.

## Conclusion

L'angle de caméra n'est pas rédhibitoire, contrairement au risque anticipé. La
détection et la classification d'équipes fonctionnent bien sur cette source.

Mais avec 47 % de frames exploitables et des trous pouvant atteindre 12
minutes, les statistiques **individuelles** — distance parcourue, vitesse par
joueur — ne sont pas fiables sur ce type de captation. Les statistiques **par
équipe** et les séquences annotées le sont.

Ce que cela veut dire commercialement : vendre « la distance parcourue par
chaque joueur » exposerait à des chiffres faux. Vendre l'analyse d'équipe et la
vidéo annotée, avec les périodes non mesurées explicitement signalées, tient.

Deux façons d'améliorer le chiffre, par ordre de coût :
1. **Repositionner la caméra** chez les clubs pilotes — plus haut, plus reculée.
   Gratuit, et c'est le levier le plus puissant.
2. Entraîner le modèle de points clés, qui reconnaît aussi des repères
   sémantiques qu'un détecteur de lignes ignore (point de penalty, arcs).


---

# Second match : DJOLIBA x ASKO (Ligue des Champions CAF)

Source : YouTube `0d472EWLOf0`, 1920×1080 à 50 fps, 97 min. Caméra en tribune
haute, plans larges, stade quasi vide, piste d'athlétisme autour du terrain.
Même protocole : 36 images échantillonnées, même critère de jugement.

## Comparaison

| | ASKO x Barracuda | Djoliba x ASKO |
|---|---|---|
| Résolution | 1280×720 / 30 fps | **1920×1080 / 50 fps** |
| Caméra | Bord de touche, hauteur d'homme | **Tribune haute** |
| Frames exploitables | **47 %** (±8 %) | **≥ 92 %** (36/36, 0 échec) |
| Plus longue série sans repère | ~12 min | aucune observée |
| Personnes hors terrain écartées | ~33 % | 15 % |
| Ballon détecté (YOLO générique) | 1/5 images | **7/10 images** |
| Taille du ballon | 13×12 px | 12 à 19 px |
| Hauteur médiane d'un joueur | 96 à 137 px | 84 à 135 px |

Le taux de 92 % est une borne basse : avec zéro échec sur 36 tirages, la règle
de trois plafonne le taux d'échec à 3/36 ≈ 8 %.

## Ce que cette comparaison démontre

**La captation est le levier dominant, et de loin.** Même équipe, même
championnat, même code d'analyse sans une ligne modifiée : la disponibilité de
l'homographie passe de 47 % à plus de 92 % uniquement en changeant la position
de la caméra et la résolution.

Aucun modèle, aucun réentraînement, aucune optimisation ne rattrapera cet
écart. Un cahier des charges de captation vaut plus que des mois d'ingénierie :

- caméra en hauteur, au-dessus du niveau du terrain ;
- plan large, montrant au moins un tiers du terrain ;
- 1080p minimum — à 720p le ballon disparaît ;
- caméra reculée, sans banc ni abri dans le champ.

## Validation du masque adaptatif

La teinte du gazon mesurée couvre **H = 41 à 63** sur les deux matchs, soit 22
degrés d'amplitude. Aucune borne fixe ne couvre les deux stades : le choix
d'estimer la teinte dominante sur chaque image, pris sur le premier match, est
directement validé par le second. Le masque a fonctionné sans réglage sur une
pelouse à bandes de tonte, un stade vide et une piste d'athlétisme.


---

# Première exécution réelle du classifieur d'équipes

507 vignettes de joueurs extraites du match Djoliba x ASKO (40 instants sur 8
minutes), passées dans SigLIP + UMAP + K-Means. Premier passage du composant
sur de vraies images.

## Deux bugs, trouvés en exécutant

1. `AutoProcessor` charge le tokenizer de SigLIP, qui exige SentencePiece et
   échoue à l'import. On n'utilise que le volet image : `AutoImageProcessor`
   suffit.
2. `get_image_features` renvoie un objet de sortie en transformers 5, pas un
   tenseur. L'embedding 768-D est dans `pooler_output`.

Aucun des deux n'était détectable sans exécuter le code.

## Le défaut de conception

Avec deux clusters, les arbitres en turquoise atterrissaient dans le cluster de
l'équipe en rouge — environ une vignette sur cinq de ce groupe. K-Means ne sait
pas dire « ni l'un ni l'autre ».

Première parade, **échouée** : rejeter les points trop éloignés des centroïdes.
Sur les 507 vignettes, le seuil retenu en écartait exactement zéro. Les
arbitres étant présents à l'ajustement, UMAP les place dans la dispersion
normale du nuage.

Parade retenue : regrouper en trois clusters, garder les deux plus peuplés
comme équipes. Résultat mesuré :

| | k=2 | k=3, deux plus gros |
|---|---|---|
| Équipe A | 302 (60 %) | 202 (40 %) |
| Équipe B | 205 (40 %) | 195 (38 %) |
| Non attribué | 0 | **110 (22 %)** |

Le troisième groupe contient les arbitres, les gardiens en violet et bleu, et
des vignettes trop floues pour être jugées. Contrôle visuel des deux groupes
d'équipe : 24 vignettes sur 24 du bon maillot, aucun arbitre.

Les 22 % écartés incluent de vrais joueurs sur images floues. C'est le prix
assumé : perdre une détection douteuse coûte moins cher que d'attribuer un
arbitre à une équipe.

## Coût de calcul mesuré

Sur CPU (8 cœurs), SigLIP base traite **3,2 vignettes par seconde**. Avec une
vingtaine de joueurs par frame, cela fait ~6 s de classification par frame.
Un match complet est hors de portée sans GPU — ce n'est pas une optimisation
à faire plus tard, c'est une condition d'exécution.
