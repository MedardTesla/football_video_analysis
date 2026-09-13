# Cahier des charges de captation

Document destiné aux clubs. Il dit **comment filmer** pour qu'une analyse soit
exploitable, et ce qui change dans le rapport quand ces conditions ne sont pas
réunies.

## Pourquoi ce document existe

Deux matchs de la même équipe ont été analysés avec le **même code, sans une
ligne modifiée** (`ANALYSE_TERRAIN.md`). Seul le dispositif de captation
changeait.

| | Caméra de bord de touche | Caméra en tribune haute |
|---|---|---|
| Résolution | 1280×720, 30 fps | 1920×1080, 50 fps |
| Position | Bord de touche, hauteur d'homme | Tribune haute, plans larges |
| **Terrain localisable** | **47 %** des images (±8 %) | **≥ 92 %** (36/36) |
| Plus long trou sans repère | ~12 minutes | aucun observé |
| Personnes hors terrain à écarter | ~33 % | 15 % |
| Ballon détecté | 1 image sur 5 | 7 images sur 10 |

La qualité d'analyse passe donc du simple au double **par le seul choix de
l'emplacement de la caméra**. Aucun modèle, aucun réentraînement, aucune
optimisation logicielle ne rattrape cet écart. C'est le levier le moins cher et
le plus puissant dont dispose un club : il est gratuit.

## Les cinq règles

### 1. Filmer en hauteur

**La règle :** placer la caméra au-dessus du niveau du terrain — tribune, toit
de vestiaire, plateforme, échafaudage, mât. Plus haut vaut mieux.

**Pourquoi :** le logiciel calcule les positions réelles en repérant les lignes
du terrain (rond central, surfaces, lignes de touche). À hauteur d'homme, ces
lignes se confondent en une bande étroite et deviennent invisibles. C'est
l'unique cause de l'écart 47 % / 92 % ci-dessus.

**Sans cela :** les positions ne sont pas calculables sur une grande partie du
match, et les distances parcourues perdent leur sens.

### 2. Cadrer large

**La règle :** montrer en permanence au moins un tiers du terrain, avec des
lignes visibles dans l'image.

**Pourquoi :** un plan serré sur le ballon est agréable à regarder mais ne
contient aucun repère de terrain. Le logiciel ne peut alors plus situer les
joueurs.

**Sans cela :** les séquences en plan serré sont marquées non mesurées. Elles
apparaissent quand même dans la vidéo annotée, mais ne comptent pas dans les
statistiques.

### 3. Filmer en 1080p au minimum

**La règle :** 1920×1080. Ni 720p, ni « HD » non précisé.

**Pourquoi :** le ballon mesure entre 12 et 19 pixels de large sur une source
1080p. À 720p il tombe à 13×12 pixels et disparaît quatre images sur cinq. La
possession se calcule à partir de sa position.

**Sans cela :** la possession devient peu fiable. Le reste de l'analyse tient.

### 4. Reculer la caméra

**La règle :** aucun banc de touche, abri, panneau ni groupe de spectateurs
dans le champ, autant que possible.

**Pourquoi :** sur la captation de bord de touche, **un tiers des personnes
détectées n'étaient pas des joueurs** — entraîneurs debout, remplaçants,
public. Le logiciel écarte automatiquement ce qui est hors pelouse, mais un
remplaçant assis dans l'herbe reste compté.

**Sans cela :** des identités parasites apparaissent dans les statistiques.

### 5. Limiter les changements de zoom

**La règle :** un cadrage stable vaut mieux qu'un suivi serré du ballon. Si la
caméra doit suivre le jeu, privilégier le panoramique au zoom.

**Pourquoi :** mesuré sur un extrait 1080p de diffusion, le modèle qui repère
le terrain perd toute accroche quand le cadrage s'éloigne trop de l'échelle sur
laquelle il a appris. Sur cet extrait, sept secondes de plan d'ensemble
rendaient **zéro repère sur 32**, alors que les plans resserrés du même match en
donnaient treize. Le logiciel compense désormais en réessayant à une seconde
échelle, ce qui a ramené la couverture de 73 % à 100 % sur l'échantillon
mesuré — mais cette compensation a un coût de calcul, et rien ne garantit
qu'elle couvre tous les cadrages.

**Sans cela :** la couverture baisse par intermittence, sans que ce soit
visible à l'œil sur la vidéo.

## Fiche de contrôle avant un match

- [ ] Caméra au-dessus du niveau du terrain
- [ ] Au moins un tiers du terrain visible en permanence
- [ ] Résolution réglée sur 1920×1080 ou mieux
- [ ] Banc, abri et public hors du champ
- [ ] Trépied ou support stable, cadrage fixe privilégié
- [ ] Batterie et carte mémoire pour la durée complète du match
- [ ] Un essai de deux minutes filmé et vérifié **avant** le coup d'envoi

Le dernier point est le plus rentable : deux minutes d'essai évitent de
découvrir après le match qu'un poteau masquait le rond central.

## Ce que le club obtient, selon la captation

| | Captation conforme | Captation de bord de touche |
|---|---|---|
| Vidéo annotée (joueurs, équipes, ballon) | Oui | Oui |
| Statistiques par équipe (possession, zones de contrôle) | Oui | Oui |
| Distance et vitesse **par joueur** | Oui | **Non fiables** |
| Part du match effectivement mesurée | > 90 % | ~50 % |

Cette distinction n'est pas commerciale, elle est mesurée. Sur une captation de
bord de touche, avec la moitié des images sans repère et des trous pouvant
atteindre douze minutes, une distance individuelle serait un chiffre inventé.

Le rapport indique toujours la part du match réellement mesurée (`couverture`),
et **les totaux sont des planchers** : une distance affichée est toujours
inférieure ou égale à la réalité, jamais extrapolée.

## Ce que nous ne promettons pas

- **Une couverture de 100 %.** Aucune captation ne l'atteint. Le rapport dit
  toujours ce qui n'a pas été mesuré.
- **Une identité stable pour chaque joueur sur 90 minutes.** Un joueur perdu
  puis retrouvé est recollé quand c'est physiquement plausible ; sinon il
  apparaît comme deux identités, ce que le rapport signale.
- **La reconnaissance des joueurs par leur nom.** Le club nomme lui-même les
  identités après l'analyse.
- **Une analyse tactique.** Le logiciel mesure des positions et des distances.
  L'interprétation reste à l'entraîneur.

## Si le club ne peut pas filmer en hauteur

C'est le cas fréquent, et ce n'est pas rédhibitoire. Dans l'ordre, du plus
efficace au moins efficace :

1. **Trouver trois mètres de hauteur.** Un toit de vestiaire, une plateforme
   d'échafaudage, un mât d'éclairage accessible, une nacelle empruntée. Trois
   mètres suffisent à séparer les lignes ; ce n'est pas une tribune qu'il faut.
2. **Reculer et cadrer plus large**, même sans gagner en hauteur. Une partie de
   l'effet vient du cadrage, pas seulement de l'altitude.
3. **Accepter le périmètre réduit** : vidéo annotée et statistiques d'équipe,
   sans les distances individuelles. C'est déjà ce que la plupart des clubs de
   ce niveau n'ont pas du tout aujourd'hui.

L'option 3 reste un produit vendable. Ce qui ne l'est pas, c'est de livrer des
distances individuelles calculées sur la moitié d'un match en laissant croire
qu'elles couvrent le tout.
