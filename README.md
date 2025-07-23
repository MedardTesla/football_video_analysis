# ⚽ Système d’Analyse de Match de Football avec YOLO, Deep Learning et Vision par Ordinateur

Ce projet utilise des techniques avancées de **vision par ordinateur**, de **machine learning** et de **deep learning** pour analyser automatiquement des vidéos de matchs de football. Il permet de détecter, suivre, et analyser les mouvements des joueurs, du ballon et des arbitres à partir d’une simple vidéo.

## 🎯 Objectifs du projet

- Détecter les **joueurs**, **arbitres** et **ballons** grâce à un détecteur d’objets (YOLO).
- Suivre les objets détectés à travers les images grâce à des **trackers**.
- Identifier les équipes des joueurs à l’aide du **clustering K-Means** basé sur les couleurs des maillots.
- Détecter les mouvements de la caméra avec le **flux optique** (optical flow).
- Corriger la perspective de la scène pour mesurer les distances en **mètres** et non en pixels.
- Calculer :
  - La **distance parcourue** par chaque joueur
  - La **vitesse** moyenne (en m/s)

## 🧠 Technologies utilisées

| Catégorie              | Outils / Modèles utilisés                          |
|------------------------|----------------------------------------------------|
| Détection d’objets     | YOLO (You Only Look Once)                          |
| Suivi d’objets         | Trackers OpenCV (KCF, CSRT, etc.)                 |
| Apprentissage non supervisé | K-Means pour segmentation par couleur           |
| Estimation du mouvement| Flux optique (Lucas-Kanade)                        |
| Transformation géométrique | Homographie pour correction de perspective     |
| Langage                | Python                                             |
| Bibliothèques          | OpenCV, NumPy, scikit-learn, Matplotlib, etc.     |






