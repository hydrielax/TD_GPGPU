# Compte-rendu TP GPGPU : Réseau de neurones

*Paul TRINCKLIN et Alexis DELAGE*

Code du projet : [github.com/hydrielax/TD_GPGPU](https://github.com/hydrielax/TD_GPGPU/tree/master/TP1)

## 1. Prise en main

### a) Analyse du code séquentiel : entraînement d'un réseau de neurones dense

Le code séquentiel se décompose en plusieurs fichiers :
- `mnist.c` : Lecture des données du dataset MNIST (images et labels)
- `matrix.c` : Fonctions de calcul matriciel (addition, produit, transposition) et de gestion de l'espace mémoire pour les matrices
- `ann.c` : définition de la structure du réseau de neurone, et implémentation de la méthode de descente de gradient pour l'entraînement d'un réseau dense (propagation avant et arrière).
- `main.c` : corps principal du programme, qui se compose des étapes suivantes :
    - Lecture des données.
    - Initialisation des poids du réseau.
    - Entraînement itératif du réseau par une succession d'epochs: chaque epoch est lui-même décomposé en entraînement successif par minibatchs de données de taille fixe (16), pour lesquels on envoie le données sur la première couche, on applique la propagation avant (calcul des activations) et enfin la rétro-propagation du gradient (mise à jour des poids du réseau).
    - Calcul du score par propagation avant (précision) et du temps d'exécution.
### b) Méthodologie d'approche du problème

Pour commencer, nous avons fait tourner le code C++ et documenté toutes les fonctions pour comprendre le fonctionnement du code.

Nous avons ensuite analysé les temps d'exécution de chaque partie du code afin d'identifier les parties du code les plus coûteuses. 

Ces fonctions gourmandes sont `fordward` et `backward`, principalement à cause du produit matriciel. 

## 2. Premiers test

### a) Utilisation de `cudaMallocManaged`

* On passe d'abord le réseau de neurones (variable `nn` et attributs) dans une mémoire partagée (avec `cudaMallocManaged`) afin d'y accéder à la fois depuis le CPU et le GPU. On pourra modifier cela plus tard, cela permet juste d'avoir une première idée facilement avant d'optimiser la gestion de la mémoire.
* Nos premiers tests rencontrent quelques problèmes, dûs à des soucis de gestion entre variables et pointeurs. Nous avons pu les résoudre (sans pour autant comprendre la cause de l'erreur), en créant notamment un fichier de test pour tester des fonctions de manière atomique.

### b) Parallélisation du calcul du produit matriciel

* Nous avions observé précédemment que l'opération de produit matriciel était la plus coûteuse. Afin d'apporter une première optimisation, on décide donc de paralléliser cette opération, en nous appuyant sur le travail fait au TD3.
* Notre premier essai fût infructueux ! En effet la version du TD3 utilisé n'était pas fonctionnelle (peut-être)

## 3. Implémentation choisie pour la parallélisation

### a) Gestion de la mémoire

Pour la gestion de la mémoire, nous avons choisi de répartir les différentes variables sur les mémoires CPU et GPU de la manière suivante pour les objets `matrix_t` :
```js
matrix_t:struct [CPU] {
    m:*double [GPU]
    rows:uint [CPU]
    columns:uint [CPU]
}
```
Ensuite, pour les structures de `layers` et de réseaux de neurones (`nn`), nous avons choisi d'allouer à chaque fois la mémoire sur le CPU sauf pour les matrices, qui respectent le schéma précédent.

Afin de nous retrouver dans notre structure, nous avons choisi de renommer tous les objects de type `matrix_t` avec l'attribut `m` sur le GPU en suivant la convention de nommage `g_<nom>`, afin de les différencier des variables complètement sur le GPU en `d_<nom>`. Les autres matrices complètement mémorisées sur le CPU (utilisées uniquement à des fins d'initialisation de valeur) ne suivent pas de convention particulière.

### b) Parallélisation de fonctions

Chacune des fonctions de calcul matriciel a transformée en kernel, même si l'on a vu qu'en pratique c'est le produit matriciel qui prenait le plus longtemps à se faire (complexité en $m\times n\times p$ alors que les autres fonctions ont une complexité en $m\times n$). On a donc parallélisé la somme, la différence, le produit de Hadamard, transposition, l'application d'une fonction terme à terme et la multiplication par un scalaire sur les matrices. 

### c) Problèmes et bugs rencontrés

La gestion des bugs rencontrés est ce qui nous a pris le plus de temps sur le projet. La plupart ne sont que des erreurs triviales (inversion de paramètres entre une variable CPU et une variable GPU, échange entre le nombre de colonnes et de lignes, problèmes de dimension, typage) mais elles n'ont pas toujours été faciles à détecter, car on pouvait obtenir des segmentation fault et rechercher à la main l'endroit de l'erreur, ou alors le code s'exécutait entièrement sans que le calcul soit réalisé.  

Nous avons aussi rencontré des problèmes lors de l'exécution lorsqu'il y avait des erreurs de segmentation (où du moins c'est ce que l'on suppose) au niveau du GPU : dans ces-là, aucune erreur n'était renvoyée et le programme continuait de tourner en arrière-plan sans jamais pouvoir être stoppé (nous avons dû faire un arrêt forcé de la machine pour continuer).

## 4. Evaluation des performances

La parallélisation de toutes les fonctions de calcul matriciel a très fortement accéléré la durée d'exécution d'un epoch. Voici les temps d'exécution obtenus pour les deux versions de notre code:

| | Code séquentiel | Code parallélisé |
| -- | -- | -- |
| Temps total | 8m 5,747 s | 54,588 s |
| Temps moyen par epoch | 12,14 s | 1,36 s |
| Précision initiale | 8,9 % | 12,14 % |
| Précision après la 1ere epoch | 74,61 % | 73,82 % |
| Précision après la 40e epoch | 94,53 % | 94,33 % |

On observe bien une accélération d'un facteur 9 entre les deux versions, ce qui est attendu pour ce genre de problème.

La précision dans les 2 cas est similaire, ce qui est logique puisque les calculs réalisés sont les mêmes. 