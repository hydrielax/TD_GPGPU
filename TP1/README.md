# Compte-rendu TP1 GPGPU

Paul TRINCKLIN et Alexis DELAGE

## 1. Prise en main

On a documenté toutes les fonctions pour comprendre le fonctionnement du code.

Analysé le temps d'exécution afin d'identifier les parties du code les plus coûteuses. => fonctions `fordward` et `backward`, principalement dû au produit matriciel.

## 2. Premiers test

### Utilisation de `cudaMallocManaged`

* On passe d'abord le réseau de neurones (variable `nn` et attributs) dans une mémoire partagée (avec `cudaMallocManaged`) afin d'y accéder à la fois depuis le CPU et le GPU. On pourra modifier cela plus tard, cela permet juste d'avoir une première idée facilement avant d'optimiser la gestion de la mémoire.
* Nos premiers tests rencontrent quelques problèmes, dûs à des soucis de gestion entre variables et pointeurs. Nous avons pu les résoudre (sans pour autant comprendre la cause de l'erreur), en créant notamment un fichier de test pour tester des fonctions de manière atomique.

### Parallélisation du calcul du produit matriciel

* Nous avions observé précédemment que l'opération de produit matriciel était la plus coûteuse. Afin d'apporter une première optimisation, on décide donc de paralléliser cette opération, en nous appuyant sur le travail fait au TD3.
* Notre premier essai fût infructueux ! En effet la version du TD3 utilisé n'était pas fonctionnelle (peut-être)