# Compte-rendu TP1 GPGPU

Paul TRINCKLIN et Alexis DELAGE

## 1. Prise en main

On a documenté toutes les fonctions pour comprendre le fonctionnement du code.

Analysé le temps d'exécution afin d'identifier les parties du code les plus coûteuses. => fonctions `fordward` et `backward`, principalement dû au produit matriciel.

## 2. Premiers test

* On passe le réseau de neurones (variable `nn` et attributs) dans une mémoire partagée (avec `cudaMallocManaged`) afin d'y accéder à la fois depuis le CPU et le GPU.
* On exécute le produit matriciel sur GPU.
