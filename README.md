# Utilisation

## Kmeans

Choisir le dataset et le nombre maximum de clusters à calculer.

```python
# Exemple
name = "cure-t0-2000n-2D.arff"
n_centers = 20
```

Le programme affiche la courbe des différentes metrics en fonction du nombre de clusters ainsi la solution optimale de clustering sur le dataset (nombre de clusters et méthode optimale (normale ou mini-batch))

## Agglomerative clustering

Choisir le dataset et le nombre maximum de clusters à calculer.

```python
# Exemple
name = "2d-4c-no4.arff"
n_centers = 30
```

Comme précédemment, le programme affiche la courbe des différentes metrics en fonction du nombre de clusters ainsi la solution optimale de clustering sur le dataset (nombre de clusters et linkage optimal). Le dendrogramme correspondant est aussi affiché.

## DBSCAN

Choisir le dataset et les différentes valeurs d'epsilon à tester.

```python
# Exemple
name="banana.arff"
epsilon = [i/20 for i in range(1, 20)] # De 0.05 à 0.95
```

Le programme affiche la courbe des différentes metrics en fonction de la valeur d'épsilon (pour le score de silhouette, un score de -1 signifie que la valeur d'epsilon ne permet pas d'avoir au moins 2 clusters). La solution optimale de clustering est affichée avec epsilon optimal.