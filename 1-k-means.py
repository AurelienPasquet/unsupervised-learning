import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

from tqdm.auto import tqdm

##################################################################
# Exemple :  k-Means Clustering

path = '../artificial/'
name = "banana.arff"

# Load raw data
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

# Plotting initial data
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
plt.show()

model_metrics = {
    mode: {
        "inertie": [],
        "silhouette": [],
        "runtime": []
    }
    for mode in ["normal", "mini-batch"]
}

n_centers = 20
n_start = 2 # Silhouette score works with 2+ centers

# Training and collecting data from models with i clusters
for i in tqdm(range(2, n_centers+1)): 
    
    # modes: normal | mini-batch
    for mode in model_metrics:
        
        start_time = time.time()
        
        if mode == "normal":
            model = cluster.KMeans(n_clusters=i, init='k-means++', n_init=1)
        else:
            model = cluster.MiniBatchKMeans(n_clusters=i, init='k-means++', n_init=1)
    
        model.fit(datanp)
        
        model_metrics[mode]["runtime"].append(time.time() - start_time)
        model_metrics[mode]["inertie"].append(model.inertia_)
        model_metrics[mode]["silhouette"].append(metrics.silhouette_score(datanp, model.labels_))   

# Plotting model metrics
for metric in model_metrics["normal"]:
    for mode in model_metrics:
        plt.plot(range(n_start, n_centers+1), model_metrics[mode][metric], label=mode)
    plt.title(f"{metric} en fonction du nombre de centres")
    plt.xlabel("centres")
    plt.ylabel(metric)
    plt.xticks(range(n_start, n_centers+1))
    plt.legend()
    plt.show()

# Get optimal number of clusters and optimal mode
if max(model_metrics["normal"]["silhouette"]) > max(model_metrics["mini-batch"]["silhouette"]):
    best_n_clusters = np.argmax(model_metrics["normal"]["silhouette"])
    best_mode = "normal"
else:
    best_n_clusters = np.argmax(model_metrics["mini-batch"]["silhouette"])
    best_mode = "mini-batch"
best_n_clusters += 2 # index 0 means 2 clusters, adding 2 to compensate the offset

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
start_time = time.time()

if best_mode == "normal":
    model = cluster.KMeans(n_clusters=best_n_clusters, init='k-means++', n_init=1)
else:
    model = cluster.MiniBatchKMeans(n_clusters=best_n_clusters, init='k-means++', n_init=1)

model.fit(datanp)
end_time = time.time()
labels = model.labels_

# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(best_n_clusters))
plt.show()

print("nb clusters =",best_n_clusters,"mode =",best_mode,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((end_time - start_time)*1000,2),"ms")

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
print(dists)

