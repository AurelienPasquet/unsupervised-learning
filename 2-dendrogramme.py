import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram

from tqdm.auto import tqdm

##################################################################
# Exemple :  Dendrogramme and Agglomerative Clustering

path = './artificial/'
name="2d-4c-no4.arff"
n_clusters = 30

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

#################################################

from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix)#, **kwargs)

linkages = ["ward", "complete", "average", "single"]
models_metrics = {
    linkage: {
        "models": [],
        "metrics": {
            "silhouette": [],
            "runtime": []
        }
    }
    for linkage in linkages
}

for i in tqdm(range(2, n_clusters+1)): # Silhouette score works with 2+ centers
    for linkage in models_metrics:
        start_time = time.time()
        model = cluster.AgglomerativeClustering(distance_threshold=None, linkage=linkage, n_clusters=i)
        model.fit(datanp)
        end_time = time.time()
        models_metrics[linkage]["models"].append(model)
        models_metrics[linkage]["metrics"]["silhouette"].append(metrics.silhouette_score(datanp, model.labels_))
        models_metrics[linkage]["metrics"]["runtime"].append(end_time - start_time)

# Plot silhouette scores and runtimes for each linkage
centers = [i for i in range(2, n_clusters+1)]

rows = 1
cols = len(models_metrics["ward"]["metrics"])

plt.figure(figsize=(12, 6))
for i, metric in enumerate(models_metrics["ward"]["metrics"].keys()):
    plt.subplot(rows, cols, i + 1)
    for j, linkage in enumerate(models_metrics.keys()):
        plt.plot(centers, models_metrics[linkage]["metrics"][metric], marker='o', linestyle='-', label=linkage)
    plt.title(metric)
    plt.xlabel("centers")
    plt.ylabel(metric)
    plt.legend()
plt.tight_layout()
plt.show()
    
# plt.figure(figsize=(10, 8))
# for i, linkage in enumerate(models_metrics.keys()):
#     for j, metric in enumerate(models_metrics[linkage]["metrics"].keys()):
#         plt.subplot(rows, cols, i*cols + j + 1)
#         plt.title(f"{metric} ({linkage})")
#         plt.plot(centers, models_metrics[linkage]["metrics"][metric], marker='o', linestyle="-")
#         plt.xlabel("centers")
#         plt.ylabel(metric)
# plt.tight_layout()
# plt.show()

# Get best linkage and best model
best_linkage = max(models_metrics, key=lambda linkage: models_metrics[linkage]["metrics"]["silhouette"])
best_silhouette_index = np.argmax(models_metrics[best_linkage]["metrics"]["silhouette"])
best_model = models_metrics[best_linkage]["models"][best_silhouette_index]

# Get the best number of clusters regarding silhouette score 
nb_clusters = best_silhouette_index + 2 # index 0 means 2 clusters, adding 2 to compensate the offset
labels = best_model.labels_

# Plot data with clusters
plt.scatter(f0, f1, c=labels, s=8)
plt.title(str(name) + " - Nb clusters ="+ str(nb_clusters) + " - Linkage : " + best_linkage)
plt.show()

# setting distance_threshold=0 ensures we compute the full tree.
model = cluster.AgglomerativeClustering(distance_threshold=0, linkage=best_linkage, n_clusters=None)

model = model.fit(datanp)
plt.figure(figsize=(10, 7))
plt.title("Hierarchical Clustering Dendrogram")

# plot the top p levels of the dendrogram
plot_dendrogram(model)#, truncate_mode="level", p=6)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
