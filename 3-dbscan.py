import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from tqdm.auto import tqdm

##################################################################
# Exemple : DBSCAN Clustering


path = './artificial/'
name="banana.arff"
# Measurements with epsilon from 0.05 to 0.95
epsilon = [i/20 for i in range(1, 20)]
print(type(epsilon))

# Load raw data
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
# 
print("------------------------------------------------------")
print("Appel DBSCAN (1) ... ")
tps1 = time.time()
epsilon_=2 #2  # 4
min_pts= 5 #10   # 10
model = cluster.DBSCAN(eps=epsilon_, min_samples=min_pts)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)


####################################################
# Standardisation des donnees

scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()

models_metrics = {
    "models": [],
    "metrics": {
        "silhouette": [],
        "runtime": []
    }
}

for e in tqdm(epsilon):
    start_time = time.time()
    model = cluster.DBSCAN(eps=e, min_samples=min_pts)
    model.fit(data_scaled)
    end_time = time.time()
    models_metrics["models"].append(model)
    
    # Remove outliers for a more accurate silhouette score
    non_outliers_mask = model.labels_ != -1
    data_no_outliers = data_scaled[non_outliers_mask]
    labels_no_outliers = model.labels_[non_outliers_mask]
    
    # Only calculate silhouette score if there are at least two clusters
    if len(set(labels_no_outliers)) > 1:
        silhouette_score = metrics.silhouette_score(data_no_outliers, labels_no_outliers)
    else:
        silhouette_score = -1 # Default value when nb clusters < 2
        
    models_metrics["metrics"]["silhouette"].append(silhouette_score)
    models_metrics["metrics"]["runtime"].append(end_time - start_time)

# Plot metrics
rows = 1
cols = 2
plt.figure(figsize=(12, 6))
for i, metric in enumerate(models_metrics["metrics"].keys()):
    plt.subplot(rows, cols, i+1)
    plt.title(metric)
    plt.plot(epsilon, models_metrics["metrics"][metric], marker='o', linestyle='-')
    plt.xlabel("epsilon")
    plt.ylabel(metric)
plt.show()

# Get best model and best epsilon
best_model = models_metrics["models"][np.argmax(models_metrics["metrics"]["silhouette"])]
best_epsilon = epsilon[np.argmax(models_metrics["metrics"]["silhouette"])]

# Plot data with clusters and outliers
plt.scatter(f0_scaled, f1_scaled, c=best_model.labels_, s=8)
plt.title("Données après clustering DBSCAN (2) - Epislon= "+str(best_epsilon)+" MinPts= "+str(min_pts))
plt.show()
