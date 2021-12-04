import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations
import random

#===============================================================================================================
#===============================================================================================================

def KMeans(data, k, distance_measure, threshold, iterations):
    tolerance = threshold
    rows = np.shape(data)[0]

    init_ind = np.random.choice(rows, k, replace = False)
    initial_centroids = data[init_ind,:]
    
    #distances: 1.euclidean, 2.cityblock (manhattan), 3.mahalanobis, 4.chebyshev
    selected_distance = distance_measure
    print("Distance metric used: {}".format(selected_distance))
    if selected_distance not in ["euclidean", "cityblock", "mahalanobis", "chebyshev"]:
        raise Exception("Invalid distance measure. Distance should be between euclidean, cityblock, mahalanobis and chebyshev")

    dist = cdist(data, initial_centroids, selected_distance)
    
    clusters_ids = np.argmin(dist, axis = 1)
    clusters = [[] for i in range(k)]
        
    for j, cluster_id in enumerate(clusters_ids):
        clusters[cluster_id].append(data[j])
    
    new_centroids = [[]]*k
    old_centroids = initial_centroids
    c = 0
    conv = []
    for i in range(iterations):
        c = i
        print("Round {}:".format(c))
        for group in range(k):
            new_centroids[group] = np.mean([idx for idx in clusters[group]],axis=0)
        new_centroids = np.vstack(new_centroids)
        
        dist = cdist(data, new_centroids, selected_distance)
        
        clusters_ids = np.argmin(dist, axis = 1)
        clusters = [[] for i in range(k)]
        
        for j, cluster_id in enumerate(clusters_ids):
            clusters[cluster_id].append(data[j])
        for v in range(k):
            clusters[v] = np.vstack(clusters[v])
        
        convergence_measure = (np.linalg.norm(np.array(new_centroids) - np.array(old_centroids)))/np.linalg.norm(old_centroids)
        conv.append(convergence_measure)

        silhouette_coefficients = []
        for sc in range(k):
            intra_matrix = cdist(clusters[sc], clusters[sc], "euclidean")
            columns = np.shape(intra_matrix)[1]
            a = np.sum(intra_matrix, axis=1)/(columns-1)
            inter_matrix = cdist(clusters[sc], new_centroids[np.arange(len(new_centroids)) !=sc, :], "euclidean")
            b = np.mean(inter_matrix, axis=1)
            s = np.mean((b-a)/np.maximum(a,b))
            silhouette_coefficients.append(s)

        score = np.mean(silhouette_coefficients)

        if  convergence_measure < tolerance:
            break
        old_centroids = new_centroids
        new_centroids = [[]]*k
    
    return new_centroids, clusters_ids, score, np.std(silhouette_coefficients), conv
 
#===============================================================================================================
#===============================================================================================================
