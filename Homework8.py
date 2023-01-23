#!/usr/bin/env python
# coding: utf-8

# In[341]:


import matplotlib.pyplot as plt
import numpy as np
import statistics
import scipy.linalg as linalg
import math
from scipy.stats import norm


# In[342]:


data_set = np.genfromtxt("hw08_data_set.csv", delimiter = ",")
initial_centroids = np.genfromtxt("hw08_initial_centroids.csv", delimiter = ",")

K = 9
N = 1000
samples = [100,100,100,100,100,100,100,100,200]

def norm_dist(X, mean, covariance):
    return np.exp((-0.5)*np.matmul((X - mean).reshape(X.shape[0],1,2), np.matmul(linalg.cho_solve(linalg.cho_factor(covariance), np.eye(2)),(X - mean).reshape(X.shape[0],2,1))).reshape(X.shape[0],)) / (2*np.pi*np.sqrt(linalg.det(covariance))) 
def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = X[np.random.choice(range(N), K, False),:]
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)
def nearest_centroid(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)


# In[348]:


clusters = np.argmin(np.sum(initial_centroids*initial_centroids, axis = 1) - 2*(np.matmul(data_set, initial_centroids.T)), axis = 1)
P_cluster = np.histogram(clusters, np.arange(0, K + 1), density = True)[0]

means = np.stack([np.mean(data_set[clusters == k], axis = 0)  for k in range(K)])
covariances = np.stack([np.cov(data_set[clusters == k], rowvar = False) for k in range(K)])

for i in range(100):
    H = np.stack([norm_dist(data_set, means[k], covariances[k])*P_cluster[k]  for k in range(K)], axis = 1)
    H = 1/np.sum(H, axis = 1, keepdims = True) * H
    P_cluster = np.sum(H, axis = 0) / N
    means = np.stack([np.sum(H[:,[k]]*data_set, axis = 0) / np.sum(H[:,k]) for k in range(K)])
    covariances = np.stack([np.sum(H[:,[[k]]]*np.matmul((data_set - means[k]).reshape(N,2,1),(data_set - means[k]).reshape(N,1,2)), axis = 0) / np.sum(H[:,k]) for k in range(K)])

memberships = clusters
memberships = np.argmin(np.sum(means*means, axis = 1) - 2*(np.matmul(data_set, means.T)), axis = 1)

print("Means:")
print(means)


# In[344]:


original_class_means = np.array([[5,5],[-5,5],[-5,-5],[5,-5],[5,0],[0,5],[-5,0],[0,-5],[0,0]])


class_deviations = np.array([[[0.8,-0.6],[-0.6,0.8]],[[0.8,0.6],[0.6,0.8]],[[0.8,-0.6],[-0.6,0.8]],[[0.8,0.6],[0.6,0.8]],
                             [[0.2,0],[0,1.2]],[[1.2,0],[0,0.2]],[[0.2,0],[0,1.2]],[[1.2,0],[0,0.2]],[[1.6,0],[0,1.6]]])

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    plt.figure(figsize=(10,10))

    for c in range(K):
        main_plot = plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
        a,b = np.meshgrid(np.linspace(-8,8,201), np.linspace(-8,8,201))
        plt.contour(a, b, norm_dist(np.concatenate((a.flatten()[:,None], b.flatten()[:,None]), axis = 1), means[c], covariances[c]).reshape(201,201), levels = [0.05],colors = main_plot[0].get_color())
        plt.contour(a, b, norm_dist(np.concatenate((a.flatten()[:,None], b.flatten()[:,None]), axis = 1), original_class_means[c], class_deviations[c]).reshape(201,201), levels = [0.05], colors = "black", linestyles = "dashed")    
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.ylim([-8, 8])
    plt.xlim([-8, 8])

plot_current_state(means,memberships,data_set)


# In[ ]:




