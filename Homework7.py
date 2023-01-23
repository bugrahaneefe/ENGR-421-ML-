#!/usr/bin/env python
# coding: utf-8

# In[8]:


#import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import scipy.linalg as la
#import scipy.spatial.distance as dt
from scipy.spatial import distance
from scipy import stats

data_set = np.genfromtxt("hw07_data_set_images.csv", delimiter = ",")
label_set = np.genfromtxt("hw07_data_set_labels.csv", delimiter = ",")

data_train = data_set[:2000]
label_train = label_set[:2000]
data_test = data_set[2000:4000]
label_test = label_set[2000:4000]

N = len(label_train)
D = data_train.shape[1]
K=10

class_means=[]
for c in range(K):
    class_means.append([np.mean(data_train[label_train == c+1],axis=0)])

scatter_within_matrix = np.zeros((D,D))
scatter_within= [(np.matmul(np.transpose(data_train[label_train == (c + 1)] - class_means[c]), (data_train[label_train == (c + 1)] - class_means[c]))) for c in range(K)]
scatter_within_matrix = scatter_within[0]+ scatter_within[1]+scatter_within[2]+scatter_within[3]+scatter_within[4]+scatter_within[5]+scatter_within[6]+scatter_within[7]+scatter_within[8]+scatter_within[9]

scatter_between_matrix = np.zeros((D,D))
for i in range(K):
    scatter_between_matrix = (scatter_between_matrix + ((data_train[label_train == i+1]).shape[0]) * np.matmul((((np.mean(data_train[label_train == i+1], axis = 0)) - (np.mean(class_means, axis = 0))).reshape(D,1)), np.transpose((((np.mean(data_train[label_train == i+1], axis = 0)) - (np.mean(class_means, axis = 0))).reshape(D,1)))) )

print(scatter_within_matrix[0:4,0:4])
print(scatter_between_matrix[0:4,0:4])


# In[9]:


values, vectors = np.linalg.eig(np.dot(np.linalg.inv(scatter_within_matrix), scatter_between_matrix))
values = values.real
vectors = vectors.real
print(values[0:9])


# In[11]:


two_vectors = vectors[:, 0:2]

Z_train = np.matmul(data_train - np.mean(data_train, axis = 0), two_vectors)

Z_test = np.matmul(data_test - np.mean(data_test, axis = 0), two_vectors)

point_colors = ["#0000FF", "#458B74", "#CD3333","#FF9912", "#66CD00", "#CDC8B1","#CAFF70", "#9A32CD", "#FF1493","#00BFFF","#0000FF", "#458B74", "#CD3333","#FF9912", "#66CD00", "#CDC8B1","#CAFF70", "#9A32CD", "#FF1493","#00BFFF"]

class_labels = ["tshirt/top","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]
same_list=[]
plt.figure(figsize=(6,6))
for i in range(N):
    plt.scatter(Z_train[label_train == i + 1, 0], Z_train[label_train == i + 1, 1], color=point_colors[np.int(label_train[i]-1)],label = class_labels[np.int(label_train[i]-1)] if point_colors[np.int(label_train[i]-1)] not in same_list else '',s=10,marker="o")
    
    if point_colors.count(point_colors[np.int(label_train[i]-1)])>1:
        same_list.append(point_colors[np.int(label_train[i]-1)])
plt.ylim([-6, 6])
plt.xlim([-6, 6])

plt.legend(loc="upper left",markerscale = 2)
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.show()


# In[12]:


same_list=[]
plt.figure(figsize=(6,6))
for i in range(len(label_test)):
    plt.scatter(Z_test[label_test == i + 1, 0],  Z_test[label_test == i + 1, 1], color=point_colors[np.int(label_test[i])-1],label = class_labels[np.int(label_test[i]-1)] if point_colors[np.int(label_test[i]-1)] not in same_list else '',s=10,marker ="o")
    if point_colors.count(point_colors[np.int(label_test[i]-1)])>1:
        same_list.append(point_colors[np.int(label_test[i]-1)])
plt.ylim([-6, 6])
plt.xlim([-6, 6])

plt.legend(loc="upper left",markerscale = 2)
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.show()


# In[17]:


nine_vectors = vectors[:,0:9]
Z_train = np.matmul(data_train - np.mean(data_train, axis = 0), nine_vectors)
Z_test = np.matmul(data_test - np.mean(data_test, axis = 0), nine_vectors)

knn=11
train_knn=[]
for i in range(len(Z_train[:,1])):
    distances = np.zeros(Z_train.shape[0])
    for j in range(len(Z_train[:,1])):
        distances[j]=distance.euclidean(Z_train[i,:],Z_train[j,:])
    sorteddist=np.argsort(distances)[:5]
    
    
    myclasses= []
    for i in sorteddist:
        myclasses.append(label_train[i])
    train_knn.append(stats.mode(myclasses)[0])
confusion_matrix_train = pd.crosstab( np.array(train_knn)[:,0], label_train, rownames = ['y_hat'], colnames = ['y_train'])
print(confusion_matrix_train)


# In[19]:


test_knn=[]
for i in range(len(Z_test[:,1])):
    distances = []
    for j in range(len(Z_test[:,1])):
        distances.append(distance.euclidean(Z_test[i,:],Z_test[j,:]))
    sorteddist=sorted(distances)

    myclasses= []
    for t in range(knn):
        ind= distances.index(sorteddist[t])
        myclasses.append(int(label_test[ind]))
    test_knn.append(stats.mode(myclasses)[0])
confusion_matrix_test = pd.crosstab( np.array(test_knn)[:,0], label_test, rownames = ['y_hat'], colnames = ['y_test'])
print(confusion_matrix_test)

