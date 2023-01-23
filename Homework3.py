#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


# In[12]:


data = np.genfromtxt("hw03_data_set_images.csv", delimiter = ",")
label = np.genfromtxt("hw03_data_set_labels.csv")

training_data= np.zeros((25*5,320))
training_labels = np.zeros(25*5)
test_data = np.zeros((14*5,320))
test_labels = np.zeros(14*5)

for i in range(5):
    training_data[25*i:(25+(25*i)),:] = data[(39*i):((i*39)+25),:]
    training_labels[25*i:(25+(25*i))] = label[(39*i):((i*39)+25)]
    test_data[(14*i):((14*i)+14),:] = data[((39*i)+25):((i*39+39)),:]
    test_labels[(14*i):((14*i)+14)] = label[((39*i)+25):((i*39+39))]

training_labels = training_labels.astype(int)
training_data = training_data.astype(int)


# In[13]:


W = np.random.uniform(low = -0.01, high = 0.01, size = (320, 5))
wo = np.random.uniform(low = -0.01, high = 0.01, size = (1, 5))
eta = 0.001
epsilon = 0.001

#print(W)
#print(wo)


# In[14]:


def sigmoid(x, W, wo):
    return (1 / ( 1 + np.exp(-(np.matmul(x,W) + wo))))


# In[15]:


iteration = 1
objectives = []

trainlabels = np.stack([1*(training_labels==(i+1)) for i in range(np.max(training_labels))], axis = 1)
#print(trainlabels)
#training_data= training_data.reshape(125,320)

while 1:
    Y_P = sigmoid(training_data, W, wo)
    objectives = np.append(objectives, (0.5)*np.sum(np.sum((trainlabels-Y_P)**2,axis=1),axis=0))
    
    W_old = W
    wo_old = wo

    W = W - np.asarray([-np.sum(np.repeat((trainlabels[:,c] - Y_P[:,c])[:, None], training_data.shape[1], axis = 1) * training_data, axis = 0) for c in range(5)]).transpose()*eta
    wo = wo - (-np.sum(trainlabels - Y_P, axis = 0)*eta)

    if np.sqrt(np.sum((wo - wo_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break
    iteration = iteration + 1

#print(W)
#print(wo)


# In[16]:


plt.figure(figsize = (8, 4))
plt.plot(range(1, iteration + 1), objectives, "k")
plt.ylabel("Error")
plt.xlabel("Iteration")
plt.show()


# In[17]:


y_prediction = []
y_prediction = np.argmax(sigmoid(training_data,W,wo),axis=1) +1
confusion = pd.crosstab(y_prediction,training_labels,rownames = ["y_pred"],colnames=["y_truth"])
print(confusion)


# In[18]:


y_prediction = []
y_prediction = np.argmax(sigmoid(test_data,W,wo),axis=1) +1
confusion = pd.crosstab(y_prediction,test_labels,rownames = ["y_pred"],colnames=["y_truth"])
print(confusion)


# In[ ]:




