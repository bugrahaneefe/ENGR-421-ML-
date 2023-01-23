#!/usr/bin/env python
# coding: utf-8

# In[131]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))


# In[132]:


training_set = np.genfromtxt("hw05_data_set_train.csv", delimiter = ",")
test_set = np.genfromtxt("hw05_data_set_test.csv", delimiter = ",")

x_train = training_set[:,0]
x_test = test_set[:,0]
y_train = training_set[:,1]
y_test = test_set[:,1]

K = np.max(y_train)
N = training_set.shape[0]
D = training_set.shape[1]

N_train = len(y_train)
N_test = len(y_test)


# In[139]:


def predict(x_test,test_y_prediction,y_prediction,node_splits,is_terminal):
    for i in range(x_test.shape[0]):
        index = 1
        while True:
            if is_terminal[index] == True:
                test_y_prediction.append(y_prediction[index])
                break
            else:
                if x_test[i] > node_splits[index]:
                    index = 2*index
                else: 
                    index = 2*index + 1
    return np.array(test_y_prediction)
def decision_tree(p):
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}

    y_prediction = {}
    
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True
    while True:
        # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items() if value == True]
        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
                
            if len(y_train[data_indices]) <= p:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False
    
                
                best_scores = np.repeat(0.0, D)
                best_splits = np.repeat(0.0, D)
                for d in range(D):
                    unique_values = np.sort(np.unique(x_train[data_indices]))
                    split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                    split_scores = np.repeat(0.0, len(split_positions))
                    for s in range(len(split_positions)):
                        left_indices = data_indices[x_train[data_indices] > split_positions[s]]
                        right_indices = data_indices[x_train[data_indices] <= split_positions[s]]
                        split_scores[s] = (1 / len(data_indices) * (np.sum((y_train[left_indices] - np.mean(y_train[left_indices]))** 2))) + (1 / len(data_indices) * np.sum(( y_train[right_indices] - np.mean(y_train[right_indices])) ** 2))
                    best_scores[d] = np.min(split_scores)
                    best_splits[d] = split_positions[np.argmin(split_scores)]
                
                # decide where to split on which feature
                split_d = np.argmin(best_scores)
                node_features[split_node] = split_d
                node_splits[split_node] = best_splits[split_d]
                
                # create left node using the selected split
                left_indices = data_indices[x_train[data_indices] > best_splits[split_d]]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True
          
                # create right node using the selected split
                right_indices = data_indices[x_train[data_indices] <= best_splits[split_d]]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
    
    for i in node_indices.keys():
        y_prediction[i] = np.mean(y_train[node_indices[i]])
    
    train_y_prediction = []
    train_y_prediction = predict(x_train,train_y_prediction,y_prediction,node_splits,is_terminal)
    
    
    test_y_prediction = []
    test_y_prediction = predict(x_test,test_y_prediction,y_prediction,node_splits,is_terminal)
    
                   
    train_rmse = np.sqrt((np.sum((y_train - train_y_prediction)**2)) / y_train.shape[0])
    test_rmse = np.sqrt((np.sum((y_test - test_y_prediction)**2)) / y_test.shape[0])

    return is_terminal, node_splits, y_prediction, train_rmse, test_rmse


# In[140]:


P = 30
is_terminal, node_splits, y_prediction, train_rmse, test_rmse = decision_tree(P)

data_interval = np.linspace(0,2,1001)
y_hat = []
for i in range(data_interval.shape[0]):
    index = 1
    while True:
        if is_terminal[index] == True:
            y_hat.append(y_prediction[index])
            break
        else:
            if data_interval[i] > node_splits[index]:
                index = 2*index
            else: 
                index = 2*index + 1
                
y_hat = np.array(y_hat)

#training
plt.figure(figsize=(10,6))
plt.plot(x_train[:], y_train[:], "b.", label = "training")
plt.plot(data_interval, y_hat, "k")   
plt.ylim([-1, 2])
plt.legend()

    
plt.xlabel("Time (sec)")
plt.ylabel("Signal (milivolt)")
plt.show()

#test
plt.figure(figsize=(10,6))
plt.plot(x_test[:], y_test[:], "r.", label = "test")
plt.plot(data_interval, y_hat, "k")   
plt.ylim([-1, 2])
plt.legend()


plt.xlabel("Time (sec)")
plt.ylabel("Signal (milivolt)")
plt.show()



print("RMSE on training set is", train_rmse ," when P is 30")
print("RMSE on test set is",test_rmse ," when P is 30")


# In[141]:


x = np.arange(10,55,5)
#print(x)
train = []
test=[]
for i in x:
    is_terminal, node_splits, y_prediction, train_rmse, test_rmse = decision_tree(i)
    train.append(train_rmse)
    test.append(test_rmse)
#print(train_rmse)
#print(train)
plt.figure(figsize=(10,6))
plt.plot(x, train[:],'.b-', markersize = 10, label = "training")
plt.plot(x, test[:], '.r-', markersize = 10, label = "test")
plt.legend()

plt.legend(loc='upper left')

plt.xlabel("Pre-prunning size -P-")
plt.ylabel("RMSE")
plt.show()


# In[ ]:




