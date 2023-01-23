#!/usr/bin/env python
# coding: utf-8

# In[195]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats


# In[196]:


train = np.genfromtxt("hw04_data_set_train(1).csv", delimiter = ",")
test = np.genfromtxt("hw04_data_set_test(1).csv", delimiter = ",")

train_x = np.array(train[:,0])
train_y = np.array(train[:,1])
test_x = np.array(test[:,0])
test_y = np.array(test[:,1])

data_interval = np.linspace(np.around(np.min(train_x)), np.around(np.max(train_x)), 2001)
K = np.max(train_y)
N = train.shape[0]

bin_width = 0.1
origin = 0.0

left_borders = np.arange(np.min(train_x), np.max(train_x), bin_width)
right_borders = np.arange(np.min(train_x) + bin_width, np.max(train_x) + bin_width, bin_width)

p_hat = np.asarray([np.average(train_y[((left_borders[b] < train_x) & (train_x <= right_borders[b]))]) for b in range(len(left_borders))]) 

plt.figure(figsize = (10, 5))
plt.plot(train_x,train_y,"b.", markersize = 10,label="Training")
plt.plot(data_interval[1:], np.repeat(p_hat,100), "k")   

plt.legend(loc='upper right')
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.ylim([-1, 2])
plt.show()

plt.figure(figsize = (10, 5))
plt.plot(test_x,test_y,"r.", markersize = 10,label="Test")
plt.plot(data_interval[1:], np.repeat(p_hat,100), "k")   

plt.legend(loc='upper right')
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.ylim([-1, 2])
plt.show()



# In[197]:


test_y_head = np.zeros((test_x.shape[0],))
for i in range(test_x.shape[0]):
    for b in range(len(left_borders)):
        if(((left_borders[b] < test_x[i]) & (test_x[i] <= right_borders[b]))):
            test_y_head[i]= test_y_head[b]

RMSE = np.sqrt(np.sum((test_y - test_y_head)**2)/ test_y.shape[0])
print("Regressogram => RMSE is", RMSE, "when h is", bin_width)


# In[198]:


#p_hat = [np.average(train_y*(((x - 0.5 * bin_width) < train_x) & (train_x <= (x + 0.5 * bin_width)))) for x in data_interval]
p_hat = np.asarray([np.sum(train_y*(((x - 0.5 * bin_width) < train_x) & (train_x <= (x + 0.5 * bin_width)))) for x in data_interval]) / (N*bin_width)

plt.figure(figsize = (10, 5))
plt.plot(train_x,train_y,"b.", markersize = 10,label="Training")
plt.plot(data_interval, p_hat, "k-")   

plt.legend(loc='upper right')
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.ylim([-1, 2])
plt.show()
plt.figure(figsize = (10, 5))
plt.plot(test_x,test_y,"r.", markersize = 10,label="Test")
plt.plot(data_interval, p_hat, "k-")   

plt.legend(loc='upper right')
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.ylim([-1, 2])
plt.show()


# In[199]:


y_test_head = np.zeros((test_x.shape[0],))
y_test_head = np.asarray([np.average(train_y[((i- 0.5 * bin_width) < train_x) & (train_x <= (i + 0.5 * bin_width))]) for i in test_x])

RMSE = np.sqrt(np.sum((test_y - y_test_head)**2)/ test_y.shape[0])
print("Running Mean Smoother => RMSE is", RMSE,  "when h is", bin_width)


# In[200]:


bin_width = 0.02
p_hat = np.asarray([np.sum((train_y)*1.0/ np.sqrt(2*math.pi) *                          np.exp(-0.5 * (x-train_x)**2 / bin_width**2)) 
                    for x in data_interval]) / (N*bin_width)
plt.figure(figsize = (10, 5))
plt.plot(train_x,train_y,"b.", markersize = 10,label="Training")
plt.plot(data_interval, p_hat, "k-")   

plt.legend(loc='upper right')
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.ylim([-1, 2])
plt.show()
plt.figure(figsize = (10, 5))
plt.plot(test_x,test_y,"r.", markersize = 10,label="Test")
plt.plot(data_interval, p_hat, "k-")   

plt.legend(loc='upper right')
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.ylim([-1, 2])
plt.show()


# In[201]:


y_test_head = np.zeros((test_x.shape[0],))
y_test_head = np.asarray([(np.sum((train_y) * (1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - train_x)**2 / bin_width**2)))) 
                                 / np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - train_x)**2 / bin_width**2)) for x in test_x])

RMSE = np.sqrt(np.sum((test_y - y_test_head)**2)/ test_y.shape[0])
print("Kernel Smoother => RMSE is", RMSE, "when h is", bin_width)


# In[ ]:





# In[ ]:




