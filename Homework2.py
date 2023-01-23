import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg as linalg
a = np.random.multivariate_normal(np.array([0.0,4.5]),np.array([[3.2,0.0],[0.0,1.2]]),105)

b = np.random.multivariate_normal(np.array([-4.5,-1.0]),np.array([[1.2,0.8],[0.8,1.2]]), 145)

c = np.random.multivariate_normal(np.array([4.5,-1.0]),np.array([[1.2,-0.8],[-0.8,1.2]]), 135)

d = np.random.multivariate_normal(np.array([0.0,-4.0]),np.array([[1.2,0.0],[0.0,3.2]]), 115)

#print(a)
#print(b[:5,:])
mean_1= np.mean(a,axis=0)
mean_2= np.mean(b,axis=0)
mean_3= np.mean(c,axis=0)
mean_4= np.mean(d,axis=0)

covarience_1 = np.cov(np.stack(((a[i]) for i in range(a.shape[0])),axis=1))
covarience_2 = np.cov(np.stack(((b[i]) for i in range(b.shape[0])),axis=1))
covarience_3 = np.cov(np.stack(((c[i]) for i in range(c.shape[0])),axis=1))
covarience_4 = np.cov(np.stack(((d[i]) for i in range(d.shape[0])),axis=1))


sample_means = np.stack((mean_1,mean_2,mean_3,mean_4,))
sample_covariences = np.stack((covarience_1,covarience_2,covarience_3,covarience_4))
class_priors = np.stack([i.shape[0] for i in [a,b,c,d]]) / np.sum([a.shape[0],b.shape[0],c.shape[0],d.shape[0]])
#print(sample_means)
#print(sample_covariences)
#print(class_priors)

confusion =np.zeros((4,4), dtype=int)
yanlis = np.empty((0,2), float)
for data in a:
    values = np.argmax(np.stack([-np.log(2*math.pi) - 0.5*np.log(np.linalg.det(sample_covariences[c]))-0.5*np.matmul((data-sample_means[c]), np.matmul(linalg.cho_solve(linalg.cho_factor(sample_covariences[c]), np.eye(2)), np.transpose((data - sample_means[c])))) +np.log(class_priors[c]) for c in range(4)]), axis=0)
    #print(np.stack([data]))
    #print(data)
    if (values == 0):
        confusion[0,0] +=1
    elif (values !=0):
        confusion[values,0] = confusion[values,0]+1
        yanlis = np.append(yanlis,np.stack([data]),axis=0)
    
for data in b:
    values = np.argmax(np.stack([-np.log(2*math.pi) - 0.5*np.log(np.linalg.det(sample_covariences[c]))-0.5*np.matmul((data-sample_means[c]), np.matmul(linalg.cho_solve(linalg.cho_factor(sample_covariences[c]), np.eye(2)), np.transpose((data - sample_means[c])))) +np.log(class_priors[c]) for c in range(4)]), axis=0)
    if (values == 1):
        confusion[1,1] +=1
    elif (values !=1):
        confusion[values,1] = confusion[values,1]+1
        yanlis = np.append(yanlis,np.stack([data]),axis=0)
for data in c:
    values = np.argmax(np.stack([-np.log(2*math.pi) - 0.5*np.log(np.linalg.det(sample_covariences[c]))-0.5*np.matmul((data-sample_means[c]), np.matmul(linalg.cho_solve(linalg.cho_factor(sample_covariences[c]), np.eye(2)), np.transpose((data - sample_means[c])))) +np.log(class_priors[c]) for c in range(4)]), axis=0)
    if (values == 2):
        confusion[2,2] +=1
    elif (values !=2):
        confusion[values,2] = confusion[values,2]+1
        yanlis = np.append(yanlis,np.stack([data]),axis=0)
for data in d:
    values = np.argmax(np.stack([-np.log(2*math.pi) - 0.5*np.log(np.linalg.det(sample_covariences[c]))-0.5*np.matmul((data-sample_means[c]), np.matmul(linalg.cho_solve(linalg.cho_factor(sample_covariences[c]), np.eye(2)), np.transpose((data - sample_means[c])))) +np.log(class_priors[c]) for c in range(4)]), axis=0)
    if (values == 3):
        confusion[3,3] +=1
    elif (values !=3):
        confusion[values,3] = confusion[values,3]+1
        yanlis = np.append(yanlis,np.stack([data]),axis=0)

print("| y_truth","|","1","|","2","|","3","|","4")
print("| y_pred")
for m in range(len(confusion)):
    print("|",m+1,end="\t")
    for j in range(len(confusion)):
        print("|",confusion[m][j], end="\t")
    print("\n")
    
plt.figure(figsize = (5, 5))

plt.plot(a[:,0], a[:,1], "r.", markersize = 3)
plt.plot(b[:,0], b[:,1], "g.", markersize = 3)
plt.plot(c[:,0], c[:,1], "b.", markersize = 3)
plt.plot(d[:,0], d[:,1], "m.", markersize = 3)
plt.plot(yanlis[:,0], yanlis[:,1], "ok", markersize = 5,markerfacecolor='none')

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()
