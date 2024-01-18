# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:22:05 2023

@author: alberto
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load the Iris csv data using the Pandas library
filename = 'Data/Data_for_project.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
attributeNames = df.columns[1:-2].tolist()

# Extract vector y, convert to NumPy array
raw_data = df.values  
X=raw_data[:,range(1,10)]
y=raw_data[:,10]
N=X.shape[0]
M=X.shape[1]

for i in range(0,N):
    X[i][4]= 1.0 if X[i][4]=="Present" else 0.0
    
X=X.astype(float)

    
standard=zscore(X,ddof=1)
    

U,S,Vh= svd(standard.astype(np.float64),full_matrices=False)

V=Vh.T

rho=(S*S)/(S*S).sum()

threshold=0.75

plt.figure()

plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

  

# Project the centered data onto principal component space
Z = standard @ V

# Indices of the principal components to be plotted
i = 0
j = 1
z = 2

# Plot PCA of the data
f = plt.figure(figsize=(8,8))
ax = f.add_subplot(111, projection='3d')
plt.title('Heart Disease Data: PCA')
#Z = array(Z)
for c in range(2):
    # select indices belonging to class c:
    class_mask = y==c
    ax.plot3D(Z[class_mask,i], Z[class_mask,j], Z[class_mask,z],'.')
plt.legend(["Chd=0","Chd=1"])
ax.set_xlabel('PC{0}'.format(i+1))
ax.set_ylabel('PC{0}'.format(j+1))
ax.set_zlabel('PC{0}'.format(z+1))


# Output result to screen
plt.show()


pcs = [0,1,2,3,4]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .1
r = np.arange(1,M+1)
plt.figure(figsize=(10,5))
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Heart diseases: PCA Component Coefficients')
plt.show()


plt.figure(figsize=(12,6))
plt.title('Heart Disease Data: Boxplot (standardized)')
plt.boxplot(standard)
plt.xticks(range(1,M+1), attributeNames, rotation=45)
plt.show()

# Next, we plot histograms of all attributes.
plt.figure(figsize=(14,9))
plt.subplots_adjust(hspace=0.5)
u = np.floor(np.sqrt(M))
v = np.ceil(float(M)/u)
for i in range(M):
    plt.subplot(int(u),int(v),i+1)
    plt.hist(X[:,i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: plt.yticks([])
    if i==1: plt.title('Heart Disease: Histograms')

"""plt.show()

plt.hist(X[:,1])
plt.xlabel(attributeNames[1])
plt.title('Tobacco Histogram')
plt.show()"""

plt.figure(figsize=(30,30))
plt.subplots_adjust(hspace=0.5)
u = 5
v = 5
for i in range(5):
    for j in range(5):
        if(j>i):
            break
        plt.subplot(u,v,i*5+j+1)
        if(j==0):
            plt.ylabel(f"PC {i+1}",fontsize=40)
        if(i==4):
            plt.xlabel(f"PC {j+1}",fontsize=40)
        for c in range(2):
            # select indices belonging to class c:
            class_mask = y==c
            plt.plot(Z[class_mask,i],Z[class_mask,j],'.')
            
plt.show()
"""
plt.figure()
plt.plot(X[:,3],X[:,6],'.')
plt.xlabel(attributeNames[3])
plt.ylabel(attributeNames[6])
plt.show()"""
    
