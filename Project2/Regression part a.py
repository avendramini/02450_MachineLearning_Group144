import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn import model_selection, tree
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

# Load the Iris csv data using the Pandas library
filename = './Data/Data_for_project.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
attributeNames = df.columns[1:-2].tolist()

# Extract vector y, convert to NumPy array
raw_data = df.values  

colonna_da_escludere = 9

# Seleziona tutte le colonne tranne quella da escludere
colonne_da_mantenere = [i for i in range(raw_data.shape[1]) if i != colonna_da_escludere]

X=raw_data[:,colonne_da_mantenere]
X[:,0]=1
y=raw_data[:,colonna_da_escludere]
y=y.astype(float)
N=X.shape[0]
M=X.shape[1]

for i in range(0,N):
    X[i][5]= 1.0 if X[i][5]=="Present" else 0.0

X=X.astype(float)


#different values of the lambda 
lambdas = np.power(10.,range(-5,9))

#REGRESSION PART A (TO MODIFY)
K=10
CV = model_selection.KFold(K, shuffle=True)
M = X.shape[1]
w = np.empty((M,K,len(lambdas)))
train_error = np.empty((K,len(lambdas)))
test_error = np.empty((K,len(lambdas)))
f = 0
y = y.squeeze()
for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
    
    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    for l in range(0,len(lambdas)):
        # Compute parameters for current value of lambda and current CV fold
        # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and test performance
        train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
        test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)

    f=f+1

opt_val_err = np.min(np.mean(test_error,axis=0))
index_opt_lambda=np.argmin(np.mean(test_error,axis=0))
opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
train_err_vs_lambda = np.mean(train_error,axis=0)
#Generalization error for different values of lambda
test_err_vs_lambda = np.mean(test_error,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))

min_error=np.min(np.mean(test_error,axis=0))
weights_lambda=mean_w_vs_lambda[:,index_opt_lambda]

print(f"Minimal error:{min_error}")
print(f"Weights with chosen lambda: {weights_lambda}")

print(f"Optimal lambda:{opt_lambda}")
"""print(f"Optimal value error: {opt_val_err}")
print(f" Average Train error on lambdas: {train_err_vs_lambda}")
print(f"Average Test error on Lambdas: {test_err_vs_lambda}")
print(f"Average weight based on Lambda: {mean_w_vs_lambda}")"""




"""#classification baseline
baseline_chd1=np.sum(y)
baseline_chd0=N-baseline_chd1
baseline_class=1
if baseline_chd1<baseline_chd0:
    baseline_class=0
    
print(baseline_class)"""