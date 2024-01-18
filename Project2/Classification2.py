import numpy as np
import pandas as pd


from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import statistics
from math import sqrt

warnings.filterwarnings('ignore')
filename = './Data/Data_for_project.csv'
df = pd.read_csv(filename)
raw_data = df.values

attr_col = list(df.columns).index('chd')
cols = list(range(0, attr_col)) + list(range(attr_col + 1, len(df.columns)))

X = raw_data[:, cols]
N = X.shape[0]
for i in range(0, N):
    X[i][5] = 1.0 if X[i][5] == "Present" else 0.0


y = raw_data[:, attr_col]  # the 'target' column
attributeNames = list(df.columns[cols])
classLabels = raw_data[:, -1]
classNames = np.unique(classLabels)
N, M = X.shape
C = len(classNames)
X = X.astype(float)
y = y.astype(float)
X = stats.zscore(X)
log_model_with_reg = LogisticRegression(penalty='l2', C=1 / 10.0, solver='lbfgs',
                                        fit_intercept=True)
log_model_no_reg = LogisticRegression(solver='lbfgs', fit_intercept=True)

log_model_with_reg = log_model_with_reg.fit(X, y)
log_model_no_reg = log_model_no_reg.fit(X, y)

print('Weights for LogReg model with regularization:')
print('{:>20} {:>20}'.format('Intercept', str(np.round(log_model_with_reg.intercept_[0], 3))))
for m in range(M):
    print('{:>20} {:>20}'.format(attributeNames[m], str(np.round(log_model_with_reg.coef_[0][m], 3))))


fig = plt.figure(figsize=(10, 5))
colors = ['blue', 'green', 'red', 'purple']
# creating the bar plot
plt.bar(attributeNames[1:], log_model_with_reg.coef_[0][1:], color=colors)

plt.xlabel("Atrribute names")
plt.ylabel("Weight")
plt.title("Weights for LogReg model with regularization")
plt.show()

print()
print()

print('Weights for LogReg model without regularization:')
print('{:>20} {:>20}'.format('Intercept', str(np.round(log_model_no_reg.intercept_[0], 3))))
for m in range(M):
    print('{:>20} {:>20}'.format(attributeNames[m], str(np.round(log_model_no_reg.coef_[0][m], 3))))



fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(attributeNames[1:], log_model_no_reg.coef_[0][1:], color=colors)

plt.xlabel("Atrribute names")
plt.ylabel("Weight")
plt.title("Weights for LogReg model without regularization")
plt.show()





def plot_confidence_interval(x, values, z=1, color='#2187bb', horizontal_line_width=0.25):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')

    return mean, confidence_interval


plt.xticks([1, 2, 3], ['BLvsLR', 'BLvsANN', 'ANNvsLR'])
plt.title('Confidence Interval')
plt.axhline(y=0, color='black', linestyle='--')
plot_confidence_interval(1, [-0.36, -0.21])
plot_confidence_interval(2, [-0.31, -0.11])
plot_confidence_interval(3, [-0.01, 0.13])
plt.show()