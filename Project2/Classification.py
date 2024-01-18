# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:45:20 2023

@author: egk
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy import stats as st
from sklearn import model_selection, tree
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)
import torch
import numpy as np
from scipy import stats
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_neural_net(model, loss_fn, X, y,
                     n_replicates=3, max_iter=10000, tolerance=1e-6):
    """
    Train a neural network with PyTorch based on a training set consisting of
    observations X and class y. The model and loss_fn inputs define the
    architecture to train and the cost-function update the weights based on,
    respectively.

    Usage:
        Assuming loaded dataset (X,y) has been split into a training and
        test set called (X_train, y_train) and (X_test, y_test), and
        that the dataset has been cast into PyTorch tensors using e.g.:
            X_train = torch.tensor(X_train, dtype=torch.float)
        Here illustrating a binary classification example based on e.g.
        M=2 features with H=2 hidden units:

        >>> # Define the overall architechture to use
        >>> model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, H),  # M features to H hiden units
                    torch.nn.Tanh(),        # 1st transfer function
                    torch.nn.Linear(H, 1),  # H hidden units to 1 output neuron
                    torch.nn.Sigmoid()      # final tranfer function
                    )
        >>> loss_fn = torch.nn.BCELoss() # define loss to use
        >>> net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3)
        >>> y_test_est = net(X_test) # predictions of network on test set
        >>> # To optain "hard" class predictions, threshold the y_test_est
        >>> See exercise ex8_2_2.py for indepth example.

        For multi-class with C classes, we need to change this model to e.g.:
        >>> model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, H), #M features to H hiden units
                            torch.nn.ReLU(), # 1st transfer function
                            torch.nn.Linear(H, C), # H hidden units to C classes
                            torch.nn.Softmax(dim=1) # final tranfer function
                            )
        >>> loss_fn = torch.nn.CrossEntropyLoss()

        And the final class prediction is based on the argmax of the output
        nodes:
        >>> y_class = torch.max(y_test_est, dim=1)[1]

    Args:
        model:          A function handle to make a torch.nn.Sequential.
        loss_fn:        A torch.nn-loss, e.g.  torch.nn.BCELoss() for binary
                        binary classification, torch.nn.CrossEntropyLoss() for
                        multiclass classification, or torch.nn.MSELoss() for
                        regression (see https://pytorch.org/docs/stable/nn.html#loss-functions)
        n_replicates:   An integer specifying number of replicates to train,
                        the neural network with the lowest loss is returned.
        max_iter:       An integer specifying the maximum number of iterations
                        to do (default 10000).
        tolerenace:     A float describing the tolerance/convergence criterion
                        for minimum relative change in loss (default 1e-6)


    Returns:
        A list of three elements:
            best_net:       A trained torch.nn.Sequential that had the lowest
                            loss of the trained replicates
            final_loss:     An float specifying the loss of best performing net
            learning_curve: A list containing the learning curve of the best net.

    """

    import torch
    # Specify maximum number of iterations for training
    logging_frequency = 1000  # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        print('\n\tReplicate: {}/{}'.format(r + 1, n_replicates))
        # Make a new net (calling model() makes a new initialization of weights)
        net = model()

        # initialize weights based on limits that scale with number of in- and
        # outputs to the layer, increasing the chance that we converge to
        # a good solution
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)

        # We can optimize the weights by means of stochastic gradient descent
        # The learning rate, lr, can be adjusted if training doesn't perform as
        # intended try reducing the lr. If the learning curve hasn't converged
        # (i.e. "flattend out"), you can try try increasing the maximum number of
        # iterations, but also potentially increasing the learning rate:
        # optimizer = torch.optim.SGD(net.parameters(), lr = 5e-3)

        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())

        # Train the network while displaying and storing the loss
        print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss', 'Rel. loss'))
        learning_curve = []  # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X)  # forward pass, predict labels on training set
            loss = loss_fn(y_est, y)  # determine loss
            loss_value = loss.data.numpy()  # get numpy array instead of tensor
            learning_curve.append(loss_value)  # record loss for later display

            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value - old_loss) / old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value

            # display loss with some frequency:
            if (i != 0) & ((i + 1) % logging_frequency == 0):
                print_str = '\t\t' + str(i + 1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                print(print_str)
            # do backpropagation of loss and optimize weights
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()

        # display final loss
        print('\t\tFinal loss:')
        print_str = '\t\t' + str(i + 1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
        print(print_str)

        if loss_value < best_final_loss:
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve

    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve


def draw_neural_net(weights, biases, tf,
                    attribute_names=None,
                    figsize=(12, 12),
                    fontsizes=(15, 12)):
    '''
    Draw a neural network diagram using matplotlib based on the network weights,
    biases, and used transfer-functions.

    :usage:
        >>> w = [np.array([[10, -1], [-8, 3]]), np.array([[7], [-1]])]
        >>> b = [np.array([1.5, -8]), np.array([3])]
        >>> tf = ['linear','linear']
        >>> draw_neural_net(w, b, tf)

    :parameters:
        - weights: list of arrays
            List of arrays, each element in list is array of weights in the
            layer, e.g. len(weights) == 2 with a single hidden layer and
            an output layer, and weights[0].shape == (2,3) if the input
            layer is of size two (two input features), and there are 3 hidden
            units in the hidden layer.
        - biases: list of arrays
            Similar to weights, each array in the list defines the bias
            for the given layer, such that len(biases)==2 signifies a
            single hidden layer, and biases[0].shape==(3,) signifies that
            there are three hidden units in that hidden layer, for which
            the array defines the biases of each hidden node.
        - tf: list of strings
            List of strings defining the utilized transfer-function for each
            layer. For use with e.g. neurolab, determine these by:
                tf = [type(e).__name__ for e in transfer_functions],
            when the transfer_functions is the parameter supplied to
            nl.net.newff, e.g.:
                [nl.trans.TanSig(), nl.trans.PureLin()]
        - (optional) figsize: tuple of int
            Tuple of two int designating the size of the figure,
            default is (12, 12)
        - (optional) fontsizes: tuple of int
            Tuple of two ints giving the font sizes to use for node-names and
            for weight displays, default is (15, 12).

    Gist originally developed by @craffel and improved by @ljhuang2017
    [https://gist.github.com/craffel/2d727968c3aaebd10359]

    Modifications (Nov. 7, 2018):
        * adaption for use with 02450
        * display coefficient sign and magnitude as color and
          linewidth, respectively
        * simplifications to how the method in the gist was called
        * added optinal input of figure and font sizes
        * the usage example how  implements a recreation of the Figure 1 in
          Exercise 8 of in the DTU Course 02450
    '''

    # Determine list of layer sizes, including input and output dimensionality
    # E.g. layer_sizes == [2,2,1] has 2 inputs, 2 hidden units in a single
    # hidden layer, and 1 outout.
    layer_sizes = [e.shape[0] for e in weights] + [weights[-1].shape[1]]

    # Internal renaming to fit original example of figure.
    coefs_ = weights
    intercepts_ = biases

    # Setup canvas
    fig = plt.figure(figsize=figsize)
    ax = fig.gca();
    ax.axis('off');

    # the center of the leftmost node(s), rightmost node(s), bottommost node(s),
    # topmost node(s) will be placed here:
    left, right, bottom, top = [.1, .9, .1, .9]

    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Determine normalization for width of edges between nodes:
    largest_coef = np.max([np.max(np.abs(e)) for e in weights])
    min_line_width = 1
    max_line_width = 5

    # Input-Arrows
    layer_top_0 = v_spacing * (layer_sizes[0] - 1) / 2. + (top + bottom) / 2.
    for m in range(layer_sizes[0]):
        plt.arrow(left - 0.18, layer_top_0 - m * v_spacing, 0.12, 0,
                  lw=1, head_width=0.01, head_length=0.02)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing),
                                v_spacing / 8.,
                                color='w', ec='k', zorder=4)
            if n == 0:
                if attribute_names:
                    node_str = str(attribute_names[m])

                else:
                    node_str = r'$X_{' + str(m + 1) + '}$'
                plt.text(left - 0.125, layer_top - m * v_spacing + v_spacing * 0.1, node_str,
                         fontsize=fontsizes[0])
            elif n == n_layers - 1:
                node_str = r'$y_{' + str(m + 1) + '}$'
                plt.text(n * h_spacing + left + 0.10, layer_top - m * v_spacing,
                         node_str, fontsize=fontsizes[0])
                if m == layer_size - 1:
                    tf_str = 'Transfer-function: \n' + tf[n - 1]
                    plt.text(n * h_spacing + left, bottom, tf_str,
                             fontsize=fontsizes[0])
            else:
                node_str = r'$H_{' + str(m + 1) + ',' + str(n) + '}$'
                plt.text(n * h_spacing + left + 0.00,
                         layer_top - m * v_spacing + (v_spacing / 8. + 0.01 * v_spacing),
                         node_str, fontsize=fontsizes[0])
                if m == layer_size - 1:
                    tf_str = 'Transfer-function: \n' + tf[n - 1]
                    plt.text(n * h_spacing + left, bottom,
                             tf_str, fontsize=fontsizes[0])
            ax.add_artist(circle)

    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers - 1:
            x_bias = (n + 0.5) * h_spacing + left
            y_bias = top + 0.005
            circle = plt.Circle((x_bias, y_bias), v_spacing / 8.,
                                color='w', ec='k', zorder=4)
            plt.text(x_bias - (v_spacing / 8. + 0.10 * v_spacing + 0.01),
                     y_bias, r'$1$', fontsize=fontsizes[0])
            ax.add_artist(circle)

            # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                colour = 'g' if coefs_[n][m, o] > 0 else 'r'
                linewidth = (coefs_[n][m, o] / largest_coef) * max_line_width + min_line_width
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                                  c=colour, linewidth=linewidth)
                ax.add_artist(line)
                xm = (n * h_spacing + left)
                xo = ((n + 1) * h_spacing + left)
                ym = (layer_top_a - m * v_spacing)
                yo = (layer_top_b - o * v_spacing)
                rot_mo_rad = np.arctan((yo - ym) / (xo - xm))
                rot_mo_deg = rot_mo_rad * 180. / np.pi
                xm1 = xm + (v_spacing / 8. + 0.05) * np.cos(rot_mo_rad)
                if n == 0:
                    if yo > ym:
                        ym1 = ym + (v_spacing / 8. + 0.12) * np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing / 8. + 0.05) * np.sin(rot_mo_rad)
                else:
                    if yo > ym:
                        ym1 = ym + (v_spacing / 8. + 0.12) * np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing / 8. + 0.04) * np.sin(rot_mo_rad)
                plt.text(xm1, ym1, \
                         str(round(coefs_[n][m, o], 4)), \
                         rotation=rot_mo_deg, \
                         fontsize=fontsizes[1])

    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers - 1:
            layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
            layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        x_bias = (n + 0.5) * h_spacing + left
        y_bias = top + 0.005
        for o in range(layer_size_b):
            colour = 'g' if intercepts_[n][o] > 0 else 'r'
            linewidth = (intercepts_[n][o] / largest_coef) * max_line_width + min_line_width
            line = plt.Line2D([x_bias, (n + 1) * h_spacing + left],
                              [y_bias, layer_top_b - o * v_spacing],
                              c=colour,
                              linewidth=linewidth)
            ax.add_artist(line)
            xo = ((n + 1) * h_spacing + left)
            yo = (layer_top_b - o * v_spacing)
            rot_bo_rad = np.arctan((yo - y_bias) / (xo - x_bias))
            rot_bo_deg = rot_bo_rad * 180. / np.pi
            xo2 = xo - (v_spacing / 8. + 0.01) * np.cos(rot_bo_rad)
            yo2 = yo - (v_spacing / 8. + 0.01) * np.sin(rot_bo_rad)
            xo1 = xo2 - 0.05 * np.cos(rot_bo_rad)
            yo1 = yo2 - 0.05 * np.sin(rot_bo_rad)
            plt.text(xo1, yo1, \
                     str(round(intercepts_[n][o], 4)), \
                     rotation=rot_bo_deg, \
                     fontsize=fontsizes[1])

            # Output-Arrows
    layer_top_0 = v_spacing * (layer_sizes[-1] - 1) / 2. + (top + bottom) / 2.
    for m in range(layer_sizes[-1]):
        plt.arrow(right + 0.015, layer_top_0 - m * v_spacing, 0.16 * h_spacing, 0, lw=1, head_width=0.01,
                  head_length=0.02)

    plt.show()


# Load the data set
filename = './Data/Data_for_project.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as
# is also described in the table in the exercise
attributeNames = df.columns[1:-2].tolist()

# Extract vector y, convert to NumPy array
raw_data = df.values

column_to_predict = 10

# Seleziona tutte le colonne tranne quella da escludere
columns_for_prediction = [i for i in range(raw_data.shape[1]) if i != column_to_predict]

X = raw_data[:, columns_for_prediction]

y = raw_data[:, column_to_predict]
y = y.astype(float)
N = X.shape[0]
M = X.shape[1]

# Change present/not present to binary
for i in range(0, N):
    X[i][5] = 1.0 if X[i][5] == "Present" else 0.0

X = X.astype(float)

ones = np.ones((N, 1), float)

X = X[:, 1:]
print(X)
# K-fold cross validation
K = 10
# Outer loop
K1 = 10
# inner loop
K2 = 10

CV1 = model_selection.KFold(n_splits=K1, shuffle=True)

z_score = stats.zscore(X)

Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
GLR_train_err = np.empty((K, 1))
GLR_test_err = np.empty((K, 1))
GLR_opt_lambda = np.empty(K)
GBL_train_err = np.empty((K, 1))
GBL_test_err = np.empty((K, 1))
GANN_train_err = np.empty((K, 1))
GANN_test_err = np.empty((K, 1))
GANN_opt_h=np.empty(K)
w_rlr = np.empty((M, K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M, K))

CI_BLvsLR=np.empty(K,dtype=object)
CI_BLvsANN=np.empty(K,dtype=object)
CI_ANNvsLR=np.empty(K,dtype=object)
p_BLvsLR=np.empty((K,1))
p_BLvsANN=np.empty((K,1))
p_ANNvsLR=np.empty((K,1))
zBL_LR=np.empty(0)
zBL_ANN=np.empty(0)
zANN_LR=np.empty(0)

# Add the column of ones
X = np.hstack((ones, z_score))

# Initialize variable
logistic_test_error = np.zeros(K1)
ann_test_error = np.zeros(K1)
baseline_test_error = np.zeros(K1)


def rlr2_validate(X, y, hs, cvf=10):  # Modify to include lambda for logistic regression
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.

        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds

        Returns:
        LR_opt_val_err         validation error for optimum lambda
        LR_opt_lambda          value of optimal lambda
        LR_mean_w_vs_lambda    weights as function of lambda (matrix)
        LR_train_err_vs_lambda train error as function of lambda (vector)
        LR_test_err_vs_lambda  test error as function of lambda (vector)
    '''
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    ANN_train_error = np.empty((cvf, len(hs)))
    ANN_test_error = np.empty((cvf, len(hs)))

    f = 0
    errors = []
    for train_index, test_index in CV.split(X, y):
        # print(f"INTERNAL CROSS-VALIDATION, FOLD NUMBER: {f}")
        X_train = torch.Tensor(X[train_index, :])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index, :])
        y_test = torch.Tensor(y[test_index])

        # Standardize the training and set set based on training set moments
        mu = torch.mean(X_train[:, 1:], 0)
        sigma = torch.std(X_train[:, 1:], 0)
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        for h in range(0, len(hs)):
            print("Number of hidden layers:" + str(hs[h]))

            def model(): return torch.nn.Sequential(
                torch.nn.Linear(M, hs[h]),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(hs[h], 1),  # n_hidden_units to 1 output neuron
                torch.nn.Sigmoid()  # final tranfer function
                # no final tranfer function, i.e. "linear output"
            )

            loss_fn = torch.nn.MSELoss()
            X_train2 = torch.Tensor(X_train)
            y_train2 = torch.Tensor(y_train)
            X_test2 = torch.Tensor(X_test)
            y_test2 = torch.Tensor(y_test)
            y_train2 = y_train2.unsqueeze(1)
            y_test2 = y_test2.unsqueeze(1)

            # Trying all hs
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train2,
                                                               y=y_train2,
                                                               n_replicates=1,
                                                               max_iter=10000)
            print('\n\tBest loss: {}\n'.format(final_loss))

            y_sigmoid = net(X_test)  # activation of final note, i.e. prediction of network
            y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8)  # threshold output of sigmoidal function
            y_test = y_test.type(dtype=torch.uint8)
            # Determine  errors and error rate
            e = (y_test_est != y_test)
            ANN_error_rate = (sum(e).type(torch.float) / len(y_test)).data.numpy()
            errors.append(ANN_error_rate)  # store error rate for current CV fold

        f = f + 1

    ANN_opt_val_err = np.min(np.mean(ANN_test_error, axis=0))
    ANN_opt_h = hs[np.argmin(np.mean(ANN_test_error, axis=0))]
    ANN_train_err_vs_h = np.mean(ANN_train_error, axis=0)
    ANN_test_err_vs_h = np.mean(ANN_test_error, axis=0)
    print("Optimal validation error" + str(ANN_opt_val_err))
    print("Optimal number of hidden layers:" + str(ANN_opt_h))
    print("Train error vs hidden layer:" + str(ANN_train_err_vs_h))
    print("Test error vs hidden layer:" + str(ANN_test_err_vs_h))
    return y_sigmoid, ANN_opt_val_err, ANN_opt_h, ANN_train_err_vs_h, ANN_test_err_vs_h


k1 = 0
for par_index, test_index in CV1.split(X):
    print('Computing CV1 fold: {0}/{1}..'.format(k1 + 1, K1))
    print()

    # Find training and test set for current CV fold
    X_par, y_par = X[par_index, :], y[par_index]
    X_test, y_test = X[test_index, :], y[test_index]

    CV2 = model_selection.KFold(n_splits=K2, shuffle=False)

    # Find lambda for logistic regression

    lambda_int = np.power(10., range(-5, 9))
    logreg_generalisation_error_rate = np.zeros(len(lambda_int))
    for i in range(0, len(lambda_int)):
        k2 = 0
        logreg_valid_errorrate = np.zeros(K2)

        for train_index, val_index, in CV2.split(X_par):
            # extract training and test set for current CV fold
            X_train, y_train = X_par[train_index, :], y_par[train_index]

            X_val, y_val = X_par[val_index, :], y_par[val_index]

            logreg_model = LogisticRegression(penalty='l2', C=1 / lambda_int[i], solver='lbfgs')
            logreg_model = logreg_model.fit(X_train, y_train)

            logreg_y_val_estimated = logreg_model.predict(X_val).T
            logreg_valid_errorrate[k2] = np.sum(logreg_y_val_estimated != y_val) / len(y_val)
            k2 = k2 + 1

        logreg_generalisation_error_rate[i] = np.sum(logreg_valid_errorrate) / len(logreg_valid_errorrate)

    logreg_min_error = np.min(logreg_generalisation_error_rate)
    opt_lambda_index = np.argmin(logreg_generalisation_error_rate)
    opt_lambda = lambda_int[opt_lambda_index]

    logreg_model = LogisticRegression(penalty='l2', C=1 / lambda_int[opt_lambda_index], solver='lbfgs')
    logreg_model = logreg_model.fit(X_par, y_par)

    logreg_y_test_estimated = logreg_model.predict(X_test).T
    logistic_test_error[k1] = np.sum(logreg_y_test_estimated != y_test) / len(y_test)
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    lambdaI = GLR_opt_lambda[k1]* np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:, k1] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    LR_y_test_est = X_test @ w_rlr[:, k1]
    print('Error rate - regularized log-reg - CV1 fold {0}/{1}: {2}%'.format(k1 + 1, K1,
                                                                             np.round(100 * logistic_test_error[k1],
                                                                                      decimals=2)))
    print('Optimal lambda: {0}'.format(opt_lambda))
    print()

    # Baseline
    class_1_count = y_par.sum()  # class 1
    class_0_count = len(y_par) - y_par.sum()  # class 0
    baseline_class = float(np.argmax([class_0_count, class_1_count]))
    BL_y_test_est = np.full(y_test.shape[0], y_train.mean())
    baseline_test_error[k1] = np.sum(y_test != baseline_class) / len(y_test)

    print('Error rate - baseline log-reg - CV1 fold {0}/{1}: {2}%'.format(k1 + 1, K1,
                                                                          np.round(100 * baseline_test_error[k1],
                                                                                   decimals=2)))
    print()

    k1 = k1 + 1


    # ANN
    hs = [1,5,10,15,20,25]
    y_train_est, ANN_opt_val_err, ANN_opt_h, ANN_train_err_vs_h, ANN_test_err_vs_h = rlr2_validate(X_train, y_train, hs,
                                                                                         cvf=10)
    ann_test_error[k1] = ANN_test_err_vs_h
    print('Error rate - ANN - CV1 fold {0}/{1}: {2}%'.format(k1 + 1, K1,
                                                                          np.round(100 * ann_test_error[k1],decimals=2)))
    print('Optimal hidden layers: {0}'.format(ANN_opt_h))
    print()
    print()
    print()
    # e = np.array(e)
    # ANN_opt_val_err = np.array(ANN_opt_val_err)
    # ANN_opt_h = np.array(ANN_opt_h)
    # ANN_train_err_vs_h = np.array(ANN_train_err_vs_h)
    # ANN_test_err_vs_h = np.array(ANN_test_err_vs_h)
    # se = (ANN_test_err_vs_h) ** 2  # squared error
    # mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
    # GANN_test_err[k1] = mse[0]
    #
    #
    # train_se = (y_train_est.float() - y_train.float()
    #             ) ** 2  # squared error
    # train_mse = (sum(train_se).type(torch.float) /
    #              len(y_train)).data.numpy()  # mean
    # # store error rate for current CV fold
    # GANN_train_err[k1] = train_mse[0]
    #
    # GANN_opt_h[k1] = ANN_opt_h
    #
    # zBL=np.abs(y_test - BL_y_test_est) ** 2
    # zLR=np.abs(y_test-LR_y_test_est)**2
    # zANN=se.detach().numpy().squeeze()
    #
    # alpha = 0.05
    # CIBL = st.t.interval(1-alpha, df=len(zBL)-1, loc=np.mean(zBL), scale=st.sem(zBL))# Confidence interval
    # CILR = st.t.interval(1-alpha, df=len(zLR)-1, loc=np.mean(zLR), scale=st.sem(zLR))
    # CIANN = st.t.interval(1-alpha, df=len(zANN)-1, loc=np.mean(zANN), scale=st.sem(zANN))
    #
    # # BL VS LR
    # z = zBL - zLR
    # zBL_LR = np.concatenate((zBL_LR, zBL - zLR))
    # CI_BLvsLR[k1] = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    # p_BLvsLR[k1] = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
    #
    # #BL VS ANN
    # z = zBL - zANN
    # zBL_ANN = np.concatenate((zBL_ANN, zBL - zANN))
    # CI_BLvsANN[k1] = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    # p_BLvsANN[k1] = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
    #
    # #ANN VS LR
    # z = zANN - zLR
    # zANN_LR = np.concatenate((zANN_LR, zANN - zLR))
    # CI_ANNvsLR[k1] = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    # p_ANNvsLR[k1] = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value



