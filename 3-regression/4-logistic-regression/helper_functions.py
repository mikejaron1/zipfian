import numpy as np

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def compute_cost(X, y, coeffs):
    y_pred = sigmoid(X.dot(coeffs))
    return y.dot(y_pred) + (1 - y).dot(1 - y_pred)

def accuracy_score(X, y, coeffs):
    y_pred = predict(X, coeffs)
    return float(sum(y_pred == y)) / len(y)

def predict(X, coeffs):
    return np.around(sigmoid(X.dot(coeffs))).astype(bool)

def precision(X, y, coeffs):
    y_pred = predict(X, coeffs)
    TP = y_pred.astype(int).dot(y.astype(int))
    predicted_positive = sum(y_pred)
    return float(TP) / predicted_positive

def recall(X, y, coeffs):
    y_pred = predict(X, coeffs)
    TP = y_pred.astype(int).dot(y.astype(int))
    condition_positive = sum(y)
    return float(TP) / condition_positive

def f1_score(X, y, coeffs):
    p = precision(X, y, coeffs)
    r = recall(X, y, coeffs)
    return float(2 * p * r) / (p + r)


def add_intercept(X):
    '''
    INPUT: 2 dimensional numpy array
    OUTPUT: 2 dimensional numpy array

    Return a new 2d array with a column of ones added as the first
    column of X.
    '''
    return np.hstack((np.ones((X.shape[0], 1)), X))

