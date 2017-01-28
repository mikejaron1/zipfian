from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from scipy.spatial.distance import pdist

import numpy as np
import matplotlib.pyplot as plt


def simulate_data(n_points=300):
    np.random.seed(0)
    X = np.random.randn(n_points, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    return X,Y

def rmse(theta, thetahat):
    return np.sqrt(np.mean(theta - thetahat) ** 2)

def k_fold_logreg(features, target):
    '''
    Returns error for k-fold cross validation.
    '''
    index, num_folds = 0, 5
    m = len(features)
    kf = KFold(m, n_folds = num_folds)
    error = np.empty(num_folds)
    logreg = LogisticRegression()
    for train, test in kf:
        logreg.fit(features[train], target[train])
        pred = logreg.predict(features[test])
        error[index] = rmse(pred, target[test])
        index += 1

    return np.mean(error)

def plot_boundary(X, Y, logreg):
    '''
    Input numpy array, LogisticRegression
    '''
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()


def kernalize(XXTrans, d=2):
    '''
    Input numpy array
    Output numpy arra
    
    Input is the symmetric matrix X.XT and the output is the kernel matrix
    '''
    
    ker = lambda x : (1+x)**d
    vecfunc = np.vectorize(ker)
    return vecfunc(XXTrans) # this is kernel matrix


def plot_boundary_kernel(Xtest, Ytest, X, logregK):
    '''
    This funciton will be used to plot decision boundary for test data.
    This is similar to the previous plot function except that this uses kernalize function 
    before predicting the labels
    '''
    h = 0.02  # step size in the mesh
    x_min, x_max = Xtest[:, 0].min() - .5, Xtest[:, 0].max() + .5
    y_min, y_max = Xtest[:, 1].min() - .5, Xtest[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    xx_test = kernalize(np.c_[xx.ravel(), yy.ravel()].dot(X.T))

    Z = logregK.predict(xx_test)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=Ytest, edgecolors='k', cmap=plt.cm.Paired)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

def main():
    X, Y = simulate_data()
    fig1 = plt.figure(1)
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
    plt.show()

    print k_fold_logreg(X,Y)

    # plot boundary
    logreg = LogisticRegression()
    logreg.fit(X,Y)
    fig2 = plt.figure(2)
    plot_boundary(X, Y, logreg)

    # create kernel matrix
    XXTrans = X.dot(X.T)
    kernel = kernalize(XXTrans)

    # fit logistic to kernel matrix    
    logregK = LogisticRegression()
    logregK.fit(kernel,Y)

    #create test data
    Xtest, Ytest = simulate_data(n_points=100)
    fig3 = plt.figure(3)
    plot_boundary_kernel(Xtest, Ytest, X, logregK)


if __name__ == '__main__':
    main()





