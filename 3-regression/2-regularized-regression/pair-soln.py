import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, linear_model
from sklearn import preprocessing


# set globally used data
boston = load_boston()
features = boston.data
target = boston.target
xtrain, xtest, ytrain, ytest = \
    train_test_split(features, target, test_size=0.3, random_state=42)

# preprocessing
features = preprocessing.scale(features)
xtrain = preprocessing.scale(xtrain)
xtest = preprocessing.scale(xtest)  

''' Functions for pair solution '''
def rmse(predictor, actual):
    return np.sqrt(np.mean((predictor - actual) ** 2))

def explore_params_ridge(features, target):
    '''  Plot weight vectors by exploring params for Lasso regression. '''
    k = features.shape[1]
    alphas = np.linspace(0., 10.)
    params = np.zeros((len(alphas), k))
    for i, a in enumerate(alphas):
        # X_data = preprocessing.scale(features)
        X_data = features
        y = target
        fit = Ridge(alpha=a, normalize=True).fit(X_data, y)
        params[i] = fit.coef_
    plt.figure(figsize=(14,6))

    for param in params.T:
        plt.plot(alphas, param)
    plt.ylabel('Weight vector of features')
    plt.xlabel('Tuning parameter')
    plt.title('Ridge regession weights with varying tuning parameters')
    plt.show()

def split_ridge_error(xtrain, ytrain, xtest, ytest, alpha):
    ''' Fit the same dataset with Ridge Regression with varying 
        tuning parameters. '''
    ridge = Ridge(alpha)

    ridge.fit(xtrain, ytrain)

    pred_train = ridge.predict(xtrain)
    pred_test = ridge.predict(xtest)

    train_error = rmse(pred_train, ytrain)
    test_error = rmse(pred_test, ytest)
        
    return train_error, test_error

def error_curves_ridge(xtrain, ytrain, xtest, ytest):
    ''' Plot test error and training error curves for Ridge regression. '''
    alphas = np.linspace(0.,4.)
    errors_train, errors_test = np.zeros(len(alphas)), np.zeros(len(alphas))
    for i in xrange(len(alphas)):
        errors_train[i], errors_test[i] = split_ridge_error(xtrain, ytrain, xtest, ytest, alphas[i])

    plt.plot(alphas, errors_train, label='train')
    plt.plot(alphas, errors_test, label='test')
    plt.xlabel('Tuning Parameter')
    plt.ylabel('RMSE')
    plt.title('RMSE curves for Ridge regression')
    plt.legend(loc='best')
    plt.show()

''' Functions for Lasso questions '''
def explore_params_lasso(features, target):
    '''  Plot weight vectors by exploring params for Lasso regression. '''
    k = features.shape[1]
    alphas = np.linspace(0.1, .4)
    params = np.zeros((len(alphas), k))
    for i, a in enumerate(alphas):
        X_data = preprocessing.scale(features)
        y = target
        fit = Lasso(alpha=a, normalize=True).fit(X_data, y)
        params[i] = fit.coef_
    plt.figure(figsize=(14,6))

    for param in params.T:
        plt.plot(alphas, param)
    plt.ylabel('Weight vector of features')
    plt.xlabel('Tuning parameter')
    plt.title('Lasso regession weights with varying tuning parameters')
    plt.show()


def split_lasso_error(xtrain, ytrain, xtest, ytest, alpha):
    ''' Fit the same dataset with Lasso Regression with varying 
        tuning parameters. '''
    lasso = Lasso(alpha)

    lasso.fit(xtrain, ytrain)

    pred_train = lasso.predict(xtrain)
    pred_test = lasso.predict(xtest)

    train_error = rmse(pred_train, ytrain)
    test_error = rmse(pred_test, ytest)
    
    return train_error, test_error


def error_curves_lasso(xtrain, ytrain, xtest, ytest):
    ''' Plot test error and training error curves for Ridge regression. '''
    alphas = np.linspace(0.01,.4)
    errors_train, errors_test = np.zeros(len(alphas)), np.zeros(len(alphas))
    for i in xrange(len(alphas)):
        errors_train[i], errors_test[i] = split_lasso_error(xtrain, ytrain,\
                                             xtest, ytest, alphas[i])

    plt.plot(alphas, errors_train, label='train')
    plt.plot(alphas, errors_test, label='test')
    plt.xlabel('Tuning Parameter')
    plt.ylabel('RMSE')
    plt.title('RMSE curves for Lasso regression')
    plt.legend(loc='best')
    plt.show()


def main():
    ''' Ridge questions '''
    # print ridge_error(features, target)
    # explore_params_ridge(features, target)
    error_curves_ridge(xtrain, ytrain, xtest, ytest)
    # Select the parameter with the smallest RMSE in test error curve
    ''' Lasso questions '''
    # explore_params_lasso(features, target)
    # error_curves_lasso(xtrain, ytrain, xtest, ytest)
    # Select the parameter with the smallest RMSE in test error curve

if __name__ == '__main__':
    main()