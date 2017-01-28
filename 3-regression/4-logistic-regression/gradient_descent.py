import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from helper_functions import compute_cost, predict, sigmoid, accuracy_score, \
                             add_intercept, precision, recall, f1_score
from sklearn.datasets import make_classification

class GradientDescent(object):

    def __init__(self, fit_intercept=True, normalize=False):
        '''
        INPUT: GradientDescent, boolean
        OUTPUT: None

        Initialize class variables. cost is the function used to compute the
        cost.
        '''
        self.coeffs = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.mu = None
        self.sigma = None

    def cost(self, X, y):
        '''
        INPUT: Gradient Descent, 2 dimensional numpy array, numpy array
        OUTPUT: float

        Compute the cost according to the instance variable cost (our
        designated cost function).
        '''
        return compute_cost(X, y, self.coeffs)

    def score(self, X, y):
        '''
        INPUT: 2 dimensional numpy array, numpy array
        OUTPUT: float

        Compute the score of 
        '''
        return accuracy_score(self.maybe_modify_matrix(X), y, self.coeffs)

    def gradient(self, X, y, coeffs):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array, numpy array
        OUTPUT: numpy array

        Compute the gradient of the cost function evaluated using coeffs
        instance variable.
        '''
        return X.T.dot(y - sigmoid(X.dot(coeffs)))

    def run(self, X, y, alpha=0.01, num_iterations=100):
        self.calculate_normalization_factors(X)
        X = self.maybe_modify_matrix(X)
        self.coeffs = np.zeros(X.shape[1])
        for i in xrange(num_iterations):
            self.coeffs += alpha / X.shape[0] * self.gradient(X, y, self.coeffs)

    def predict(self, X):
        '''
        INPUT: GradientDescent, 2 dimesional numpy array
        OUTPUT: numpy array

        Use coeffs instance variable to compute the prediction for X.
        '''
        return predict(self.maybe_modify_matrix(X), self.coeffs)

    def calculate_normalization_factors(self, X):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array
        OUTPUT: None

        Initialize mu and sigma instance variables to be the numpy arrays
        containing the mean and standard deviation for each column of X.
        '''
        self.mu = np.average(X, 0)
        self.sigma = np.std(X, 0)
        # Don't normalize intercept column
        self.mu[self.sigma == 0] = 0
        self.sigma[self.sigma == 0] = 1

    def maybe_modify_matrix(self, X):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array
        OUTPUT: 2 dimensional numpy array

        Depending on the settings, normalizes X and adds a feature for the
        intercept.
        '''
        if self.normalize:
            X = (X - self.mu) / self.sigma
        if self.fit_intercept:
            return add_intercept(X)
        return X


def run_loan_data():
    print "Loan Data..."
    df = pd.read_csv('data/loanf.csv')
    df = df.dropna()
    df['Label'] = df['Interest.Rate'] <= 12
    print df['Label'].value_counts()
    X = df[['FICO.Score', 'Loan.Length', 'Monthly.Income', 'Loan.Amount']].values
    y = df['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape
    gd = GradientDescent()
    gd.run(X_train, y_train)
    print "Coeffs:", gd.coeffs
    print "Accuracy:", gd.score(X_test, y_test)
    X_test_intercept = add_intercept(X_test)
    print "Precision:", precision(X_test_intercept, y_test, gd.coeffs)
    print "Recall:", recall(X_test_intercept, y_test, gd.coeffs)
    print "F1 score:", f1_score(X_test_intercept, y_test, gd.coeffs)


def run_fake_data():
    print "Fake Data..."
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, class_sep=5,
                               random_state=5)
    gd = GradientDescent()
    print X.shape
    gd.run(X, y)
    print "accuracy:", gd.score(X, y)


if __name__ == '__main__':
    run_fake_data()
    run_loan_data()
