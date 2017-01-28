from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.datasets import load_boston
from sklearn import cross_validation, linear_model
import matplotlib.pyplot as plt

# Globally defined variables
boston = load_boston()
features = boston.data # Attribute data
target = boston.target # housing price
xtrain, xtest, ytrain, ytest = \
    train_test_split(features, target, test_size=0.3)
'''
Attribute Information (in order):
- CRIM     per capita crime rate by town
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    proportion of non-retail business acres per town
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      nitric oxides concentration (parts per 10 million)
- RM       average number of rooms per dwelling
- AGE      proportion of owner-occupied units built prior to 1940
- DIS      weighted distances to five Boston employment centres
- RAD      index of accessibility to radial highways
- TAX      full-value property-tax rate per $10,000
- PTRATIO  pupil-teacher ratio by town
- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT    % lower status of the population
- MEDV     Median value of owner-occupied homes in $1000's
'''

def rmse(theta, thetahat):
    ''' Part 1, number 5 '''
    return np.sqrt(np.mean(theta - thetahat) ** 2)


def calc_linear():
	''' Part 1, number 4 '''
	linear = LinearRegression()
	linear.fit(xtrain, ytrain)
	
	train_pred = linear.predict(xtrain)
	test_pred = linear.predict(xtest)

	err_train = rmse(ytrain, train_pred)
	err_pred = rmse(ytest, test_pred)
	return err_train, err_pred

def k_fold_linear():
	''' Returns error for k-fold cross validation. '''
	err_linear, index, num_folds = 0, 0, 10
	m = len(features)
	kf = KFold(m, n_folds = num_folds)
	error = np.empty(num_folds)
	linear = LinearRegression()
	for train, test in kf:
		linear.fit(features[train], target[train])
		pred = linear.predict(features[test])
		error[index] = rmse(pred, target[test])
		index += 1

	return np.mean(error)

def plot_learning_curve(estimator, label=None):
	scores = list()
	train_sizes = np.linspace(10,100,10).astype(int)
	for train_size in train_sizes:
		cv_shuffle = cross_validation.ShuffleSplit(train_size=train_size, 
						test_size=200, n=len(target), random_state=0)
		test_error = cross_validation.cross_val_score(estimator, features, 
						target, cv=cv_shuffle)
		scores.append(test_error)

	plt.plot(train_sizes, np.mean(scores, axis=1), label=label or estimator.__class__.__name__)
	plt.ylim(0,1)
	plt.ylabel('Explained variance on test set')
	plt.xlabel('Training test size')
	plt.legend(loc='best')
	plt.show()


def plot_errors():
	m = features.shape[1]
	err_test, err_train = [], []
	linear = LinearRegression()
	for ind in xrange(m):
		linear.fit(xtrain[:,:(ind+1)], ytrain)

		train_pred = linear.predict(xtrain[:,:(ind + 1)])
		test_pred = linear.predict(xtest[:,:(ind + 1)])

		err_test.append(rmse(test_pred, ytest))
		err_train.append(rmse(train_pred, ytrain))

	x = range(m)
	plt.figure()
	plt.plot(x, np.log(err_test), label='log(Test error)')
	plt.plot(x, err_train, label='Training error')
	plt.legend()
	plt.show()

def main():
    ''' Solutions to Part 1: One-fold Cross Validation '''
    print '(Training error, Test error) = ' 
    print calc_linear()
    # The error on the test set is more accurate for prediction
    ''' Solutions to Part 2: K-fold Cross Validation'''
    # print 'k-fold error:', k_fold_linear()
    # estimator = LinearRegression()
    # plot_learning_curve(estimator, label='LinearRegression')
    # 10-fold Cross-Validation returns roughly the same error as 1-fold
    # plot_errors()

if __name__ == '__main__':
    main()
    