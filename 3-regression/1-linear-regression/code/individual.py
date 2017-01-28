import pandas as pd
import statsmodels.api as sms
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np


def cost_function(mpg, yhat):
    return np.dot(mpg - yhat, mpg - yhat)


def minimize_cost(intercept_space, slope_space, weight, mpg):
    opt_slope, opt_intercept = slope_space[0], intercept_space[0]
    ybar = np.sum(mpg) / len(mpg)
    SSres_old = cost_function(mpg, weight * opt_slope + opt_intercept)
    for intercept in intercept_space:
        for slope in slope_space:
            yhat = intercept + slope * weight
            SSres = cost_function(mpg, yhat)
            if SSres < SSres_old:
                opt_slope, opt_intercept = slope, intercept
                SSres_old = SSres
    return opt_slope, opt_intercept


def show_plots(slope, intercept, weight, mpg):
    ''' Solution to part 1 question 3, 4 '''
    ybar = sum(mpg)/len(mpg)
    yhat = slope * weight + intercept
    plt.figure()
    plt.plot(weight, yhat, 'r', lw=2, label='Estimated Solution')
    plt.scatter(weight, mpg, edgecolor='none', alpha=0.6)
    plt.xlabel('Weight in pounds', fontsize=14)
    plt.ylabel('Miles per Gallon', fontsize=14)
    ''' Solution to part 1 question 4. '''
    xbar = sum(weight)/len(weight)
    cov = np.cov(weight, mpg)
    beta1 = cov[0][1]/cov[0][0]
    beta0 = ybar - beta1 * xbar
    plt.plot(weight, weight * beta1 + beta0, color='g', lw=2,
             label='Actual Solution')
    print 'slope, intercept: %.4f, %.4f ' % (beta1, beta0)
    plt.legend()
    plt.show()


def get_statsmodel(weight, mpg, univariate):
    ''' Solution to part 1 question 5 '''
    # Statsmodel API
    model = sms.OLS(mpg, sms.add_constant(weight))
    result = model.fit()
    result.summary()
    model = smf.ols('mpg ~ weight', data=univariate)
    result = model.fit()
    return result


def residual_plot(result):
    ''' Solution to part 1 question 6 '''
    plt.scatter(range(len(result.resid)), list(result.resid),
                edgecolor='none', alpha=0.5)
    plt.axhline(0, c='r', ls='--')
    plt.axhline(10, c='g', ls='--')
    plt.axhline(-10, c='g', ls='--')
    plt.xlim(0, 500)
    plt.ylabel('Residual', fontsize=14)
    plt.xlabel('Data Points In Order', fontsize=14)
    plt.title('Residual Plot', fontsize=16)
    plt.show()


def main1():
    ''' Importing data and solution to question 3 '''
    univariate = pd.read_csv('data/cars_univariate.csv')
    mpg, weight = univariate['mpg'], univariate['weight']
    intercept_space = np.linspace(46., 48., 51)
    slope_space = np.linspace(0., -0.2, 51)
    slope, intercept = minimize_cost(intercept_space, slope_space, weight, mpg)
    # show_plots(slope, intercept, weight, mpg)
    # result = get_statsmodel(weight, mpg, univariate)
    # print result.summary()
    # residual_plot(result)
    # out = result.outlier_test()
    # out[out['bonf(p)'] < 0.05]


def preprocess(multivariate):
    ''' Solution to Part 2: Multivariate Regression question 1 '''
    # multivariate.info()
    cols = multivariate.columns - ['model', 'origin', 'car_name']
    multivariate = multivariate[cols]
    multivariate = multivariate[multivariate['horsepower'] != '?']
    multivariate['horsepower'] = multivariate['horsepower'].astype(float)
    # multivariate.info()
    return multivariate


def run_statsmodel(model_str, multivariate):
    ''' Solution to Part 2: Multivariate Regression question 2 '''
    print model_str
    print '------------------'
    model = smf.ols(model_str, data=multivariate)
    result = model.fit()
    result.summary()
    df = pd.concat([result.params, result.pvalues], axis=1)
    print df.rename(columns={0: 'coeff.', 1: 'p-val'})
    print '\n\n'


def test_multicollinearity(multivariate):
    ''' Solution to Part 2: Multivariate Regression question 2.
        Testing out a few variables based on the scatter_matrix. explore
        slopes and their p-values to see how significant the effect is mpg '''

    model_1 = 'mpg ~ acceleration + cylinders + \
               displacement + horsepower + weight'
    model_2 = 'mpg ~ weight'
    model_3 = 'mpg ~ cylinders'
    model_4 = 'mpg ~ displacement'
    model_5 = 'mpg ~ horsepower'
    model_6 = 'mpg ~ acceleration'
    model_7 = 'mpg ~ acceleration + weight'
    model_8 = 'mpg ~ acceleration + cylinders + weight'

    run_statsmodel(model_7, multivariate)
    run_statsmodel(model_8, multivariate)
    run_statsmodel(model_1, multivariate)
    # run_statsmodel(model_3, multivariate)
    # run_statsmodel(model_4, multivariate)
    # run_statsmodel(model_5, multivariate)
    # run_statsmodel(model_6, multivariate)


def multivariate_model(multivariate):
    ''' Solution to part 2, number 2. New model to predict mpg. '''
    model = smf.ols('mpg ~ weight + horsepower + acceleration',
                    data=multivariate)
    result = model.fit()
    return result.summary()


def find_correlation(multivariate):
    ''' Solution to part 3, question 1 '''
    corr_mat = multivariate.corr()
    return corr_mat[corr_mat >= 0.8]


def new_model(multivariate):
    ''' Solution to part 3, question 1. New model without collinearity. '''
    model = smf.ols('mpg ~ weight + horsepower ',
                    data=multivariate)
    result = model.fit()
    return result.summary()


def run_statsmodel(model_str, multivariate):
    ''' Solution to Part 3: Multicollinearirty question 2 '''
    model = smf.ols(model_str, data=multivariate)
    result = model.fit()
    return result.rsquared


def print_VIF(multivariate):
    ''' Solution to Part 3: Multicollinearirty question 2'''
    model_1 = 'weight ~ acceleration + cylinders + displacement + horsepower'
    model_2 = 'horsepower ~ acceleration + cylinders + displacement + weight'
    model_3 = 'displacement ~ acceleration + cylinders + horsepower + weight'
    model_4 = 'cylinders ~ acceleration + displacement + horsepower + weight'
    model_5 = 'acceleration ~ cylinders + displacement + horsepower + weight'
    rsquared = {}
    rsquared['weight'] = run_statsmodel(model_1, multivariate)
    rsquared['horsepower'] = run_statsmodel(model_2, multivariate)
    rsquared['displacement'] = run_statsmodel(model_3, multivariate)
    rsquared['cylinders'] = run_statsmodel(model_4, multivariate)
    rsquared['acceleration'] = run_statsmodel(model_5, multivariate)
    for k, v in rsquared.items():
        print 'Dependent variable %s, VIF %.4f' % (k, 1./(1 - v))


def main2():
    ''' Solution to parts 2: Multivariate Regression '''
    multivariate = pd.read_csv('data/cars_multivariate.csv')
    multivariate = preprocess(multivariate)
    # print multivariate.head()
    # pd.scatter_matrix(multivariate, figsize=(15,12), edgecolor='none')
    # plt.show()
    # test_multicollinearity(multivariate)
    # find_multicollinearity(multivariate)
    # print multivariate_model(multivariate)
    ''' Solution to part 3: Multicollinearity '''
    # print find_correlation(multivariate)
    ''' collinear features: acceleration, cylinders, displacement,
                            horsepower and weight '''
    # print new_model(multivariate)
    ''' Solution to part 3 question 2 '''
    # print_VIF(multivariate)


if __name__ == '__main__':
    # main1()
    main2()
