from gradient_ascent import GradientAscent
from regression_functions import log_likelihood, log_likelihood_gradient, \
                                 predict, accuracy, precision, recall
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    data = np.genfromtxt('testdata.csv', delimiter=',')
    X = data[:,0:2]
    y = data[:,2]

    # # Run Gradient Ascent algorithm
    ga = GradientAscent(log_likelihood, log_likelihood_gradient, predict)
    ga.run(X, y, step_size=.0001)
    print "Results from Gradient Ascent implementation:"
    print "  coeffs:", ga.coeffs
    y_pred = ga.predict(X)
    print "  accuracy:", accuracy(y, y_pred)
    print "  precision:", precision(y, y_pred)
    print "  recall:", recall(y, y_pred)

    ## Run Stochastic Gradient Ascent algorithm
    sga = GradientAscent(log_likelihood, log_likelihood_gradient, predict)
    sga.sgd_run(X, y)
    print "Results from Stochastic Gradient Ascent implementation:"
    print "  coeffs:", sga.coeffs
    y_pred = sga.predict(X)
    print "  accuracy:", accuracy(y, y_pred)
    print "  precision:", precision(y, y_pred)
    print "  recall:", recall(y, y_pred)

    # Run sklearn's Logistic Regression
    lr = LogisticRegression()
    lr.fit(X, y)
    print "Results from sklearn Logistic Regression:"
    print "  coeffs:", lr.intercept_[0], lr.coef_[0][0], lr.coef_[0][1]

    # ## Run Gradient Ascent again with regularization
    likelihood_regularized = lambda X, y, coeffs: \
                             log_likelihood(X, y, coeffs, l=10)
    gradient_regularized = lambda X, y, coeffs: \
                           log_likelihood_gradient(X, y, coeffs, l=1)
    gar = GradientAscent(likelihood_regularized, gradient_regularized, predict)
    gar.run(X, y)
    print "Results from Gradient Ascent with regularization"
    print "  coeffs:", gar.coeffs
    y_pred = gar.predict(X)
    print "  accuracy:", accuracy(y, y_pred)
    print "  precision:", precision(y, y_pred)
    print "  recall:", recall(y, y_pred)

    ## draw a scatterplot of the data
    fig, ax = plt.subplots()
    true = X[y == 1]
    false = X[y == 0]
    plt.scatter(true[:,0], true[:,1], marker='+', color='r')
    plt.scatter(false[:,0], false[:,1], marker='.', color='b')

    # # ## draw the line given by the coefficients
    # # GA
    x1, x2 = plt.xlim()
    b0, bx, by = ga.coeffs
    y1 = (-b0 - bx * x1) / by
    y2 = (-b0 - bx * x2) / by
    ax.add_line(Line2D([x1, x2], [y1, y2], color='black', label='gradient ascent'))
    # SGA
    x1, x2 = plt.xlim()
    b0, bx, by = sga.coeffs
    y1 = (-b0 - bx * x1) / by
    y2 = (-b0 - bx * x2) / by
    ax.add_line(Line2D([x1, x2], [y1, y2], color='black', label='stochastic gradient ascent'))
    #Sklearn GA
    b0, bx, by = lr.intercept_[0], lr.coef_[0][0], lr.coef_[0][1]
    y1 = (-b0 - bx * x1) / by
    y2 = (-b0 - bx * x2) / by
    ax.add_line(Line2D([x1, x2], [y1, y2], color='green', label='sklearn'))
    # GA regularized
    b0, bx, by = gar.coeffs
    y1 = (-b0 - bx * x1) / by
    y2 = (-b0 - bx * x2) / by
    ax.add_line(Line2D([x1, x2], [y1, y2], color='magenta', label='gradient ascent regularized'))

    plt.legend()
    plt.title('Fake Data with Decision Boundary')
    plt.show()
