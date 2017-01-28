```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class LogitSVMMargin(object):

    def __init__(self, fname, xnames, yname):
        self.df = pd.read_csv(fname)
        self.xnames = xnames
        self.yname = yname
        self.x = self.df[xnames]
        self.y = self.df[yname]


    def plot_data(self, size=None, show=False):
        color = self.y.map(lambda x: 'red' if x == 1 else 'blue')

        x1_name, x2_name = self.xnames
        x1 = self.x[x1_name]
        x2 = self.x[x2_name]

        if size is None:
            plt.scatter(x1, x2, color=color, edgecolor='black',
                        alpha=0.3)
        else:
            plt.scatter(x1, x2, color=color, s=size*100,
                        edgecolor='black', alpha=0.3)

        plt.xlabel(self.xnames[0], fontweight='bold', fontsize=14)
        plt.ylabel(self.xnames[1], fontweight='bold', fontsize=14)

        if show:
            plt.show()

    def fit_logit(self):
        logit = LogisticRegression(random_state=42)
        logit.fit(self.x, self.y)
        return logit

    def fit_svc(self):
        svc = SVC(kernel='linear')
        svc.fit(self.x, self.y)
        return svc

    @staticmethod
    def plot_logit_decision(logit, show=True):
        coefs = logit.coef_[0]
        intercept = logit.intercept_

        x1_coef = coefs[0]
        x2_coef = coefs[1]

        x1_range = np.linspace(5, 35)
        x2_range = -1. * (x1_coef * x1_range + intercept) / x2_coef

        plt.plot(x1_range, x2_range, color='g')

        if show:
            plt.show()

    @staticmethod
    def plot_svc_decision(svc, show=True):
        # get the separating hyperplane
        w = svc.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(5, 35)
        yy = a * xx - (svc.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors
        b = svc.support_vectors_[0]
        yy_down = a * xx + (b[1] - a * b[0])
        b = svc.support_vectors_[-1]
        yy_up = a * xx + (b[1] - a * b[0])

        # plot the line, the points, and the nearest vectors to the plane
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        if show:
            plt.show()

    def calc_margin(self, est):
        coefs = est.coef_
        intercept = est.intercept_

        beta_norm = np.linalg.norm(coefs, ord=2)
        coefs_row = coefs
        margins = abs(coefs_row.dot(self.x.T) + intercept) / beta_norm
        return margins.ravel()


if __name__ == '__main__':
    obj = LogitSVMMargin('data/data_scientist.csv', ['gym_hours', 'email_hours'], 'data_scientist')
    obj.plot_data(size=None, show=True)

    logit = obj.fit_logit()
    logit_margin = obj.calc_margin(logit)
    obj.plot_data(size=logit_margin, show=False)
    obj.plot_logit_decision(logit, show=True)

    svc = obj.fit_svc()
    svc_margin = obj.calc_margin(svc)
    obj.plot_data(size=svc_margin, show=False)
    obj.plot_svc_decision(svc, show=True)

    print 'Logit sum of margin:', logit_margin.sum()
    print 'SVC sum of margin:', svc_margin.sum()
```




