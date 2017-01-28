from collections import Counter, defaultdict
import numpy as np


class NaiveBayes(object):

    def __init__(self, alpha=1):
        '''
        INPUT:
        - alpha: float, laplace smoothing constant
        '''

        self.class_totals = None
        self.class_feature_totals = None
        self.class_counts = None
        self.alpha = 1

    def _compute_likelihood(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        Compute the totals for each class and the totals for each feature
        and class.
        '''

        self.class_totals = Counter()
        self.class_feature_totals = defaultdict(Counter)

        for feat_vect, label in zip(X, y):
            # A grand sum of the value of all the features for a given class
            self.class_totals[label] += np.sum(feat_vect)

            for j, val in enumerate(feat_vect):
                # The sum of each feature for a given class
                self.class_feature_totals[label][j] += val

    def fit(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT: None
        '''

        # compute priors
        self.class_counts = Counter(y)

        # compute likelihoods
        self._compute_likelihood(X, y)

    def predict(self, X):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix

        OUTPUT:
        - predictions: numpy array
        '''

        predictions = np.zeros(X.shape[0])
        for i, feat_vect in enumerate(X):
            posteriors = Counter()
            for label, cnt in self.class_counts.iteritems():
                posteriors[label] = np.log(cnt)
                for j, val in enumerate(feat_vect):
                    numerator = self.class_feature_totals[label][j] + \
                                self.alpha
                    denominator = self.class_totals[label] + \
                                  self.alpha * X.shape[1]
                    likelihood = numerator / float(denominator)
                    posteriors[label] += val * np.log(likelihood)
            predictions[i] = posteriors.most_common()[0][0]
        return predictions

    def score(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - accuracy: float between 0 and 1

        Calculate the accuracy, the percent predicted correctly.
        '''

        return sum(self.predict(X) == y) / float(len(y))
