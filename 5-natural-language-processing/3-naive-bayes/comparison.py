'''
Output look something like this:

tuning naive bayes...
alpha  score
 0.00  0.146634
 0.02  0.837728
 0.04  0.840571
 0.06  0.839864
 0.08  0.834170
 0.10  0.834167
 0.20  0.804260
 0.30  0.787169
 0.40  0.767946
 0.50  0.755127
 0.60  0.740888
 0.70  0.731638
 0.80  0.720959
 0.90  0.704589
 1.00  0.691061
 1.10  0.683228
 1.20  0.674684

running models...
                Name   Score TrainTime  TestTime
       Random Forest   0.730      1.49      0.06
       Decision Tree   0.608     16.31      0.03
                 kNN   0.798      1.89     24.69
         Naive Bayes   0.750      0.59      0.06
                 SVM   0.807     97.71     34.25
            Logistic   0.864      2.07      0.39
'''

import pandas as pd
import numpy as np
import time
import cPickle as pickle
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split, LeaveOneOut, KFold
from sklearn.feature_extraction.text import TfidfVectorizer


def get_data(filename):
    with open(filename) as f:
        df = pickle.load(f)
    data = df[0].values
    labels = df[1].values
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return data, y


def tune_naive_bayes(data, y):
    print "tuning naive bayes..."
    kfold = KFold(len(data))
    alphas = np.concatenate((np.arange(0, 0.1, 0.02), np.arange(.1, 1.3, 0.1)))
    scores = defaultdict(list)
    for train_index, test_index in kfold:
        data_train, data_test = data[train_index], data[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tfidf = TfidfVectorizer()
        X_train = tfidf.fit_transform(data_train)
        print X_train.shape
        X_test = tfidf.transform(data_test)
        for alpha in alphas:
            nb = MultinomialNB(alpha=alpha)
            nb.fit(X_train, y_train)
            scores[alpha].append(nb.score(X_test, y_test))

    print "alpha  score"
    for alpha in alphas:
        print " %.2f  %f" % (alpha, np.average(scores[alpha]))


def run_models(data, y):
    data_train, data_test, y_train, y_test = train_test_split(data, y)

    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(data_train).toarray()
    X_test = tfidf.transform(data_test).toarray()

    print "running models..."
    models = [("Random Forest", RandomForestClassifier()),
              ("Decision Tree", DecisionTreeClassifier()),
              ("kNN", KNeighborsClassifier()),  
              ("Naive Bayes", MultinomialNB()),
              ("SVM", OneVsRestClassifier(SVC())),
              ("Logistic", OneVsRestClassifier(LogisticRegression()))]

    print "%20s %7s %9s %9s" % ("Name", "Score", "TrainTime", "TestTime")

    for name, model in models:
        start = time.time()
        model.fit(X_train, y_train)
        trained = time.time()
        score = model.score(X_test, y_test)
        tested = time.time()

        # Silly stuff to make it print nicely
        print "%20s   %.3f %9s %9s" % (name, score, \
                                       str(round(trained - start, 2)), \
                                       str(round(tested - trained, 2)))


if __name__ == '__main__':
    data, y = get_data("data/data.pkl")
    tune_naive_bayes(data, y)
    run_models(data, y)
