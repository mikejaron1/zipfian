# This is but one way of approaching this. There is far from one correct
# solution.
# Here's the output of this code:

# distribution of labels:
# 0: 1410
# 1: 1410
# -----------------------------
# Without Lemmatization:
# acc f1  prec    recall
# 0.9191  0.9165  0.9315  0.9020  LogisticRegression
# 0.8794  0.8726  0.9094  0.8386  KNeighborsClassifier
# 0.9064  0.9057  0.8980  0.9135  MultinomialNB
# 0.8823  0.8748  0.9177  0.8357  RandomForestClassifier
# -----------------------------
# With Lemmatization:
# acc f1  prec    recall
# 0.9163  0.9134  0.9311  0.8963  LogisticRegression
# 0.8908  0.8866  0.9066  0.8674  KNeighborsClassifier
# 0.9163  0.9156  0.9091  0.9222  MultinomialNB
# 0.8865  0.8813  0.9083  0.8559  RandomForestClassifier
# -----------------------------


from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


# You can get an accuracy of around 90% with either Logistic Regression, Naive
# Bayes or Random Forest.
# Lemmatizing doesn't seem to affect the accuracy.


def get_descriptions(filename):
    with open(filename) as f:
        return [line for line in f]


def get_labels(filename):
    with open(filename) as f:
        return np.array([int(line) for line in f])


def lemmatize_descriptions(descriptions):
    lem = WordNetLemmatizer()
    lemmatize = lambda d: " ".join(lem.lemmatize(word) for word in d.split())
    return [lemmatize(desc) for desc in descriptions]


def get_vectorizer(descriptions, num_features=5000):
    vect = TfidfVectorizer(max_features=num_features, stop_words='english')
    return vect.fit(descriptions)


def run_model(Model, X_train, X_test, y_train, y_test):
    m = Model()
    m.fit(X_train, y_train)
    y_predict = m.predict(X_test)
    return accuracy_score(y_test, y_predict), \
        f1_score(y_test, y_predict), \
        precision_score(y_test, y_predict), \
        recall_score(y_test, y_predict)


def compare_models(descriptions, labels, models):
    desc_train, desc_test, y_train, y_test = \
        train_test_split(descriptions, labels)

    print "-----------------------------"
    print "Without Lemmatization:"
    run_test(models, desc_train, desc_test, y_train, y_test)

    print "-----------------------------"
    print "With Lemmatization:"
    run_test(models, lemmatize_descriptions(desc_train),
             lemmatize_descriptions(desc_test), y_train, y_test)

    print "-----------------------------"


def run_test(models, desc_train, desc_test, y_train, y_test):
    vect = get_vectorizer(desc_train)
    X_train = vect.transform(desc_train).toarray()
    X_test = vect.transform(desc_test).toarray()

    print "acc\tf1\tprec\trecall"
    for Model in models:
        name = Model.__name__
        acc, f1, prec, rec = run_model(Model, X_train, X_test, y_train, y_test)
        print "%.4f\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, name)


if __name__ == '__main__':
    descriptions = get_descriptions('data/train.txt')
    labels = get_labels('data/labels.txt')
    print "distribution of labels:"
    for i, count in enumerate(np.bincount(labels)):
        print "%d: %d" % (i, count)
    models = [LogisticRegression, KNeighborsClassifier, MultinomialNB,
              RandomForestClassifier]
    compare_models(descriptions, labels, models)
