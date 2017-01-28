from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


def confusion_rates(cm):
    [[tn, fp], [fn, tp]] = cm

    N = fp + tn
    P = tp + fn

    tpr = tp / P
    fpr = fp / N
    fnr = fn / P
    tnr = tn / N

    return np.array([[tpr, fpr], [fnr, tnr]])


def profit_curve(classifiers, cb, X_train, y_train, X_test, y_test):
    pos = np.sum(y_test == 1) / len(y_test)
    neg = 1 - pos
    class_probs = np.array([pos, neg])
    
    for classifier in classifiers:
        # SVM needs to have probabilities turned on to use them
        if classifier.__name__ == 'SVC':
            model = classifier(probability=True)
        else:
            model = classifier()
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:,1]
        indicies = np.argsort(probabilities)[::-1]

        profit = []
        for i in xrange(len(indicies)):
            pred_false = indicies[i:]
            y_predict = np.ones(len(indicies))
            y_predict[pred_false] = 0
            rates = confusion_rates(confusion_matrix(y_test, y_predict))
            profit.append(np.sum(class_probs * rates * cb))

        percentages = np.arange(len(indicies)) / len(indicies) * 100
        plt.plot(percentages, profit, label=classifier.__name__)

    plt.legend(loc="lower right")
    plt.title("Profits of classifiers")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    #plt.ylim(20)
    plt.show()


def prepare_data(filename):
    churn_df = pd.read_csv(filename)

    # Clean up categorical columns
    churn_df['Churn?'] = (churn_df['Churn?'] == 'True.').astype(int)
    yes_no_cols = ["Int'l Plan", "VMail Plan"]
    churn_df[yes_no_cols] = (churn_df[yes_no_cols] == "yes").astype(int)

    # set label array
    y = churn_df.pop('Churn?').values

    # Drop unwanted columns
    churn_df = churn_df.drop(['State', 'Area Code', 'Phone'], axis=1)

    # set feature matrix
    X = churn_df.values

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = prepare_data('data/churn.csv')

    # Cost-Benefit Matrix
    cb = np.array([[79, -20],
                   [0, 0]])

    # Define classifiers for comparison
    classifiers = [RF, LR, GBC, SVC]

    # Plot profit curves
    profit_curve(classifiers, cb, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
