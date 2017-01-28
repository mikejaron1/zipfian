import pandas as pd
import numpy as np
from code.load_enron import build_data_frame
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


# PREP YOUR DATA
# 1. Download the Enron spam dataset.
'''Run command line:
    ./download_enron.sh'''

# 2. Load the data into pandas dataframe
df = build_data_frame('../model-comparison/data')

# 3. Investigate your data a little.
counts = df['label'].value_counts()
total = len(df)
print "%d datapoints total." % total
for label, cnt in counts.iteritems():
    print "%d are %s emails." % (cnt, label)
    print "%.2f%% are %s." % (cnt * 100. / total, label)
# 33716 datapoints total.
# 17171 are spam emails.
# 50.93% are spam.
# 16545 are ham emails.
# 49.07% are ham.

# 4. Create a train test split of your dataframe (70/30).
train_index, test_index = train_test_split(range(total))
train_df = df.ix[train_index]
test_df = df.ix[test_index]

# 5. Use the techniques from nlp to create a feature matrix.
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_df['text'])
X_test = tfidf.transform(test_df['text'])
y_train = train_df['label'].values == 'spam'
y_test = test_df['label'].values == 'spam'
print "%d features" % X_train.shape[1]
# 137841 features


# KNN VS NAIVE BAYES
# 1. Train the kNN classifier and time how long the training step takes.
knn = KNeighborsClassifier()
start = time.time()
knn.fit(X_train, y_train)
end = time.time()
print "knn train took %.2f seconds to train." % (end - start)
# knn train took 0.01 seconds to train.

# 2. Predict on the test set, again timeing how long it takes. What's the
# accuracy?
start = time.time()
y_predict = knn.predict(X_test)
end = time.time()
print "knn took %.2f seconds to predict." % (end - start)
print "knn got %.2f%% accuracy." % (accuracy_score(y_test, y_predict) * 100)
# knn took 202.51 seconds to predict.
# knn got 96.8% accuracy.

# 3. Do the same training and testing with Naive Bayes. Compare the timing and
# accuracy results with kNN.
nb = MultinomialNB()
start = time.time()
nb.fit(X_train, y_train)
end = time.time()
print "Naive bayes took %.2f seconds to train." % (end - start)

start = time.time()
y_predict = nb.predict(X_test)
end = time.time()
print "Naive bayes took %.2f seconds to predict." % (end - start)
print "Naive bayes got %.1f%% accuracy." \
    % (accuracy_score(y_test, y_predict) * 100)
# Naive bayes took 0.09 seconds to train.
# Naive bayes took 0.01 seconds to predict.
# Naive bayes got 98.8% accuracy.

# 4. With our Naive Bayes classifier, print out the top 20 words that indicate
# SPAM and the top 20 words that indicate HAM.
features = np.array(tfidf.get_feature_names())
top_ham = features[np.argsort(nb.feature_count_[0])[-1:-11:-1]]
top_spam = features[np.argsort(nb.feature_count_[1])[-1:-11:-1]]
print "top ham:", ", ".join(top_ham)
print "top spam:", ", ".join(top_spam)
# top ham: the, to, enron, and, ect, of, for, in, you, on
# top spam: the, to, and, you, of, your, in, for, is, this


# GENERATIVE VS DISCRIMINATIVE (NAIVE BAYES VS LOGISTIC REGRESSION)
# 1. Repeat the steps above to get the times of the train and predict as well
# as the accuracy for Logisitic Regression.
lr = LogisticRegression()
start = time.time()
lr.fit(X_train, y_train)
end = time.time()
print "Logistic Regression took %.2f seconds to train." % (end - start)

start = time.time()
y_predict = lr.predict(X_test)
end = time.time()
print "Logistic Regression took %.2f seconds to predict." % (end - start)
print "Logistic Regression got %.1f%% accuracy." \
    % (accuracy_score(y_test, y_predict) * 100)
# Logistic Regression took 1.76 seconds to train.
# Logistic Regression took 0.00 seconds to predict.
# Logistic Regression got 98.4% accuracy.

# 2. Use the 20 Newsgroups datasets in scikit-learn with kNN, Logistic
# Regression, and Naive Bayes.
print "20 newsgroups"
newsgroups = fetch_20newsgroups(subset='train')
train_index, test_index = train_test_split(range(len(newsgroups.data)))
data = np.array(newsgroups.data)
data_train = data[train_index]
y_train = newsgroups.target[train_index]
data_test = data[test_index]
y_test = newsgroups.target[test_index]

tfidf = TfidfVectorizer(stop_words='english')
X_train = tfidf.fit_transform(data_train)
X_test = tfidf.transform(data_test)

for Model in [KNeighborsClassifier, MultinomialNB, LogisticRegression]:
    m = Model()
    print Model.__name__
    start = time.time()
    m.fit(X_train, y_train)
    end = time.time()
    print "    %.2f to train." % (end - start)
    start = time.time()
    y_predict = m.predict(X_test)
    end = time.time()
    print "    %.2f to predict." % (end - start)
    print "    %.2f%% accuracy." % (100 * accuracy_score(y_test, y_predict))
# KNeighborsClassifier
#     0.00 to train.
#     5.43 to predict.
#     77.20% accuracy.
# MultinomialNB
#     0.17 to train.
#     0.03 to predict.
#     83.56% accuracy.
# LogisticRegression
#     9.27 to train.
#     0.02 to predict.
#     88.90% accuracy.

# 3. Try to perform a classification on a non-linear dataset.
print "Circle Data"
X, y = make_circles(n_samples=10000)
X = X - np.min(X)  # make all values nonnegative (requirement of kNN)
X_train, X_test, y_train, y_test = train_test_split(X, y)
for Model in [KNeighborsClassifier, MultinomialNB, LogisticRegression]:
    m = Model()
    print Model.__name__
    start = time.time()
    m.fit(X_train, y_train)
    end = time.time()
    print "    %.2f to train." % (end - start)
    start = time.time()
    y_predict = m.predict(X_test)
    end = time.time()
    print "    %.2f to predict." % (end - start)
    print "    %.2f%% accuracy." % (100 * accuracy_score(y_test, y_predict))
# KNeighborsClassifier
#     0.01 to train.
#     0.01 to predict.
#     100.00% accuracy.
# MultinomialNB
#     0.01 to train.
#     0.00 to predict.
#     54.04% accuracy.
# LogisticRegression
#     0.01 to train.
#     0.00 to predict.
#     49.84% accuracy.

# 4. Plot the decision boundary for each of the three models: kNN, Naive Bayes,
# Logistic Regression.
# Make fake data.
X, y = make_circles(n_samples=1000, noise=0.1)
X = X - np.min(X)  # make all values nonnegative (requirement of kNN)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Make mesh covering area of data.
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))

models = [KNeighborsClassifier, MultinomialNB, LogisticRegression]

for i, Model in enumerate(models):
    # plot the data
    ax = plt.subplot(1, len(models), i + 1)
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='red')
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    m = Model()
    m.fit(X_train, y_train)
    mesh_predict = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, mesh_predict, alpha=0.8, cmap=plt.cm.Paired)
    ax.set_title(Model.__name__)
plt.show()

# 5. For the following datasets, plot the error (or accuracy) of the classifier
# as the number of training examples increases (learning curves).
data = np.genfromtxt('data/housing.data')
housing_df = pd.DataFrame(data)
housing_df.columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis',
                      'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']
med_price = np.mean(housing_df['medv'])
housing_df['label'] = housing_df['medv'] > med_price
X = housing_df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis',
                'rad', 'tax', 'ptratio', 'b', 'lstat']].values
y = housing_df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
ordered_indices = np.random.permutation(range(len(X_train)))

train_sizes = range(10, len(ordered_indices), 1)
accuracies = []
for i in train_sizes:
    lr = LogisticRegression()
    lr.fit(X_train[ordered_indices[:i]], y_train[ordered_indices[:i]])
    accuracies.append(lr.score(X_test, y_test))

plt.plot(train_sizes, accuracies)
plt.show()

