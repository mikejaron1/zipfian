from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import pickle
import ipdb # helpful debugger


# STEP 1
#==============================================
# Load in the articles from their compressed pickled state
data = pickle.load( open('data/data.pkl') )

# data is a pandas df
# Make X = our features
X = data['content']

# Make y = our labels
y = data['section_name']

# Convert the thing to a python list so sklearn doesn't freak out
y = list(y)


# STEP 2
#==============================================
# Process our data

# Create our vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform our feature data
vectorized_X  = vectorizer.fit_transform(X)


# STEP 3
#==============================================
# Fit our model

# Create a model
clf = MultinomialNB()

# Fit our model with our vecotrized_X and labels
clf.fit(vectorized_X, y)



# STEP 4
#==============================================
# Export model and vectorizer to use it later

# Export our fitted model via pickle
pickle.dump(clf, open('data/my_model.pkl', 'wb') )

# Export our vectorizer as well
pickle.dump(vectorizer, open('data/my_vectorizer.pkl', 'wb') )
