# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import pandas as pd
import numpy as np

# <codecell>

data = pd.read_pickle('articles.pkl')
data.head()

# <codecell>

bodies = data['content']

# <codecell>

bodies.head()

# <codecell>

n_features = 5000
n_topics = 15
n_top_words = 30

vectorizer = TfidfVectorizer(max_features=n_features)
bags = vectorizer.fit_transform(bodies)

# <codecell>

n_features = 15000
n_topics = 15
n_top_words = 30

many_vectorizer = TfidfVectorizer(max_features=n_features)
big_bags = many_vectorizer.fit_transform(bodies)

# <codecell>


# 5000 max features
nmf = decomposition.NMF(n_components=n_topics)
W = nmf.fit_transform(bags)
# nmf.fit(bags)
# nmf.transform(bags)
H = nmf.components_

# <codecell>

# 15000 max featyres
from sklearn import decomposition

nmf = decomposition.NMF(n_components=n_topics)
W = nmf.fit_transform(big_bags)
# nmf.fit(bags)
# nmf.transform(bags)
H = nmf.components_

# <codecell>

bags

# <codecell>

np.dot(W, H)

# <codecell>

H.shape

# <codecell>

feature_words = vectorizer.get_feature_names()

# <codecell>

feature_words

# <codecell>

H.shape

# <codecell>

zip(feature_words, H[0])

# <codecell>

# Get topic 0
keys, values = zip(*sorted(zip(feature_words, H[0]), key = lambda x: x[1])[:-n_top_words:-1])

# <codecell>

val_arr = np.array(values)

# <codecell>

norms = val_arr / np.sum(val_arr)
lists = [12,3,6]
max(1,2,*[4,5])

# <codecell>

topics_dicts = []

for i in xrange(n_topics):
    # n_top_words of keys and values
    keys, values = zip(*sorted(zip(feature_words, H[i]), key = lambda x: x[1])[:-n_top_words:-1])
    val_arr = np.array(values)
    norms = val_arr / np.sum(val_arr)
    #normalize = lambda x: int(x / (max(counter.values()) - min(counter.values())) * 90 + 10)
    topics_dicts.append(dict(zip(keys, np.rint(norms* 300))))

# <codecell>

def unzip(list_arg):
    return zip(*list_arg)

# <codecell>

topics_dicts

# <codecell>

import vincent

vincent.core.initialize_notebook()

for i in xrange(n_topics):
    word_cloud = vincent.Word(topics_dicts[i])
    word_cloud.width = 400
    word_cloud.height = 400
    word_cloud.padding = 0
    word_cloud.display()

# <codecell>

word_cloud.grammar()

# <codecell>


