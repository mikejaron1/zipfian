*Things to import:*

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk.data
import numpy as np
```

## Feature importances

1. For the 4 of the 20newsgroups corpus (your choice), find the 10 most important words by:

    *Choose 4 of the categories:*

    ```python
    categories = ['comp.graphics', 'rec.sport.baseball', 'sci.med', \
                  'talk.politics.misc']
    data = fetch_20newsgroups(subset='train', categories=categories).data
    ```

    *Do tf-idf transform:*

    ```python
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data).toarray()
    words = vectorizer.get_feature_names()
    ```

    *Helper function for getting top 10 values:*

    ```python
    def get_top_values(lst, n, labels):
        '''
        INPUT: LIST, INTEGER, LIST
        OUTPUT: LIST

        Given a list of values, find the indices with the highest n values.
        Return the labels for each of these indices.

        e.g.
        lst = [7, 3, 2, 4, 1]
        n = 2
        labels = ["cat", "dog", "mouse", "pig", "rabbit"]
        output: ["cat", "pig"]
        '''
        return [labels[i] for i in np.argsort(lst)[-1:-n-1:-1]]
    ```

    * total tf-idf score
    
        ```python
        avg = np.sum(vectors, axis=0) / np.sum(vectors > 0, axis=0)
        print "top 10 by average tf-idf"
        print get_top_values(avg, 10, words)
        ```

        *Results:*

        ```
        [u'xxxx', u'narrative', u'clubbing', u'compartment', u'bram',
         u'sphinx', u'hernia', u'comarow', u'rolandi', u'kewageshig']
        ```

    * average tf-idf score (average only over non-zero values)
    
        ```python
        total = np.sum(vectors, axis=0)
        print "top 10 by total tf-idf"
        print get_top_values(total, 10, words)
        ```

        *Results:*

        ```
        [u'edu', u'com', u'article', u'writes', u'subject', u'lines',
         u'organization', u'university', u'cs', u'don']
        ```

    * highest TF score across corpus
    
        ```python
        # redo vectorization without using idf
        vectorizer2 = TfidfVectorizer(use_idf=False)
        # make documents into one giant document for this purpose
        vectors2 = vectorizer2.fit_transform(["\n".join(data)]).toarray()
        print "top 10 by tf across all corpus"
        print get_top_values(vectors2[0], 10, words)
        ```

        *Results:*

        ```
        [u'tilting', u'transfusions', u'opening', u'anecdotal', u'indivicual',
         u'jad', u'tiling', u'jazz', u'fought', u'egyptian']
        ```

2. Do the top 10 words change based on each of the different ranking methods?

    *They are very different. The sum gives more common words, but may not be helpful in distinguishing articles. The average gives uncommon words which could help distinguish articles, but maybe are too rare.*

3. Also do this for each category of article (each of the 20 newsgroups) and compare the top words of each. You should treat each category of newsgroup as a separate "corpus" for this question.

    ```python
    all_newsgroups = fetch_20newsgroups()
    all_data = np.array(all_newsgroups.data)
    for i, category in enumerate(all_newsgroups.target_names):
        data = all_data[all_newsgroups.target == i]
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(data).toarray()
        words = vectorizer.get_feature_names()
        print "Category: %s" % category
        avg = np.sum(vectors, axis=0) / np.sum(vectors > 0, axis=0)
        print "  Top 10 by average tf-idf"
        print "    %s" % ", ".join(get_top_values(avg, 10, words))
        total = np.sum(vectors, axis=0)
        print "  Top 10 by total tf-idf"
        print "    %s" % ", ".join(get_top_values(total, 10, words))
        print "-----------------------------"
    ```

## Ranking

1. For each query, find the 3 most relevant articles from the 20 Newsgroups corpus.

    ```python
    # build vectorizer
    data = fetch_20newsgroups(subset='train', categories=categories).data
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data).toarray()
    words = vectorizer.get_feature_names()

    # get queries from file
    with open('data/queries.txt') as f:
        queries = [line.strip() for line in f]

    tokenized_queries = vectorizer.transform(queries)
    cosine_similarities = linear_kernel(tokenized_queries, vectors)
    titles = newsgroups.filenames
    for i, query in enumerate(queries):
        print query
        print get_top_values(cosine_similarities[i], 3, titles)
        print
    ```