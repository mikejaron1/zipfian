from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk.data
import numpy as np


# There's a function for each section of the sprint as well as some additional
# helper functions.


def main():
    # GET THE DATA
    # set categories = None to get all the data. Will be slow.
    categories = ['comp.graphics', 'rec.sport.baseball', 'sci.med', \
                  'talk.politics.misc']
    newsgroups = fetch_20newsgroups(subset='train', categories=categories)

    # DO TFIDF TRANSFORMATION
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(newsgroups.data).toarray()

    # FEATURE IMPORTANCES
    top_n(vectorizer, vectors, newsgroups.data, 10)

    # RANKING
    ranking(vectorizer, vectors, newsgroups.filenames,
            get_queries('queries.txt'), 3)

    # SUMMARIZATION
    article = newsgroups.data[1599]  # can choose any article
    summarization(article, categories, 3)


def top_n(vectorizer, vectors, data, n):
    '''
    Print out the top 10 words by three different sorting mechanisms:
        * average tf-idf score
        * total tf-idf score
        * highest TF score across corpus
    '''
    words = vectorizer.get_feature_names()

    # Top 10 words by average tfidf
    # Take the average of each column, denoted by axis=0
    avg = np.sum(vectors, axis=0) / np.sum(vectors > 0, axis=0)
    print "top %d by average tf-idf" % n
    print get_top_values(avg, n, words)
    print

    # Top 10 words by total tfidf
    total = np.sum(vectors, axis=0)
    print "top %d by total tf-idf" % n
    print get_top_values(total, n, words)
    print

    # Top 10 words by TF
    vectorizer2 = TfidfVectorizer(use_idf=False)
    # make documents into one giant document for this purpose
    vectors2 = vectorizer2.fit_transform([" ".join(data)]).toarray()
    print "top %d by tf across all corpus" % n
    print get_top_values(vectors2[0], n, words)
    print


def get_queries(filename):
    '''
    Return a list of strings of the queries in the file.
    '''
    queries = []
    with open('queries.txt') as f:
        for line in f:
            # horrible stuff to get out the query
            queries.append(line.split("   ")[1].split("20")[0].strip())
    return queries


def ranking(vectorizer, vectors, titles, queries, n):
    '''
    Print out the top n documents for each of the queries.
    '''
    tokenized_queries = vectorizer.transform(queries)
    cosine_similarities = linear_kernel(tokenized_queries, vectors)
    for i, query in enumerate(queries):
        print query
        print get_top_values(cosine_similarities[i], 3, titles)
        print


def summarize(article, sent_detector, n):
    '''
    Choose top n the sentences based on max tf-idf score.
    '''
    sentences = sent_detector.tokenize(article.strip())
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences).toarray()
    # We are summing on axis=1 (total per row)
    total = np.sum(vectors, 1)
    lengths = np.array([len(sent) for sent in sentences])
    return get_top_values(total / lengths.astype(float), n, sentences)


def summarization(article, categories, n):
    '''
    Print top n sentences from the article.
    Print top n sentences from each category.
    '''
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # summarize an article
    print summarize(article, sent_detector, n)
    print

    for cat in categories:
        newsgroup = fetch_20newsgroups(subset='train', categories=[cat])
        print cat
        # combine all articles into one string to summarize.
        print summarize("\n".join(newsgroup.data), sent_detector, n)


def get_top_values(lst, n, labels):
    '''
    INPUT: LIST, INTEGER, LIST
    OUTPUT: LIST

    Given a list of values, find the indices with the highest n values. Return
    the labels for each of these indices.

    e.g.
    lst = [7, 3, 2, 4, 1]
    n = 2
    labels = ["cat", "dog", "mouse", "pig", "rabbit"]
    output: ["cat", "pig"]
    '''
    return [labels[i] for i in np.argsort(lst)[-1:-n-1:-1]]


if __name__ == '__main__':
    main()

