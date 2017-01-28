from gensim.models.ldamodel import LdaModel
from gensim import corpora
from gensim import matutils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
import pandas as pd
from gensim import corpora, models, similarities

data_f = pd.read_pickle('articles.pkl')
docs = data_f['content']

def tokenize_and_normalize(chunks):
    words = [ tokenize.word_tokenize(sent) for sent in tokenize.sent_tokenize(chunks) ]
    flatten = [ inner for sublist in words for inner in sublist ]
    stripped = [] 

    for word in flatten: 
        if word not in stopwords.words('english'):
            try:
                stripped.append(word.encode('latin-1').decode('utf8').lower())
            except:
                #print "Cannot encode: " + word
                pass
            
    return [ word for word in stripped if len(word) > 1 ] 

<<<<<<< HEAD
def print_features(clf, vocab, n=10):
    """ Print sorted list of non-zero features/weights. """
    coef = clf.coef_[0]
    print 'positive features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[::-1][:n] if coef[j] > 0]))
    print 'negative features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[:n] if coef[j] < 0]))

lda = fit_lda(X,vocab)

lda.show_topics(num_topics=10,num_words=50,formatted=False)
=======
parsed = [ tokenize_and_normalize(s) for s in docs ]

dictionary = corpora.Dictionary(parsed)
corpus = [dictionary.doc2bow(text) for text in parsed]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

%time lda=LdaModel(corpus_tfidf, id2word=dictionary, num_topics=15, update_every=0, passes=200)
lda.print_topics(15, 15)
>>>>>>> 1d640ebed26489de96c0c344bb60372ccf63aff4
