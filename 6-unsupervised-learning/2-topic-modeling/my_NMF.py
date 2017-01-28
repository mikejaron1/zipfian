import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class MyNMF(object):
    def __init__(self, k, max_iter = 100, tol = 1e-6):
        self.topics = k
        self.max_iter = max_iter
        self.tol = tol
        self.resid = []
        self.weights = None
        self.features = None

    def fit(self, V):
        # Initialize random matrices with mean equal to input matrix
        W = np.random.rand(V.shape[0], self.topics) + np.mean(V)
        H = np.random.rand(self.topics, V.shape[1]) + np.mean(V)
        
        r = 1
        self.resid = [r]
        cnt = 0
        while self.resid[cnt] > self.tol or cnt < self.max_iter:

            W_prev = W
            
            H = H * np.dot(W.T, V) / np.dot(W.T, np.dot(W, H))
           
            W = W * np.dot(V, H.T) / np.dot(np.dot(W, H), H.T)
            r = self.cost(V, W, H)
            resid = np.linalg.norm(W-W_prev)
            self.resid.append(resid)
            cnt += 1
        print "last residual and iteration: %.4f, %d" % (r, cnt)
        self.weights = H
        self.features = W

    def cost(self, V, W, H):
        return np.linalg.norm(V-np.dot(W,H))

if __name__ == '__main__':
    articles = pd.read_pickle('data/articles.pkl')
    articles = articles[articles['content'].apply(lambda x: len(x)) != 0]
    docs = articles['content']

    V = np.random.rand(10, 10)

    nmf = MyNMF(10, max_iter = 100)
    nmf.fit(V)
    Vhat = np.dot(nmf.features, nmf.weights)
