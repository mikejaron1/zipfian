'''
To run:
python nyt_wordcounts.py data/articles_sep.json
'''

from mrjob.job import MRJob
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import os
import json

STEMMER = PorterStemmer()


class MRWordTopicCount(MRJob):

    def mapper(self, _, line):
        d = json.loads(line)
        headline = d['headline']['main'].replace(" ", "-")
        for l in d['content']:
            for word in word_tokenize(l):
                stemmed_word = STEMMER.stem(word)
                yield ("TOTAL_%s" % word, 1)
                yield ("%s_%s" % (headline, word), 1)

    def reducer(self, key, counts):
        yield (key, sum(counts))


if __name__ == '__main__':
    MRWordTopicCount.run()
