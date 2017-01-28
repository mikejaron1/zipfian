# To run:
# - download & extract http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz
# - run: python word_count_by_topic.py 20_newsgroups

from mrjob.job import MRJob
from string import punctuation
import os


class WordCountByTopic(MRJob):

    def mapper(self, _, line):
        topic = os.environ["map_input_file"].split('/')[-2]
        for word in line.split():
            word = word.strip(punctuation).lower()
            if word:
                yield (topic + "_" + word, 1)

    def reducer(self, key, counts):
        yield (key, sum(counts))

if __name__ == '__main__':
    WordCountByTopic.run()
