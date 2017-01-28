# To run:
# - download & extract http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz
# - run: python word_count_by_topic.py 20_newsgroups

from mrjob.job import MRJob
import os
from string import punctuation


class WordCountByTopic(MRJob):

    def mapper(self, _, line):
        topic = os.environ["map_input_file"].split('/')[-2]
        for word in line.replace('_', ' ').split():
            word = word.strip(punctuation).lower()
            if word:
                yield (topic + "_" + word, 1)

    def combiner(self, key, counts):
        yield (key, sum(counts))

    def reducer(self, key, counts):
        yield (key, sum(counts))

    def mapper2(self, key, count):
        topic, word = key.split('_')
        yield word, (topic, count)

    def combiner2(self, word, values):
        yield word, max(values, key=lambda x: x[1])

    def reducer2(self, word, values):
        yield word, max(values, key=lambda x: x[1])[0]

    def steps(self):
        return [
            self.mr(mapper=self.mapper, combiner=self.combiner, reducer=self.reducer),
            self.mr(mapper=self.mapper2, combiner=self.combiner2, reducer=self.reducer2)
        ]

if __name__ == '__main__':
    WordCountByTopic.run()
