from mrjob.job import MRJob
import re
import os

WORD_RE = re.compile(r"[\w']+")


class MRWordLength(MRJob):
    def mapper(self, _, line):
        filename = os.environ['map_input_file'].split("/")[-1]
        for word in WORD_RE.findall(line):
            yield filename, len(word)

    def reducer(self, filename, lengths):
        total = 0
        count = 0
        for word_len in lengths:
            total += word_len
            count += 1
        yield filename, total / float(count)


if __name__ == '__main__':
    MRWordLength.run()
