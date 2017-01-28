# bash command to plot output:
#
# python triangle_solution.py edges.txt

from mrjob.job import MRJob
import pdb
import itertools


class MRTriangles(MRJob):

    def mapper_connections(self, _, line):
        source, dest = line.split(' ')
        yield int(source), int(dest)
        yield int(dest), int(source)

    def reducer_group_friends(self, key, values):
        yield key, list(values)

    def mapper_generate_pairs(self, key, values):
        for friend in values:
            yield sorted((key, friend)), 0

        for friend_pair in itertools.combinations(values, 2):
            yield sorted(friend_pair), 1

    def reducer_count_triangles(self, key, values):
        total = 0
        for val in values:
            if val == 0:
                return
            total += val
        yield key, total

    def mapper_suggestion(self, key, value):
        friend1, friend2 = key
        yield friend1, (friend2, value)
        yield friend2, (friend1, value)

    def reducer_suggestion(self, key, values):
        yield key, max(values, key=lambda x: x[1])[0]

    def steps(self):
        return [
            self.mr(mapper=self.mapper_connections,
                    reducer=self.reducer_group_friends),
            self.mr(mapper=self.mapper_generate_pairs,
                    reducer=self.reducer_count_triangles),
            self.mr(mapper=self.mapper_suggestion,
                    reducer=self.reducer_suggestion)
        ]

if __name__ == '__main__':
    MRTriangles.run()
