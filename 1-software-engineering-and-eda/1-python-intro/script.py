import argparse
from collections import defaultdict
import re


def read_file(filename, delimeter=","):
    f = open(filename)
    d = defaultdict(list)
    for line in f:
        m = re.search("\w+\-(?P<restaurant>[\w\s.']+)\t(?P<rating>\d)", line)
        restaurant = m.group("restaurant").lower()
        rating = int(m.group("rating"))
        d[restaurant].append(rating)
    f.close()
    return d


def average_as_string(lst):
    if lst:
        return str(round(float(sum(lst)) / len(lst), 2))
    return ""


def write_file(filename, d1, d2):
    f = open(filename, "w")
    restaurants = list(set(d1.keys()) | set(d2.keys()))
    restaurants.sort()
    for restaurant in restaurants:
        score1 = average_as_string(d1[restaurant])
        score2 = average_as_string(d2[restaurant])
        f.write("%s,%s,%s\n" % (restaurant, score1, score2))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="first filename", type=str)
    parser.add_argument("file2", help="second filename", type=str)
    parser.add_argument("out", help="out filename", type=str)
    args = parser.parse_args()
    d1 = read_file(args.file1)
    d2 = read_file(args.file2)
    write_file(args.out, d1, d2)
