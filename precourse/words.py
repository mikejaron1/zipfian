import random
import string
import collections
import sys


def word_counts(f):
    '''
    INPUT: file
    OUTPUT: dictionary

    Return a dictionary whose keys are all the words in the file (broken by
    whitespace). The value for each word is a dictionary containing each word
    that can follow the key and a count for the number of times it follows it.

    You should lowercase everything.
    Use strip and string.punctuation to strip the punctuation from the words.

    Example:
    >>> #example.txt is a file containing: "The cat chased the dog"
    >>> with open('example.txt') as f:
    ...     word_counts(f)
    {'the': {'dog': 1, 'cat': 1}, 'chased': {'the': 1}, 'cat': {'chased': 1}}
    '''

    d = {}
    last = None
    for line in f:
        for word in line.lower().split():
            if last:
                d[last] = d.get(last, {})
                d[last][word] = d[last].get(word, 0) + 1
            last = word
    return d


# Using collections module
def word_counts2(f):
    d = collections.defaultdict(collections.Counter)
    last = None
    for line in f:
        for word in line.lower().split():
            if last:
                d[last][word] += 1
            last = word
    return d


def associated_words(f):
    '''
    INPUT: file
    OUTPUT: dictionary

    Return a dictionary where the keys are tuples of two consecutive words in
    the file and the value for each key is a list of words that were found
    directly following the key.

    Words should be included in the list the number of times they appear.

    Example:
    >>> with open('alice.txt') as f:
    ...     d = associated_words(f)
    >>> d[('among', 'the')]
    ['people', 'party.', 'trees,', 'distant', 'leaves,', 'trees', 'branches,',
    'bright']
    '''

    d = {}
    previous = (None, None)
    for line in f:
        for word in line.lower().split():
            d[previous] = d.get(previous, [])
            d[previous].append(word)
            previous = (previous[1], word)
    return d


def make_random_story(f, num_words):
    '''
    INPUT: file, integer
    OUTPUT: string

    Call associated_words on file f and use the resulting dictionary to
    randomly generate text with num_words total words.

    Choose the next word by using random.choice to pick a word from the list
    of possibilities given the last two words.

    Use join method to turn a list of words into a string.

    Example:
    (this is randomly generate, so you are unlikely to get the same result)
    >>> with open('alice.txt') as f:
    ...     print make_random_story(f, 10)
    chapter i. down the middle, being held up by two
    '''

    words = associated_words(f)
    previous = (None, None)
    story = []
    for i in xrange(num_words):
        word = random.choice(words[previous])
        story.append(word)
        previous = (previous[1], word)
    return " ".join(story)


if __name__ == '__main__':
    # This code will be run if you on the command line run: python words.py
    filename = sys.argv[1]
    num_words = int(sys.argv[2])
    with open(filename) as f:
        print make_random_story(f, num_words)
