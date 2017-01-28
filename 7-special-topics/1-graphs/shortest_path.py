from Queue import Queue
from load_imdb_data import load_imdb_data
from sys import argv


def shortest_path(actors, movies, actor1, actor2):
    '''
    INPUT:
        actors: dictionary of adjacency list of actors
        movies: dictionary of adjacency list of movies
        actor1: actor to start at
        actor2: actor to search for

    OUTPUT:
        path: list of actors and movies that starts at actor1 and ends at
              actor2

    Return the shortest path from actor1 to actor2. If there is more than one
    path, return any of them.
    '''
    q = Queue()
    if actor1 not in actors or actor2 not in actors:
        return None
    q.put((actor1, (actor1,)))
    while not q.empty():
        actor, path = q.get()
        if actor == actor2:
            return path
        for movie in actors[actor]:
            for next_actor in movies[movie]:
                q.put((next_actor, path + (movie, next_actor)))
    return None


def print_path(path):
    '''
    INPUT:
        path: list of strings (node names)
    OUTPUT: None

    Print out the length of the path and all the nodes in the path.
    '''
    if path:
        print "length:", len(path) / 2
        for i, item in enumerate(path):
            if i % 2 == 0:
                print "    %s" % item
            else:
                print item
    else:
        print "No path!"


if __name__ == '__main__':
    if len(argv) == 1:
        filename = 'data/imdb_edges.tsv'
    else:
        filename = argv[1]
    actors, movies = load_imdb_data(filename)
    actor1 = raw_input("Enter actor name: ")
    actor2 = raw_input("Enter second actor name: ")
    # Use this line instead if you don't want to type it in:
    # actor1, actor2 = "Kevin Bacon", "Julia Roberts"
    print "Searching for shortest path from %s to %s" % (actor1, actor2)
    path = shortest_path(actors, movies, actor1, actor2)
    print_path(path)
