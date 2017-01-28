from graph import Vertex, Graph, build_graph
from Queue import Queue, PriorityQueue
import heapq


def shortest_path(g, start, end):
    q = Queue()
    q.put((start, 0))
    v_set = set()
    while not q.empty():
        (node, dist) = q.get()
        if node == end:
            return dist
        elif node not in v_set:
            v_set.add(node)
            for neighbor in g.get_neighbors(node):
                q.put((neighbor, dist + 1))


def dijkstras(g, start, end):
    q = []
    heapq.heappush(q, (0, start))
    v_set = set()
    while q:
        (dist, node) = heapq.heappop(q)
        if node == end:
            return dist
        elif node not in v_set:
            v_set.add(node)
            for neighbor, weight in g.get_neighbors(node).iteritems():
                heapq.heappush(q, (dist + weight, neighbor))


if __name__ == '__main__':
    g = build_graph()

    print "BFS distance from mission to sunset:",
    print shortest_path(g, 'mission', 'sunset')
    print
    print "BFS distance from mission to soma:",
    print shortest_path(g, 'mission', 'soma')
    print

    print "distance including weights from mission to sunset:",
    print dijkstras(g, 'mission', 'sunset')
    print
    print "distance including weights from mission to soma:",
    print dijkstras(g, 'mission', 'soma')
    print