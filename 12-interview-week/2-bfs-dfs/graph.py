from Queue import Queue, LifoQueue


class Vertex:
    def __init__(self, name):
        self.name = name
        self.neighbors = {}

    def add_neighbor(self, neighbor, weight):
        self.neighbors[neighbor] = weight

    def get_weight(neighbor):
        return self.neighbors.get(neighbor, None)


class Graph:
    def __init__(self):
        self.vertices = {}

    def add_node(self, name):
        self.vertices[name] = Vertex(name)

    def add_edge(self, a, b, weight):
        if a not in self.vertices:
            self.add_node(a)
        if b not in self.vertices:
            self.add_node(b)
        self.vertices[a].add_neighbor(b, weight)
        self.vertices[b].add_neighbor(a, weight)

    def get_neighbors(self, node):
        if node in self.vertices:
            return self.vertices[node].neighbors
        else:
            return []

    def get_weight(self, a, b):
        if a in self.vertices:
            return self.vertices[a].get_weight(b)
        else:
            return None


def bfs(g, start):
    q = Queue()
    q.put(start)
    v_set = set()
    while not q.empty():
        node = q.get()
        if node not in v_set:
            print node
            v_set.add(node)
            for neighbor in g.get_neighbors(node):
                q.put(neighbor)


def dfs(g, start):
    q = LifoQueue()
    q.put(start)
    v_set = set()
    while not q.empty():
        node = q.get()
        if node not in v_set:
            print node
            v_set.add(node)
            for neighbor in g.get_neighbors(node):
                q.put(neighbor)


def build_graph():
    g = Graph()
    g.add_edge('sunset', 'richmond', 4)
    g.add_edge('presidio', 'richmond', 1)
    g.add_edge('pac heights', 'richmond', 8)
    g.add_edge('western addition', 'richmond', 7)
    g.add_edge('western addition', 'pac heights', 2)
    g.add_edge('western addition', 'downtown', 3)
    g.add_edge('western addition', 'haight', 4)
    g.add_edge('mission', 'haight', 1)
    g.add_edge('mission', 'soma', 5)
    g.add_edge('downtown', 'soma', 5)
    g.add_edge('downtown', 'nob hill', 2)
    g.add_edge('marina', 'pac heights', 2)
    g.add_edge('marina', 'presidio', 4)
    g.add_edge('marina', 'russian hill', 3)
    g.add_edge('nob hill', 'russian hill', 1)
    g.add_edge('north beach', 'russian hill', 1)
    return g


if __name__ == '__main__':
    g = build_graph()

    print "--------------------------"
    print "BFS starting at Mission:"
    print "--------------------------"
    bfs(g, 'mission')
    print

    print "--------------------------"
    print "BFS starting at Pac Heights:"
    print "--------------------------"
    bfs(g, 'pac heights')
    print

    print "--------------------------"
    print "DFS starting at Mission:"
    print "--------------------------"
    dfs(g, 'mission')
    print

    print "--------------------------"
    print "DFS starting at Pac Heights:"
    print "--------------------------"
    dfs(g, 'pac heights')
    print




