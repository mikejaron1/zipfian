from Queue import Queue, LifoQueue


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
    def __str__(self):
        return self.name


def bfs(root):
    q = Queue()
    q.put(root)
    while not q.empty():
        node = q.get()
        print node
        for child in node.children:
            q.put(child)


def dfs(root):
    stack = LifoQueue()
    stack.put(root)
    while not stack.empty():
        node = stack.get()
        print node
        for child in node.children:
            stack.put(child)


def dfs_recursive(root):
    if root:
        print root
        for child in root.children:
            dfs_recursive(child)


if __name__ == '__main__':
    root = Node("USA")
    cali = Node("California")
    sf = Node("San Francisco")
    sanjose = Node("San Jose")
    la = Node("Los Angeles")
    ny = Node("New York")
    nyc = Node("New York City")
    albany = Node("Albany")

    root.children = [cali, ny]
    cali.children = [sf, sanjose, la]
    ny.children = [nyc, albany]

    print "--------------"
    print "BFS:"
    print "--------------"
    bfs(root)
    print

    print "--------------"
    print "DFS:"
    print "--------------"
    dfs(root)
    print

    print "--------------"
    print "DFS recursive:"
    print "--------------"
    dfs_recursive(root)