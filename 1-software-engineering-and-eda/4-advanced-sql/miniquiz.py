from math import sqrt


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Triangle(object):
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def perimeter(self):
        return self.p1.distance(self.p2) + \
               self.p2.distance(self.p3) + \
               self.p3.distance(self.p1)

    def is_line(self):
        # three points are on the same line if the slope between p1 and p2 is
        # the same as the slow between p2 and p3
        return float(self.p1.y - self.p2.y) / (self.p1.x - self.p2.x) == \
               float(self.p2.y - self.p3.y) / (self.p2.x - self.p3.x)
