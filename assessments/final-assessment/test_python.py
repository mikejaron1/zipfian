import nose as n
from submissions import problems as p
import numpy as np


def test_column_averages():
    result = p.column_averages('data/example.csv')
    answer = {'A': 0.35999999999999999, 'B': 1330.8, 'C': 64.200000000000003}
    n.tools.assert_equal(result, answer)


def test_filter_by_class1():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array(["a", "c", "a", "b"])
    result = p.filter_by_class(X, y, "a")
    answer = np.array([[1, 2, 3], [7, 8, 9]])
    n.tools.assert_true(np.array_equal(result, answer))


def test_filter_by_class2():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array(["a", "c", "a", "b"])
    result = p.filter_by_class(X, y, "b")
    answer = np.array([[10, 11, 12]])
    n.tools.assert_true(np.array_equal(result, answer))


def test_etsy_query():
    result = p.etsy_query("data")
    n.tools.assert_true(isinstance(result, list))
    n.tools.assert_equal(len(result), 42)
    n.tools.assert_in('UnicornTees', result)
    n.tools.assert_in('PoppinPosters', result)


if __name__ == '__main__':
    n.run()