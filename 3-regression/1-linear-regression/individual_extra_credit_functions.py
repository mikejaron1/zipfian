import numpy as np

def matrix_multiply(A, B):
    if A.shape[1] != B.shape[0]:
        return None
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in xrange(A.shape[0]):
        for j in xrange(B.shape[1]):
            result[i, j] = sum(A[i, k] * B[k, j] for k in xrange(A.shape[1]))
    return result


def transpose(A):
    result = np.zeros((A.shape[1], A.shape[0]))
    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            result[j, i] = A[i, j]
    return result


def reshape(A, size):
    total = A.shape[0] * A.shape[1]
    if total != size[0] * size[1]:
        return None
    result = np.zeros(size)
    for k in xrange(total):
        result[k / size[1], k % size[1]] = A[k / A.shape[1], k % A.shape[1]]
    return result


def elementwise_multiply(A, B):
    if A.shape != B.shape:
        return None
    result = np.zeros(A.shape)
    for i in xrange(result.shape[0]):
        for j in xrange(result.shape[1]):
            result[i, j] = A[i, j] * B[i, j]
    return result
