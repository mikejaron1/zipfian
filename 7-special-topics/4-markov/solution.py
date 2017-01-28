import itertools
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

def step(t_matrix, probs):
    states = sorted(t_matrix.keys())
    transitions = itertools.permutations(states, 2)
    update = [[] for i in range(len(states))]

    for pair in transitions:
        frm = pair[0]
        to = pair[1]

        prior = probs[states.index(frm)]
        prob = t_matrix[frm][to]
        update[states.index(to)].append(prior * prob)

    reduction = [ sum(x) for x in update ]
    total_prob = sum(reduction)
    return [ float(r) / total_prob for r in reduction ]


def iterate(transition, initial):
    present = np.array(initial[:])

    future = np.array(step(transition, present))
    cnt = 0
    # solves for small round off errors and non-convergence
    while not np.allclose(future, present):
        present = future[:]
        future = np.array(step(transition, present))
        cnt+= 1

    return future

def eigen_decomposition(T):
    w, v = np.linalg.eig(T.T)

    # get the real component of principal eigenvector
    vector = np.real(v[:,0])

    # normalize
    return vector / np.sum(vector)

if __name__ == '__main__':
    
    T = { 'A' : { 'A':0 , 'B' : 1. , 'C' : 0,'D': 0, 'E' : 0 },
      'B' : { 'A': 1./2 , 'B' : 0 , 'C' : 1./2,'D': 0, 'E' : 0 },
      'C' : { 'A': 1./3 , 'B' : 1./3 , 'C' : 0, 'D': 0, 'E' : 1./3 },
      'D' : { 'A': 1. , 'B' : 0 , 'C' : 0, 'D': 0, 'E' : 0 },
      'E' : { 'A':0 , 'B' : 1./3 , 'C' : 1./3, 'D': 1./3, 'E' : 0 }
    }

    TM = np.array([[ 0 ,1., 0, 0, 0 ],
                  [ 1./2, 0, 1./2, 0, 0 ],
                  [ 1./3, 1./3, 0, 0, 1./3 ],
                  [ 1., 0, 0, 0, 0 ],
                  [ 0, 1./3, 1./3, 1./3, 0 ]])

    probs = [0.2, 0.2, 0.2, 0.2, 0.2]

    print "One step"
    print step(T, probs)

    print "\nIteratation"
    print iterate(T, probs)

    print "\nMax Likelihood state"
    print np.argmax(iterate(T, probs))

    print "\n\n~~~~~~~~~~~MATRICES~~~~~~~~~~~\n"
    
    probs = np.array(probs)

    print "One step"
    print probs.dot(TM)

    print "After one step most likely to be at"
    print np.argmax(probs.dot(TM))

    iter_l = []
    for i in range(10):
        probs = probs.dot(TM)
        iter_l.append(probs)

    print "Eigenvalue decomposition"
    print eigen_decomposition(TM)

    # PLOTS

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    plt.ioff()

    # FIRST 10 ITERATIONS
    fig = plt.figure(figsize=(12,50))

    for i,dist in enumerate(iter_l):
        ax = plt.subplot(10,1,i + 1)
        ax.bar(np.arange(len(dist)), dist)

    # FIRST 5 ITERATIONS IN 3D
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111, projection='3d')
    for i, tup in enumerate(zip(['r', 'g', 'b', 'y', 'c'], [0, 10, 20, 30, 40])):
        c, z = tup
        xs = np.arange(len(iter_l[i]))
        ys = iter_l[i]

        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [c] * len(xs)
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('State')
    ax.set_ylabel('Iteration')
    ax.set_zlabel('Probability')

    plt.show()
