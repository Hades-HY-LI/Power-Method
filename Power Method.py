import numpy as np
import matplotlib.pyplot as plt

def power_iteration(A, v0, eps = 1e-6, maxiter=100):
    """
    Please implement the function power_iteration that takes in the matrix X and initial vector v0 and returns the eigenvector.
    A: np.array (d, d)
    v0: np.array (d,)
    """
    t = 1
    c = np.array([v0]).T
    Tmax = np.arange(1, maxiter)
    for t in Tmax:
        c_inter = np.dot(A, c[:, t-1])
        c_inter = c_inter/np.linalg.norm(c_inter)
        c = np.column_stack((c, c_inter))
        if np.linalg.norm(c[:, t] - c[:, t - 1]) < eps:
            break
    return c[:, -1]



if __name__ == '__main__':

    np.random.seed(2022)
    E = np.random.normal(size=(10,10))
    v = np.array([1]+[0]*9)
    lams = np.arange(1, 11)
    prods = []
    for lam in lams:
        X = lam*np.outer(v,v) + E
        v0 = np.ones(10)
        v0 = v0/np.linalg.norm(v0,2)
        vv = power_iteration(X, v0)
        prods.append(np.abs(v @ vv))

    plt.plot(lams, prods, '-ok')
    plt.xlabel('lambda')
    plt.ylabel('product')
    plt.savefig('Q5.png')
    plt.show()


