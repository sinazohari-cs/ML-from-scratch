import numpy as np

def generate_data(n=200):
    np.random.seed(0)

    # Class 0 (left cluster)
    x0 = np.random.randn(n//2, 2) + np.array([-2, -2])

    # Class 1 (right cluster)
    x1 = np.random.randn(n//2, 2) + np.array([2, 2])

    X = np.vstack((x0, x1))
    y = np.array([0]*(n//2) + [1]*(n//2))

    return X, y

