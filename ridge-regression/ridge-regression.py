import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """

    X = np.array(X).reshape(len(X), len(X[0]))
    y = np.array(y)
    
    I = np.identity(X.shape[1])
    # I[0][0] = 0
    
    w = np.dot(np.dot(np.linalg.inv((np.dot(X.T, X) + lam * I)), X.T), y)
    return w