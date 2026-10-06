import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n_samples, n_features = X.shape
    
    # initialize of weights and bias
    weights = np.zeros(n_features)
    bias = 0.0

    # gradient descent
    for _ in range(steps):
        # calculate sigmoid
        z = np.dot(X, weights) + bias
        y_hat = _sigmoid(z)
        
        # compute gradients based on derived formulas
        dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
        db = (1 / n_samples) * np.sum(y_hat - y)

        # update weights and bias
        weights -= lr * dw
        bias -= lr * db

    return (weights, bias)
    