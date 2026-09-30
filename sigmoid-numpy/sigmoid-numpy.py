import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # convert to a numpy array if it's not
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))