import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    # using numpy
    # y = np.asarray(x)
    # y = np.where(y > 0, y, alpha * (np.exp(y) - 1.0))
    # return y.tolist()

    # without using numpy
    return [v if v > 0 else alpha * (math.exp(v) - 1) for v in x]