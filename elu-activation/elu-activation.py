import math
import numpy as np

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """

    y = np.asarray(x)

    y = np.where(y > 0, y, alpha * (np.exp(y) - 1.0))

    return y.tolist()