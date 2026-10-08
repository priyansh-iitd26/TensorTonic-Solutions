import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Returns elementwise Leaky ReLU values as a NumPy array matching the input shape.
    """

    # np.asarray safely converts both single floats and lists into arrays
    x_arr = np.asarray(x)
    
    # np.where applies the conditional logic element-wise instantly
    return np.where(x_arr >= 0, x_arr, alpha * x_arr)