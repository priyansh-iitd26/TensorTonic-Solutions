import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # converting input lists to numpy arrays
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    # boolean indexing
    # ignoring all instances where p = 0
    mask = p_arr > 0
    
    p_arr = p_arr[mask]
    q_arr = q_arr[mask]

    # q_i = max(q_i, eps)
    q_arr = np.where(q_arr > eps, q_arr, eps)

    # computing kl divergence
    kl_divergence_value = float(np.sum(p_arr * np.log(p_arr / q_arr)))
    return kl_divergence_value