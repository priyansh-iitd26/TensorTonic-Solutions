import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    
    if len(y) == 0:
        return 0.0

    # counts of each unique class
    _, counts = np.unique(y, return_counts=True)

    # probabilities of each class
    probabilities = counts / len(y)

    # entropy
    return -np.sum(probabilities * np.log2(probabilities))