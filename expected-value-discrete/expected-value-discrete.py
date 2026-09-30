import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    sum = 0.0
    
    for i in range(len(x)):
        sum = sum + (x[i] * p[i])
    
    return float(sum)