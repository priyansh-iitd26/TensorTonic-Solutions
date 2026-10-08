import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    
    mat = np.array(A)
    N, M = mat.shape

    mat_transpose = np.zeros((M, N))

    # transpose is (i,j) to (j,i) mapping
    
    for i in range(N):
        for j in range(M):
            mat_transpose[j][i] = mat[i][j]

    return mat_transpose