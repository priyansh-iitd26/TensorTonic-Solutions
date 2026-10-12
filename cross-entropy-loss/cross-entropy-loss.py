import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # converting inputs to numpy arrays
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    
    N = len(y_true_arr)

    # advanced integer indexing on numpy arrays
    # np.arange(N) generates row indices
    # y_true_arr gives column index for every row
    # y_pred_arr[row, col] accesses that probability value
    
    predicted_probs = y_pred_arr[np.arange(N), y_true_arr]
    losses = -np.log(predicted_probs)
    
    cross_entropy_loss_value = np.mean(losses)
    return cross_entropy_loss_value