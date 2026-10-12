import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # converting python lists to numpy arrays
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    hinge_losses = np.maximum(0, margin - (y_true_arr * y_score_arr))

    if reduction == "mean":
        return float(np.mean(hinge_losses))

    return float(np.sum(hinge_losses))