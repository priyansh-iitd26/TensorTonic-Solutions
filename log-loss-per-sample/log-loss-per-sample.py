import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    # clipping every prediction to [eps, 1-eps]
    y_pred = [min(1-eps, max(eps, y_hat)) for y_hat in y_pred]

    log_losses = []

    for y, p in zip(y_true, y_pred):
        log_losses.append(-(y * math.log(p) + (1-y) * math.log(1-p)))

    return log_losses