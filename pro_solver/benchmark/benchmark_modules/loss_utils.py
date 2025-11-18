import numpy as np
import math

def rmse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x.shape) != (len(y.shape)):
        raise AssertionError("missmatch shapes")
    N = math.prod([num for num in x.shape])
    delta = (1 / N) * (np.sum((x - y) ** 2, axis=-1))
    while len(delta.shape) != 1:
        delta = np.sum(delta, axis=-1)
    return delta ** 0.5

def rel_rmse(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Relative RMSE = RMSE(x, target) / RMSE(target, 0)
    """
    err = rmse(x, target)
    baseline = rmse(target, np.zeros_like(target))

    # защита от деления на 0
    if np.any(baseline == 0):
        raise ZeroDivisionError("Target norm is zero → relative error undefined")

    return err / baseline