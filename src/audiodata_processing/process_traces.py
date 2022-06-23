import numpy as np


def rgb2gray(rgb):
    """ Transforms RGB image into grayscale """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def threshold_le(trace, th):
    """Sets im to be 0 at points where it is <= th"""
    new_trace = np.copy(trace)
    mask = trace <= th
    new_trace[mask] = 0
    return new_trace


def clear_trace(trace, th=0.3):
    """ Tries to delete incorrectly traced parts leaving only the main harmonic"""
    trace = threshold_le(rgb2gray(trace), th)
    for i in range(trace.shape[1]):
        inds = np.where(trace[:, i] > 0)[0]
        if len(inds) ==0:
            continue
        max_ind = max(inds)
        inds_bad = [ind for ind in inds if max_ind - ind > 10]
        trace[inds_bad, i] = 0
    return trace
