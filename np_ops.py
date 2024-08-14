import numpy as np


def cumcount(x):
    marker_idx = np.flatnonzero(np.diff(x)) + 1
    counts = np.diff(marker_idx, prepend=0, )
    counter = np.ones(len(x), dtype=int)
    counter[marker_idx] -= counts
    return np.add.accumulate(counter)


def np_onehot(indices, amax):
    n = indices.shape[0]

    result = np.zeros((n, amax))
    result[(np.arange(n), indices)] = 1

    return result


def put_masked_rows(row_mask, features):
    shape = (row_mask.shape[0], features.shape[1])
    result = np.zeros(shape, dtype=features.dtype)
    result[row_mask, :] = features
    return result