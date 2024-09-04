from collections import defaultdict

import numpy as np
from scipy.interpolate import BSpline


# evaluates the B-Spline basis on [0, 1] a given number of knots and a given degree on all values in a given vector.
class BSplineTransformer:
    def __init__(self, knots, degree):
        kv = np.array([0.] * degree + list(np.linspace(0, 1, knots)) + [1.] * degree)
        self.degree = degree
        self.basis = [
            BSpline.basis_element(kv[i:(i+degree+2)], extrapolate=False)
            for i in range(knots + degree - 1)
        ]

    def basis_size(self):
        return len(self.basis)

    def basis_support(self):
        return self.degree + 1

    def __call__(self, x):
        arr = np.array([be(x) for be in self.basis])
        start_idx = np.argmax(~np.isnan(arr), axis=0)
        full_idx = np.linspace(start_idx, start_idx + self.degree, self.degree + 1, dtype=np.int32)

        # Use broadcasting to set the elements of the matrix to the corresponding
        # elements of the input matrix at the specified indices.
        weights = arr[full_idx, np.arange(np.size(x))]

        return full_idx.T, weights.T


# transforms a dataset that is given as a matrix of numerical values in [0, 1] into
# an index matrix, a weight matrix, an offset matrix and a field matrix, where the weights
# are the weights of the B-Spline basis functions at the values of the input matrix.
def spline_transform_dataset(numerical_zero_one, spline_transformer, special_values=None):
    field_cols = []
    index_cols = []
    weight_cols = []
    offset_cols = []
    if special_values is None:
        special_values = {}

    curr_off = 0
    idx_offset = 0.
    for col_idx in range(numerical_zero_one.shape[1]):
        col_special_values = special_values.get(col_idx, [])
        col = numerical_zero_one[:, col_idx]
        special_mask = np.isin(col, col_special_values)
        regular_mask = ~special_mask

        idx = np.zeros((col.shape[0], spline_transformer.basis_support()), dtype=np.int32)
        weights = np.zeros((col.shape[0], spline_transformer.basis_support()), dtype=np.float32)

        idx[regular_mask, :], weights[regular_mask, :] = spline_transformer(col[regular_mask])
        special_offset = spline_transformer.basis_size()
        for sval in col_special_values:
            idx[special_mask & (col == sval), 0] = special_offset
            weights[special_mask & (col == sval), 0] = 1.
            special_offset += 1

        index_cols.append(idx + idx_offset)
        weight_cols.append(weights)

        curr_off += idx.shape[1]
        idx_offset += spline_transformer.basis_size() + len(col_special_values)

        offset_cols.append(curr_off * np.ones(col.shape, dtype=np.int32))
        field_cols.append(col_idx * np.ones(col.shape, dtype=np.int32))

    indices = np.concatenate(index_cols, axis=1)
    weights = np.concatenate(weight_cols, axis=1)
    fields = np.column_stack(field_cols)
    offsets = np.column_stack(offset_cols)

    return indices, weights, offsets - 1, fields
