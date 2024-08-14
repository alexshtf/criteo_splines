import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import scipy.interpolate
from np_ops import cumcount, np_onehot, put_masked_rows
import torch


def ints_to_unit_interval(x, discrete=False, exponent=2, scale=1, ub=None, **config):
    tr = scale * (np.log(x) ** exponent)
    if discrete:
        tr = np.floor(tr).astype(np.int32)

    if ub is not None:
        tr = np.clip(tr, None, ub) / ub

    return tr


def transform_label(df):
    label = np.atleast_2d(df['label']).T
    return torch.as_tensor(label).float()


class IntColTransformer:
    def __init__(self, col_name, pretransform_fn=ints_to_unit_interval, at_least=1, **config):
        self.col_name = col_name
        self.pretransform_fn = pretransform_fn
        self.at_least = at_least
        self.config = config

    def fit(self, df):
        col_name = self.col_name
        pretransform_fn = self.pretransform_fn
        at_least = self.at_least

        col = df[col_name]
        self.min_val = col.min()
        self.max_val = at_least + pretransform_fn(col.max(), discrete=True, **self.config)

    def transform(self, df):
        col_name = self.col_name
        pretransform_fn = self.pretransform_fn
        at_least = self.at_least

        min_val = self.min_val
        max_val = self.max_val

        col = df[col_name]

        min_int = np.iinfo(np.int32).min
        max_int = np.iinfo(np.int32).max
        tr_mask = col.fillna(min_int).astype(np.int32) > 1 + at_least
        special_mask = col.fillna(max_int).astype(np.int32) <= 1 + at_least

        transformed = at_least + pretransform_fn(col[tr_mask], discrete=True, **self.config)

        index = np.zeros((len(col), 1), dtype=np.int32)
        index[special_mask, 0] = 1 + col[special_mask] - min_val
        index[tr_mask, 0] = 1 + np.clip(transformed, None, max_val) - min_val

        index = np.atleast_2d(index).T
        return torch.as_tensor(index), (2 + max_val - min_val)


class CatColTransformer:
    infrequent = 'INFREQUENT'
    na = 'MISSING'

    def __init__(self, col_name, min_frequency, **config):
        self.col_name = col_name
        self.min_frequency = min_frequency
        self.frequent_cats = None
        self.mapping = None

    def fit(self, df):
        col = df[self.col_name].fillna(self.na)
        vcs = col.value_counts()
        self.frequent_cats = set(vcs[vcs >= self.min_frequency].index) | {self.infrequent, self.na}
        self.mapping = dict(zip(self.frequent_cats, range(len(self.frequent_cats))))

    def transform(self, df):
        col = df[self.col_name].fillna(self.na)
        col[~col.isin(self.frequent_cats)] = self.infrequent
        tr = col.map(self.mapping).astype(np.int32)
        tr = torch.as_tensor(tr.values).unsqueeze(1)
        return tr, len(self.mapping)


def compute_knot_sequence(max_knot):
    knot_count = int(np.ceil(np.log1p(max_knot) ** 2))
    knot_idx = np.arange(0, knot_count)
    return (np.exp(np.sqrt(knot_idx)) - 1)


class SplineColTransformer:
    def __init__(self, col_name, config, degree):
        self.col_name = col_name
        self.col_config = config['default'] | config['configs'][col_name]
        self.degree = degree

    def fit(self, df):
        at_least = self.col_config['at_least']
        min_int = np.iinfo(np.int32).min
        col = df[self.col_name]
        self.at_least = at_least
        self.col_min = np.min(col)
        self.col_max = np.max(col)
        self.knots = np.pad(compute_knot_sequence(self.col_max - at_least), (self.degree, self.degree), 'edge')

    def compute_masks(self, col):
        min_int = np.iinfo(np.int32).min
        max_int = np.iinfo(np.int32).max

        col = col.copy()
        col[col < self.col_min] = pd.NA

        spline_mask = col.fillna(min_int).astype(np.int32) >= self.at_least
        special_mask = col.fillna(max_int).astype(np.int32) < self.at_least
        na_feature = np.atleast_2d(col.isna()).astype(np.float64).T

        return spline_mask, special_mask, na_feature

    def compute_spline_features(self, col, spline_mask):
        values = col[spline_mask].values.astype(np.float32) - self.at_least
        values = np.clip(values, np.min(self.knots), np.max(self.knots) - 1)
        spline_features = scipy.interpolate.BSpline.design_matrix(values, self.knots, self.degree).todense()
        spline_features = put_masked_rows(spline_mask, spline_features)
        return spline_features, len(self.knots) - 1 - self.degree

    def compute_special_features(self, col, special_mask):
        specials = np.atleast_2d(col[special_mask]).T
        special_features = np_onehot(col[special_mask].astype(np.int32) - self.col_min, self.at_least - self.col_min)
        special_features = put_masked_rows(special_mask, special_features)
        return special_features

    def transform(self, df):
        col = df[self.col_name]
        spline_mask, special_mask, na_feature = self.compute_masks(col)
        spline_features, n_spline_features = self.compute_spline_features(col, spline_mask)
        special_features = self.compute_special_features(col, special_mask)

        all_features = np.hstack([special_features, spline_features, na_feature])
        indices, weights = SplineColTransformer.sparsify_features(all_features)
        print(indices.shape, weights.shape)
        return (
            torch.as_tensor(indices),
            torch.as_tensor(weights).to(dtype=torch.float32),
            1 + self.at_least - self.col_min + n_spline_features
        )

    @staticmethod
    def sparsify_features(features):
        nnz_rows, nnz_cols = np.nonzero(features)
        new_cols = cumcount(nnz_rows) - 1
        max_col = np.max(new_cols)

        indices = np.zeros((features.shape[0], 1 + max_col), dtype=np.int32)
        indices[(nnz_rows, new_cols)] = nnz_cols

        weights = np.zeros((features.shape[0], 1 + max_col), dtype=features.dtype)
        weights[(nnz_rows, new_cols)] = features[(nnz_rows, nnz_cols)]

        return indices, weights
