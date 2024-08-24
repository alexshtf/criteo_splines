import numpy as np
import polars as pl
from scipy.interpolate import BSpline
import scipy.special
from np_ops import cumcount, np_onehot, put_masked_rows
import torch
import math
import itertools


def transform_label(df):
    label = df['label'].to_numpy(writable=True).reshape(-1, 1)
    return torch.as_tensor(label).float()


class IntColumnScaler:
    def __init__(self, at_least=0):
        self.at_least = at_least
        
    def fit(self, col):
        n = len(col)
        mean = np.mean(col)
        sum_squares = np.sum(np.square(col))
        self.lomax_shape_ =  (2 * n * mean * mean) / (sum_squares) + 2
        self.lomax_scale_ = mean * (self.lomax_shape_ - 1)
        return self

    def transform(self, col):
        return 1 - np.power(1 + col / self.lomax_scale_, -self.lomax_shape_)


class SplineTransformer:
    def __init__(self, at_least=0, degree=3, n_knots=50, **kwargs):
        self.at_least = 0
        self.scaler = IntColumnScaler(at_least)
        self.degree = degree
        self.knots = np.pad(np.linspace(0, 1, n_knots), degree, 'edge')
    
    def get_discrete_(self, col):
        return col.filter((col < self.at_least) | (col.is_null()))
    
    def get_continuous_mask_(self, col):
        return (col >= self.at_least).fill_null(False)

    def get_continuous_(self, col):
        return col.filter(col >= self.at_least)
    
    def fit(self, col):
        self.discrete_keys_ = self.get_discrete_(col).extend_constant(None, 1).unique().to_list()
        self.discrete_values_ = list(range(len(self.discrete_keys_)))
        self.n_discrete_ = len(self.discrete_keys_)
        self.scaler.fit(self.get_continuous_(col).to_numpy())  
        return self
    
    def transform(self, col):
        discrete = self.get_discrete_(col).replace_strict(
            old=self.discrete_keys_, 
            new=self.discrete_values_,
            default=0)
        discrete = torch.as_tensor(discrete.to_numpy(writable=True)).to(torch.int32)
        
        scaled = self.scaler.transform(self.get_continuous_(col).to_numpy())
        spline_basis = BSpline.design_matrix(scaled, self.knots, self.degree) \
            .astype(np.float32)
        spline_indices = torch.tensor(spline_basis.indices.reshape(-1, 1 + self.degree))
        spline_weights = torch.tensor(spline_basis.data.reshape(-1, 1 + self.degree))
        
        indices = torch.zeros(len(col), spline_weights.shape[1], dtype=torch.int32)
        weights = torch.zeros(len(col), spline_weights.shape[1], dtype=torch.float32)
        continuous_mask = torch.as_tensor(self.get_continuous_mask_(col).to_numpy(writable=True))
        
        indices.masked_scatter_(continuous_mask[:, None], spline_indices + self.n_discrete_)
        weights.masked_scatter_(continuous_mask[:, None], spline_weights)
        indices.select(-1, 0).masked_scatter_(~continuous_mask, discrete)
        weights.select(-1, 0).masked_fill_(~continuous_mask, 1.)
        
        return indices, weights, (len(self.discrete_keys_) + len(self.knots) - self.degree + 1)    


class IntTransformer:
    def __init__(self, **kwargs):
        pass
    
    def fit(self, col):
        max_val = col.max()
        self.max_bin_ = int(math.floor(math.log(max_val) ** 2))
        self.min_ = col.min()
        return self
    
    def transform(self, col):
        pos_mask = (col > 0).fill_null(False)
        neg_mask = (col <= 0).fill_null(False)
        
        pos = col.filter(pos_mask).to_numpy()
        pos_bins = np.floor(np.square(np.log(pos))).astype(np.int32)
        pos_bins = 1 + np.clip(pos_bins, a_min=None, a_max=self.max_bin_) - self.min_
        
        neg = col.filter(neg_mask).to_numpy().astype(np.int32)
        neg_bins = 1 + np.clip(neg, a_min=self.min_, a_max=None) - self.min_
        
        pos_mask = torch.as_tensor(pos_mask.to_numpy())
        neg_mask = torch.as_tensor(neg_mask.to_numpy())
        pos_bins = torch.as_tensor(pos_bins)
        neg_bins = torch.as_tensor(neg_bins)
        
        indices = torch.zeros(len(col), dtype=torch.int32)
        indices.masked_scatter_(pos_mask, pos_bins)
        indices.masked_scatter_(neg_mask, neg_bins)
        indices = indices.reshape(-1, 1)
        weights = torch.ones(len(col), 1, dtype=torch.float32)

        return indices, weights, (2 + self.max_bin_ - self.min_)


class CatTransformer:
    infrequent = 'INFREQUENT'
    na = 'MISSING'

    def __init__(self, min_frequency=10, **kwargs):
        self.min_frequency = min_frequency

    def fit(self, col):
        col = col.fill_null(self.na)
        vcs = col.value_counts()
        frequent = vcs.filter(pl.col('count') >= self.min_frequency)[col.name]
        self.frequent_cats_ = list(set(frequent) | {self.infrequent, self.na})
        self.frequent_cat_idx_ = list(range(len(self.frequent_cats_)))
        return self

    def transform(self, col):
        col = col.fill_null(self.na)
        col[~col.is_in(self.frequent_cats_)] = self.infrequent
        ordinal = col.replace_strict(self.frequent_cats_, self.frequent_cat_idx_)
        
        indices = torch.as_tensor(ordinal.to_numpy().astype(np.int32)).reshape(-1, 1)
        weights = torch.ones(len(col), 1, dtype=torch.float32)
        cnt = len(self.frequent_cats_)
        return indices, weights, cnt


class ColTransformer:
    def __init__(self, col_name, inner):
        self.col_name = col_name
        self.inner = inner
    
    def fit(self, df):
        self.inner.fit(df[self.col_name])
        return self
    
    def transform(self, df):
        return self.inner.transform(df[self.col_name])