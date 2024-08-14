import numpy as np
import os
import gzip
import torch
import lz4.frame


def split_df(df, fractions):
    assert sum(fractions) == 1

    fractions = np.asarray(fractions) * len(df)
    fractions = np.cumsum(fractions[:-1].astype(np.int32))

    permutation = np.random.default_rng(seed=42).permutation(len(df))
    df = df.iloc[permutation, :]
    return tuple(np.split(df, fractions))


def label_path(prefix):
    return prefix + '/label.pth.lz4'


def int_col_path(prefix, col_name):
    return f'{prefix}/{col_name}/int.pth.lz4'


def cat_col_path(prefix, col_name):
    return f'{prefix}/{col_name}/cat.pth.lz4'


def spline_col_path(prefix, col_name, degree):
    return f'{prefix}/{col_name}/spline_{degree}.pth.lz4'


def save_tensor(path, data):
    print(f'Saving tensor to {path}')

    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)

    with lz4.frame.open(path, 'wb', compression_level=6) as f:
        torch.save(data, f)


def load_tensor(path):
    with lz4.frame.open(path, 'rb') as f:
        return torch.load(f)