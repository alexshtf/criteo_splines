import polars as pl
import numpy as np
import lz4.frame

from column_transforms import (
    transform_label,
    IntTransformer,
    CatTransformer,
    SplineTransformer,
    ColTransformer)
from config import (
    int_cols, cat_cols, spline_cols, columns,
    split_fractions,
    transform_config,
    out_train_path, out_val_path, out_test_path,
    degrees)
from preprocess_ops import split_df, label_path, cat_col_path, int_col_path, spline_col_path, save_tensor


def load_dataset(data_file):
    print(f'Loading data from {data_file}')
    with lz4.frame.open(data_file, 'rb') as f:
        df = pl.read_csv(f, separator='\t', has_header=False, new_columns=columns)

    print(f'Splitting into train, validation, and test')
    split_sizes = np.rint(np.array(split_fractions) * len(df)).astype(np.int32)
    parts = []
    offset = 0
    for sz in split_sizes:
        part = df.slice(offset, sz)
        offset += sz
        parts.append(part)    
    return tuple(parts)


train, val, test = load_dataset('train.txt.lz4')
# train, val, test = load_dataset('train_trunc.txt.lz4')
print(f'Split lengths = {(len(train), len(val), len(test))}')

tr_label = transform_label(train)
val_label = transform_label(val)
test_label = transform_label(test)

save_tensor(label_path(out_train_path), tr_label)
save_tensor(label_path(out_val_path), val_label)
save_tensor(label_path(out_test_path), test_label)

print('Using config = ', str(transform_config))

print('Transforming spline columsn with config ', spline_cols)
for col_name in spline_cols:
    for degree in degrees:
        print(f'Transforming SPLINE column {col_name} of degree {degree}')
        tr = ColTransformer(col_name, SplineTransformer(degree=degree, **transform_config))
        tr.fit(train)
        save_tensor(spline_col_path(out_train_path, col_name, degree), tr.transform(train))
        save_tensor(spline_col_path(out_val_path, col_name, degree), tr.transform(val))
        save_tensor(spline_col_path(out_test_path, col_name, degree), tr.transform(test))

print('Transforming integer columns')
for col_name in int_cols:
    print(f'Transforming INTEGER column {col_name}')
    tr = ColTransformer(col_name, IntTransformer(**transform_config))
    tr.fit(train)
    save_tensor(int_col_path(out_train_path, col_name), tr.transform(train))
    save_tensor(int_col_path(out_val_path, col_name), tr.transform(val))
    save_tensor(int_col_path(out_test_path, col_name), tr.transform(test))

print('Transforming categorical columns')
for col_name in cat_cols:
    print(f'Transforming CATEGORICAL column {col_name}')
    tr = ColTransformer(col_name, CatTransformer(**transform_config))
    tr.fit(train)
    save_tensor(cat_col_path(out_train_path, col_name), tr.transform(train))
    save_tensor(cat_col_path(out_val_path, col_name), tr.transform(val))
    save_tensor(cat_col_path(out_test_path, col_name), tr.transform(test))
