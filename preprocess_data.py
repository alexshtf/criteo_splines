import pandas as pd
import numpy as np
import lz4.frame

from column_transforms import (
    transform_label,
    IntColTransformer,
    CatColTransformer,
    SplineColTransformer)
from config import (
    int_cols, cat_cols, columns,
    split_fractions,
    int_transform_config, spline_transform_config, cat_transform_config,
    out_train_path, out_val_path, out_test_path,
    knot_counts,
    degrees)
from preprocess_ops import split_df, label_path, cat_col_path, int_col_path, spline_col_path, save_tensor


def load_dataset(data_file):
    print(f'Loading data from {data_file}')
    dtypes = {'label': np.int8} | \
             {col: pd.Int32Dtype() for col in int_cols} | \
             {col: str for col in cat_cols}
    with lz4.frame.open(data_file, 'rb') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=columns, dtype=dtypes, engine='pyarrow')

    print(f'Splitting into train, validation, and test')
    return split_df(df, split_fractions)


train, val, test = load_dataset('train.txt.lz4')
# train, val, test = load_dataset('train_trunc.txt.lz4')
print(f'Split lengths = {(len(train), len(val), len(test))}')

tr_label = transform_label(train)
val_label = transform_label(val)
test_label = transform_label(test)

save_tensor(label_path(out_train_path), tr_label)
save_tensor(label_path(out_val_path), val_label)
save_tensor(label_path(out_test_path), test_label)

print('Transforming spline columsn with config ', spline_transform_config)
for col_name in spline_transform_config['configs']:
    for degree in degrees:
        print(f'Transforming SPLINE column {col_name} of degree {degree}')
        tr = SplineColTransformer(col_name, config=spline_transform_config, degree=degree)
        tr.fit(train)
        save_tensor(spline_col_path(out_train_path, col_name, degree), tr.transform(train))
        save_tensor(spline_col_path(out_val_path, col_name, degree), tr.transform(val))
        save_tensor(spline_col_path(out_test_path, col_name, degree), tr.transform(test))

print('Transforming integer columns with config: ', int_transform_config)
for col_name in int_cols:
    print(f'Transforming INTEGER column {col_name}')
    tr = IntColTransformer(col_name, **int_transform_config)
    tr.fit(train)
    save_tensor(int_col_path(out_train_path, col_name), tr.transform(train))
    save_tensor(int_col_path(out_val_path, col_name), tr.transform(val))
    save_tensor(int_col_path(out_test_path, col_name), tr.transform(test))

print('Transforming categorical columns with config ', cat_transform_config)
for col_name in cat_cols:
    print(f'Transforming CATEGORICAL column {col_name}')
    tr = CatColTransformer(col_name, **cat_transform_config)
    tr.fit(train)
    save_tensor(cat_col_path(out_train_path, col_name), tr.transform(train))
    save_tensor(cat_col_path(out_val_path, col_name), tr.transform(val))
    save_tensor(cat_col_path(out_test_path, col_name), tr.transform(test))
