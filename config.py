transform_config = {
    'at_least': 0,
    'min_frequency': 10,
    'n_knots': 30
}

int_cols = [f'I_{1 + i}' for i in range(13)]
spline_cols = ['I_2', 'I_5', 'I_6', 'I_9']
cat_cols = [f'C_{1 + i}' for i in range(26)]
columns = ['label'] + int_cols + cat_cols
degrees = [0, 3]

split_fractions = [0.7, 0.15, 0.15]
out_prefix = '/home/ashtoff/ephemeral_drive/criteo_preprocess'
out_train_path = f'{out_prefix}tr'
out_val_path = f'{out_prefix}val'
out_test_path = f'{out_prefix}test'