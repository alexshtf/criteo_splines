spline_transform_config = {
    'default': { 'at_least': 1 },
    'configs': {
        'I_2': { },
        'I_3': { },
        'I_5': { },
        'I_6': { },
        'I_9': { 'at_least': 3 },
        'I_13': { },
    }
}
spline_cols = spline_transform_config['configs'].keys()

int_transform_config = {
    'at_least': 1,
    'exponent': 2,
    'scale': 1
}

cat_transform_config = {
    'min_frequency': 10
}

knot_counts = list(range(10, 26))
degrees = list(range(1, 4))


int_cols = [f'I_{1 + i}' for i in range(13)]
cat_cols = [f'C_{1 + i}' for i in range(26)]
columns = ['label'] + int_cols + cat_cols

split_fractions = [0.6, 0.2, 0.2]
out_prefix = '<PATH_TO_PREPROCESSED_DATA>'
out_train_path = f'{out_prefix}tr'
out_val_path = f'{out_prefix}val'
out_test_path = f'{out_prefix}test'