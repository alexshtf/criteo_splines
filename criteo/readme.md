# Criteo dataset with splines

To reproduce the results in the paper, the following steps need to be done:
### Data preparation
1. Edit the `config.py` file, and modify the `out_prefix` variable to point to the path where the data preprocessing results will be stored
2. Download the Criteo dataset files, and compress the `train.txt` file using lz4:
    ```bash
    lz4 -9 -z train.txt train.txt.lz4
    ```
3. Run the data preprocessing script:
    ```bash
    python preprocess_data.py
    ```
   This script will split the data into train, validation, and test sets, and run the preprocessing logic described in the paper for all three datasets.

### Training
The `train.py` script will train a binned or a spline model on the dataset. The `--degrees` argument specifies the
spline degrees to try (degree 0 is binning), the `--emb_dims` argument specifies the embedding dimensions to try,
and the ``study_name`` tells what is the Optuna study name. The name of
the log file is based on this argument, and later used by the analysis notebook, described below.

To reproduce the results from the paper, run the following commands:
```bash
python3 train.py --degrees 0 --emb_dims 8 --study_name criteo_0_8
python3 train.py --degrees 2 --emb_dims 8 --study_name criteo_2_8
python3 train.py --degrees 0 --emb_dims 10 --study_name criteo_0_10
python3 train.py --degrees 2 --emb_dims 10 --study_name criteo_2_10
python3 train.py --degrees 0 --emb_dims 12 --study_name criteo_0_12
python3 train.py --degrees 2 --emb_dims 12 --study_name criteo_2_12
python3 train.py --degrees 0 --emb_dims 14 --study_name criteo_0_14
python3 train.py --degrees 2 --emb_dims 14 --study_name criteo_2_14
python3 train.py --degrees 0 --emb_dims 16 --study_name criteo_0_16
python3 train.py --degrees 2 --emb_dims 16 --study_name criteo_2_16
python3 train.py --degrees 0 --emb_dims 18 --study_name criteo_0_18
python3 train.py --degrees 2 --emb_dims 18 --study_name criteo_2_18
```
It will take a few **days** to train each of them. If you want to train using multiple CUDA GPUs, you can add
the `--gpu N` argument, where N is the CUDA device index. For example, on a machine with 4 GPUs we 
may want to run the first four expeirments as:
```bash
python3 train.py --gpu 0 --degrees 0 --emb_dims 8 --study_name criteo_0_8
python3 train.py --gpu 1 --degrees 2 --emb_dims 8 --study_name criteo_2_8
python3 train.py --gpu 2 --degrees 0 --emb_dims 10 --study_name criteo_0_10
python3 train.py --gpu 3 --degrees 2 --emb_dims 10 --study_name criteo_2_10
```


### Analysis
Open the `studies.ipynb` notebook, and run it. You will obtain a table with the test errors for both 
bins and splines, for each embedding dime sion.