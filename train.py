import argparse

import numpy as np
import optuna
import torch
from sklearn.model_selection import ParameterGrid
from torch import nn
from tqdm.auto import tqdm

from config import int_cols, spline_cols, cat_cols, out_train_path, out_val_path, out_test_path
from preprocess_ops import label_path, cat_col_path, int_col_path, spline_col_path, load_tensor
from train_ops import tqdm_len, BatchIter, FwFM, EarlyStopping


class CatColumnConfig:
    def __init__(self, name):
        self.name = name

    def load(self, path, device):
        return load_tensor(cat_col_path(path, self.name), device=device)

class IntColumnConfig:
    def __init__(self, name):
        self.name = name

    def load(self, path, device):
        return load_tensor(int_col_path(path, self.name), device=device)


class SplineColumnConfig:
    def __init__(self, name, degree):
        self.name = name
        self.degree = degree

    def load(self, path, device):
        return load_tensor(spline_col_path(path, self.name, self.degree), device=device)


def load_dataset(column_configs, path, device=None):
    if device is None:
        device = torch.device('cpu')

    tuples = [
        cfg.load(path, device)
        for cfg in tqdm(column_configs, desc='Loading feature columns')
    ]
    idxs = [idx for idx, weights, n in tuples]
    weights = [weights for idx, weights, n in tuples]
    n_features = [n for idx, weights, n in tuples]
    labels = load_tensor(label_path(path), device=device).float()
    tensors = idxs + weights + [labels]

    return tensors, n_features


def make_input(batch):
    n_fields = len(batch) // 2
    idxs = batch[:n_fields]
    weights = batch[n_fields:-1]
    labels = batch[-1].squeeze(-1)
    return idxs, weights, labels


def train_epoch(tensors, model, criterion, optim, batch_size, epoch_number=None):
    epoch_n = 0
    epoch_loss = 0.
    label_sum = 0.
    pred_sum = 0.
    with tqdm_len(BatchIter(*tensors, batch_size=batch_size)) as epoch_iter:
        for i, batch in enumerate(epoch_iter, start=1):
            idxs, weights, label = make_input(batch)
            pred = model(idxs, weights)
            loss = criterion(pred, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                n_batch = label.numel()
                epoch_n += n_batch
                epoch_loss += loss * n_batch
                label_sum += label.sum()
                pred_sum += torch.sigmoid(pred).sum()

            if i % 200 == 0:
                desc = f'train loss = {epoch_loss.item() / epoch_n: .4f}, ' + \
                       f'CTR = {label_sum.item() / epoch_n: .4f}, ' + \
                       f'pCTR = {pred_sum.item() / epoch_n: .4f}, ' + \
                       f'bias = {model.bias.item(): .4f}'
                if epoch_number is not None:
                    desc = f'Epoch {1 + epoch_number}, ' + desc
                epoch_iter.set_description(desc)

            if i > args.n_batches:
                break

    return epoch_loss.item() / epoch_n


@torch.no_grad()
def test_epoch(tensors, model, criterion, batch_size, set_name='validation', epoch_number=None):
    epoch_n = 0
    epoch_loss = 0.
    label_sum = 0.
    pred_sum = 0.
    with tqdm_len(BatchIter(*tensors, batch_size=batch_size)) as epoch_iter:
        for i, batch in enumerate(epoch_iter):
            idxs, weights, label = make_input(batch)
            pred = model(idxs, weights)
            loss = criterion(pred, label)

            n_batch = label.numel()
            epoch_n += n_batch
            epoch_loss += loss * n_batch
            label_sum += label.sum()
            pred_sum += torch.sigmoid(pred).sum()
            
            if i % 50 == 0:
                desc = f'{set_name} loss = {epoch_loss.item() / epoch_n: .4f}, ' + \
                       f'CTR = {label_sum.item() / epoch_n: .4f}, ' + \
                       f'pCTR = {pred_sum.item() / epoch_n: .4f}, '
                if epoch_number is not None:
                    desc = f'Epoch {1 + epoch_number}, ' + desc
                epoch_iter.set_description(desc)

    return epoch_loss.item() / epoch_n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help='which GPU to use')
    parser.add_argument("--train_batch_size", type=int, default=1024, help='batch size used for training')
    parser.add_argument("--val_batch_size", type=int, default=8192, help='batch size used on the validation set')
    parser.add_argument("--n_epochs", type=int, default=10, help='number of training epochs')
    parser.add_argument("--n_batches", type=int, default=2 ** 31,
                        help='limit on the number of mini-batches used in each training epoch')
    parser.add_argument("--splines", type=bool, default=False, help='Weather to use splines')
    parser.add_argument("--degrees", nargs="+", type=int, default=[0], help='Spline degrees to try')
    parser.add_argument("--emb_dims", nargs="*", type=int, default=[16], help='Embedding dimensions to try')
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help='Random seeds to try')
    parser.add_argument("--force_params", type=str, default='', help='force parameters for debugging')
    parser.add_argument("--study_name", type=str, default='criteo', help='name of the optuna study')
    parser.add_argument("--n_trials", type=int, default=50, help='number of optuna trials')
    args = parser.parse_args()

    print(f'Parsed args: {args}')

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print('Using device ', device)


    def get_column_config(splines, degree):
        if splines:
            column_configs = \
                [CatColumnConfig(col) for col in cat_cols] + \
                [IntColumnConfig(col) for col in int_cols if (col not in spline_cols)] + \
                [SplineColumnConfig(col, degree) for col in spline_cols]
        else:
            column_configs = \
                [CatColumnConfig(col) for col in cat_cols] + \
                [IntColumnConfig(col) for col in int_cols]

        return column_configs


    def fit_model(lr, l2reg, n_epochs, splines, degree, emb_dim, random_seed, callback=None):
        train_batch_size = args.train_batch_size
        val_batch_size = args.val_batch_size
        print({k: v for k, v in locals().items() if k != 'callback'})

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        column_configs = get_column_config(splines, degree)
        train_tensors, n_features = load_dataset(column_configs, out_train_path, device=device)
        val_tensors, _ = load_dataset(column_configs, out_val_path, device=device)
        test_tensors, _ = load_dataset(column_configs, out_test_path, device=device)

        model = FwFM(emb_dim, n_features)
        model = model.to(device)
        optim = torch.optim.AdamW(
            [
                {'params': model.bias, 'weight_decay': 0.},
                {'params': model.nonbias_parameters(), 'weight_decay': l2reg}
            ], lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        early_stop = EarlyStopping()
        val_losses = []
        test_losses = []
        for epoch in range(n_epochs):
            train_epoch(train_tensors, model, criterion, optim, train_batch_size, epoch_number=epoch)
            val_loss = test_epoch(val_tensors, model, criterion, val_batch_size, epoch_number=epoch,
                                  set_name='validation')
            test_loss = test_epoch(test_tensors, model, criterion, val_batch_size, epoch_number=epoch,
                                   set_name='test')
            val_losses.append(val_loss)
            test_losses.append(test_loss)

            if early_stop(val_loss):
                print(f'Early stopped at epoch {1 + epoch}')
                break

            if callback is not None:
                callback(val_loss, epoch)

        return val_losses, test_losses


    def optuna_objective(trial: optuna.Trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        l2reg = trial.suggest_float('l2reg', 1e-8, 1e-3, log=True)
        if len(args.emb_dims) > 1:
            emb_dim = trial.suggest_categorical('emb_dim', list(args.emb_dims))
        else:
            emb_dim = args.emb_dims[0]
        random_seed = trial.suggest_categorical('random_seed', args.seeds)
        n_epochs = args.n_epochs
        degree = trial.suggest_categorical('degree', list(args.degrees))
        splines = args.splines

        val_losses, test_losses = fit_model(lr, l2reg, n_epochs, splines, degree, emb_dim, random_seed)
        best_epoch = np.argmin(val_losses)

        trial.set_user_attr('best_epoch', int(best_epoch))
        trial.set_user_attr('test_loss', float(test_losses[best_epoch]))

        return val_losses[best_epoch]


    study_name = args.study_name
    journal_name = f'{study_name}.log'
    study_storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(journal_name))
    study = optuna.create_study(study_name=study_name,
                                storage=study_storage,
                                direction='minimize',
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42))

    study.optimize(optuna_objective, n_trials=args.n_trials)

    best_trial = study.best_trial
    best_params = best_trial.params | {'n_epochs': 1 + best_trial.user_attrs['best_epoch']}
    print(f'Best hyperparameters: {best_params}')
    print(f'Best model test loss: {best_trial.user_attrs["test_loss"]}')
