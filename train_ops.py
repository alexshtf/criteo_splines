import torch
from torch import nn
from tqdm.auto import tqdm


class EarlyStopping:
    def __init__(self, patience=3, delta=0.):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        if val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0
            self.best_loss = val_loss

        return False


class FwFM(nn.Module):
    def __init__(self, emb_dim: int, n_features: [int], emb_kwargs=None):
        super().__init__()
        if emb_kwargs is None:
            emb_kwargs = dict()

        n_fields = len(n_features)
        self.bias = nn.Parameter(torch.zeros(1))
        self.lin = nn.Linear(n_fields * emb_dim, 1, bias=False)
        self.vecs = nn.ModuleList([nn.Embedding(n, emb_dim, **emb_kwargs) for n in n_features])
        self.r = nn.Parameter(torch.zeros(n_fields, n_fields))

        with torch.no_grad():
            nn.init.normal_(self.r, std=1.)
            self.r.set_(self.r.triu(1))
            nn.init.normal_(self.lin.weight, std=1.)
            for emb in self.vecs:
                nn.init.normal_(emb.weight, std=0.01)

    def nonbias_parameters(self):
        yield from self.lin.parameters()
        yield from self.vecs.parameters()
        yield self.r

    def forward(self, idxs, weights):
        vecs = [(emb(idx) * weight.unsqueeze(-1)).sum(-2, keepdim=True)
                for idx, weight, emb in zip(idxs, weights, self.vecs)]
        vecs = torch.cat(vecs, dim=-2)
        r = self.r.triu(1)

        lin = self.lin(vecs.flatten(-2, -1)).squeeze(-1)                        # <W, V>, where W is learnable
        pwise = (torch.matmul(vecs, vecs.transpose(-1, -2)) * r).sum([-1, -2])  # <V V^T, R>, R upper triangular
        return self.bias + lin + pwise


class BatchIter:
    """
    tensors: feature tensors (each with shape: num_instances x *)
    """

    def __init__(self, *tensors, batch_size, shuffle=True):
        self.tensors = tensors

        device = tensors[0].device
        n = tensors[0].size(0)
        if shuffle:
            idxs = torch.randperm(n, device=device)
        else:
            idxs = torch.arange(n, device=device)

        self.idxs = idxs.split(batch_size)

    def __len__(self):
        return len(self.idxs)

    def __iter__(self):
        tensors = self.tensors
        for batch_idxs in self.idxs:
            yield tuple((x[batch_idxs, ...] for x in tensors))


def tqdm_len(iterable_with_len):
    return tqdm(iterable_with_len, total=len(iterable_with_len))