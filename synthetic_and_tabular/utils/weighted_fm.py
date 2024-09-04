import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WeightedEmbeddingBag(nn.Module):
    r"""
    A version of ``nn.EmbeddingBag`` which supports mini-batches.

    Args:
        num_embeddings (int): the number of emebdding vectors to hold
        embedding_dim (int): the dimension of each embedding vector
        weight_init_fn (Tensor -> None): an optional function to initialize the embedding weight matrix.
    """
    def __init__(self, num_embeddings, embedding_dim, weight_init_fn=None, **emb_kwargs):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, **emb_kwargs)
        if weight_init_fn is not None:
            with torch.no_grad():
                weight_init_fn(self.emb.weight)

    def forward(self, input, per_sample_weights, offsets, return_l2=False):
        r"""
        Computed weighted sums of input embeddings in each bag, assuming each mini-batch comprises the same
        number of embeddings, weights, and bags. Variable number of embeddings and their corresponding weights
        per sample is possible with padding. However, the number of bags per sample has to be equal for all
        mini-batch samples. Returns a tensor of weighted-sums of embedding vectors in each sample.

        Args:
            input (Tensor): BxN matrix, where each row contains per-sample embedding indices.
            per_sample_weights (Tensor): BxN matrix, where each row contaisn per-sample embedding weights.
            offsets (Tensor): BxM offsets pointing to end-of-bag indices inside each sample.
            return_l2 (bool): If True, also return the squared L2 norm of the chosen embedding vectors
        """
        embeddings = self.emb(input)
        weighted_embeddings = embeddings * per_sample_weights.unsqueeze(2)
        padded_summed = F.pad(weighted_embeddings, [0, 0, 1, 0, 0, 0]).cumsum(dim=1)
        padded_offsets = F.pad(offsets, [1, 0, 0, 0], value=-1) + 1

        def batch_gather(input, off):
            emb_dim = input.shape[2]
            batch_size = off.shape[0]
            num_offsets = off.shape[1]
            i = torch.arange(batch_size, device=input.device).reshape(batch_size, 1, 1)
            j = off.reshape(batch_size, num_offsets, 1)
            k = torch.arange(emb_dim, device=input.device)

            return input[i, j, k]

        score = batch_gather(padded_summed, padded_offsets[:, 1:]) - batch_gather(padded_summed, padded_offsets[:, :-1])
        if return_l2:
            return score, embeddings.square().mean(0).sum()
        else:
            return score


class WeightedFM(torch.nn.Module):
    def __init__(self, num_features, embedding_dim, **emb_kwargs):
        super().__init__()
        vec_init_scale = 1. / math.sqrt(embedding_dim)
        self.vectors = WeightedEmbeddingBag(num_features, embedding_dim,
                                            weight_init_fn=lambda ws: torch.nn.init.uniform_(ws, 0, vec_init_scale),
                                            **emb_kwargs)
        self.biases = WeightedEmbeddingBag(num_features, 1,
                                           weight_init_fn=lambda ws: torch.nn.init.zeros_(ws),
                                           **emb_kwargs)
        self.bias = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, indices, weights, offsets, return_l2=False):
        if return_l2:
            vectors, l2 = self.vectors(indices, weights, offsets, return_l2=return_l2)
        else:
            vectors = self.vectors(indices, weights, offsets, return_l2=return_l2)
        biases = self.biases(indices, weights, offsets).squeeze()

        square_of_sum = vectors.sum(dim=1).square()
        sum_of_square = vectors.square().sum(dim=1)
        pairwise = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
        linear = biases.squeeze().sum(dim=1)

        if return_l2:
            return (pairwise + linear + self.bias), l2
        else:
            return pairwise + linear + self.bias


class WeightedFFM(torch.nn.Module):
    def __init__(self, num_features, field_dim, num_fields, **emb_kwargs):
        super().__init__()
        vec_init_scale = 1. / math.sqrt(field_dim)
        self.field_dim = field_dim
        self.num_fields = num_fields
        i_indices, j_indices = torch.tril_indices(num_fields, num_fields, -1)
        self.register_buffer('i_indices', i_indices)
        self.register_buffer('j_indices', j_indices)
        self.vectors = WeightedEmbeddingBag(num_features, field_dim * num_fields,
                                            weight_init_fn=lambda ws: torch.nn.init.uniform_(ws, -vec_init_scale,
                                                                                             vec_init_scale),
                                            **emb_kwargs)
        self.biases = WeightedEmbeddingBag(num_features, 1,
                                           weight_init_fn=lambda ws: torch.nn.init.zeros_(ws),
                                           **emb_kwargs)
        self.bias = torch.nn.Parameter(torch.tensor(0.))

    def _fast_ffm_pairwise(self, batch_size, vectors, fields):
        fields_i = fields[:, self.i_indices]
        fields_j = fields[:, self.j_indices]

        batches = torch.arange(batch_size, device=vectors.device)
        vectors_i = vectors[batches[:, None], fields_i, fields_j]
        vectors_j = vectors[batches[:, None], fields_j, fields_i]

        pairwise = (vectors_i * vectors_j).sum(dim=[-1, -2])
        return pairwise

    def forward(self,
                indices: torch.Tensor,
                weights: torch.Tensor,
                offsets: torch.Tensor,
                fields: torch.Tensor,
                return_l2: bool = False):
        r"""
        Returns FFM scores corresponding to a mini-batch of weighted sums of embedding bags. The scores
        are computed according to the full FFM score function, including the linear and bias terms.

        Args:
            indices (Tensor): BxN matrix of embedding indices in the mini-batch
            weights (Tensor): BxN matrix of corresponding embedding weights
            offsets (Tensor): BxM matrix of bag end-point offsets, as in ::see WeightedEmbeddingBag::
            fields (Tensor): BxM matrix of the field each bag corresponds to.
            return_l2 (bool): if True, returns also the mean squared L2 norm of the embedding vectors.
        """
        if return_l2:
            vectors, l2 = self.vectors(indices, weights, offsets, return_l2=return_l2)
        else:
            vectors = self.vectors(indices, weights, offsets, return_l2=return_l2)
        biases = self.biases(indices, weights, offsets).squeeze()

        batch_size = vectors.shape[0]
        vectors = vectors.view(batch_size, self.num_fields, self.num_fields, self.field_dim)
        pairwise = self._fast_ffm_pairwise(batch_size, vectors, fields)
        linear = biases.squeeze().sum(dim=1)
        if return_l2:
            return (pairwise + linear + self.bias), l2
        else:
            return pairwise + linear + self.bias