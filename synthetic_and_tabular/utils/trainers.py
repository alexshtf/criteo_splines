import math
from typing import Callable

import torch
from torch.utils.data import DataLoader

from weighted_fm import WeightedFFM


class FFMTrainer:
    def __init__(self, embedding_dim: int, step_size: float, batch_size: int, num_epochs: int,
                 callback: Callable[[int, float], None] = None):
        self.embedding_dim = embedding_dim
        self.step_size = step_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.callback = callback

    def train(self, num_fields, num_embeddings, train_ds, test_ds, criterion, device):
        fm = WeightedFFM(num_embeddings, self.embedding_dim, num_fields, sparse=True).to(device)
        optimizer = torch.optim.Adagrad(fm.parameters(), lr=self.step_size)
        for epoch in range(self.num_epochs):
            fm.train()
            for indices, weights, offsets, fields, target in DataLoader(train_ds, batch_size=self.batch_size,
                                                                        shuffle=True, drop_last=True, pin_memory=True):
                output = fm(indices.to(device), weights.to(device), offsets.to(device), fields.to(device))
                loss = criterion(output, target.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            fm.eval()
            with torch.no_grad():
                for indices, weights, offsets, fields, target in DataLoader(test_ds, batch_size=len(test_ds),
                                                                            shuffle=False, pin_memory=True):
                    output = fm(indices.to(device), weights.to(device), offsets.to(device), fields.to(device))
                    loss = criterion(output, target.to(device))
                    test_loss = loss.item()

            if self.callback is not None:
                self.callback(epoch, test_loss)

        return test_loss
