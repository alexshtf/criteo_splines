import torch
from torchdata.datapipes.iter import IterableWrapper, LineReader
from functools import partial
import itertools


def collate_batch(batch):
    labels = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    weights = [item[2] for item in batch]
    offsets = [item[3] for item in batch]
    fields = [item[4] for item in batch]

    indices = torch.nn.utils.rnn.pad_sequence(indices, batch_first=True, padding_value=0)
    weights = torch.nn.utils.rnn.pad_sequence(weights, batch_first=True, padding_value=0.)
    offsets = torch.tensor(offsets)
    fields = torch.tensor(fields)
    labels = torch.tensor(labels)

    return indices, weights, offsets, fields, labels


def parse_row(row: str):
    def contract_fields(fields):
        groups = ((key, sum((1 for x in grp))) for key, grp in itertools.groupby(fields))
        items, counts = itertools.tee(groups)
        items = [x[0] for x in items]
        counts = (y[1] for y in counts)
        endpoint_indices = [j - 1 for j in itertools.accumulate(counts)]

        return items, endpoint_indices

    row = row.replace(':', ' ').split(' ')
    label = float(row[0])
    fields = [int(f) for f in row[1::3]]
    indices = [int(i) for i in row[2::3]]
    weights = [float(w) for w in row[3::3]]
    fields, offsets = contract_fields(fields)
    return label, torch.as_tensor(indices), torch.as_tensor(weights), offsets, fields


def to_device(device, batch):
    if device is None:
        return batch

    return tuple(map(lambda t: t.to(device), batch))


def load_problem(path, batch_size=256, device=None):
    return LineReader(IterableWrapper([path]).open_files(), return_path=False) \
        .sharding_filter() \
        .map(parse_row) \
        .batch(batch_size=batch_size) \
        .collate(collate_fn=collate_batch) \
        .map(partial(to_device, device))
