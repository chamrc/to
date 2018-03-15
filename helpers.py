import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import csv
import sys
import importlib
import inspect
import glob
from pathlib import Path

# Helper functions


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
    else:
        return None


def isnumeric(s):
    return num(s) is not None


def mkdirp(path):
    return os.makedirs(path, exist_ok=True)


def touch(path):
    Path(path).touch()


def has(l, i):
    return l is not None and i >= 0 and i < len(l)


def w(text=''):
    sys.stdout.write('{}'.format(text))
    sys.stdout.flush()


def p(text='', show_debug=True):
    if show_debug:
        print('{} - {}() #{}: {}'.format(
            os.path.relpath(inspect.stack()[1][1], os.getcwd()),
            inspect.stack()[1][3],
            inspect.stack()[1][2], text))
    else:
        print(''.format(text))


# Neural Network helper function


def init_xavier(m):
    if isinstance(m, torch.nn.Linear):
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)
    elif isinstance(m, torch.nn.Conv1d):
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.zero_()


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):

    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available() and hasattr(tensor, 'cuda'):
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()

    if not torch.is_tensor(tensor):
        return tensor
    return torch.autograd.Variable(tensor)


# Encapsulate dimension in a closure


def collate_fn(axis=1, dim=2, mode='constant', value=0):
    def collate_data(data):
        results = list(zip(*data))
        data, labels = results[0], results[1]

        lengths = [row.shape[axis - 1] for row in data]
        max_len = max(lengths)
        pad_locs = [
            tuple(
                sum([[0, max_len - row.shape[axis - 1]]
                     if i == (axis - 1) else [0, 0]
                     for i in range(dim - 1, -1, -1)], [])) for row in data
        ]

        results[0] = torch.stack([
            F.pad(row, pad_locs[i], mode, value) for i, row in enumerate(data)
        ])
        results[1] = torch.stack(labels)
        one_hot = to_tensor(
            np.array([[1] * length + [0] * (max_len - length)
                      for length in lengths]))

        return tuple(results + [one_hot])

    return collate_data


# Write output data


def match_prefix(word=None, suffix='.py', folder='configurations/'):
    options, patterns = [], None

    if word == '' or word is None:
        patterns = ['{}**/*{}'.format(folder, suffix)]
    else:
        patterns = [
            '{}{}**/*{}'.format(folder, word, suffix), '{}{}*{}'.format(
                folder, word, suffix)
        ]

    for pattern in patterns:
        for path in glob.iglob(pattern, recursive=True):
            path = path.replace(folder, '')
            options.append(path)

    return options


def read_from_csv(path, as_type=int):
    with open(path, 'r') as csv_file:
        field_names = ['id', 'label']

        count, data = 0, []
        reader = csv.DictReader(csv_file, fieldnames=field_names)
        for row in reader:
            try:
                data.append(as_type(row['label']))
                count += 1
            except Exception as e:
                pass

        return count, data


def write_to_csv(data, path='submission.csv'):
    with open(path, 'w') as csv_file:
        field_names = ['id', 'label']

        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()

        for i, label in enumerate(data):
            writer.writerow({'id': str(i), 'label': str(label)})
