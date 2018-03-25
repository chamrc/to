import os
import sys
import inspect
import collections
import pathlib
import glob
import torch
import torch.nn.functional as F
import numpy as np

#----------------------------------------------------------------------------------------------------------
# Decorators
#----------------------------------------------------------------------------------------------------------


def static(varname, value):

    def decorate(func):
        setattr(func, varname, value)
        return func

    return decorate


#----------------------------------------------------------------------------------------------------------
# List Helpers
#----------------------------------------------------------------------------------------------------------


def __first(lmd, lst, asc=True):
    indexes = range(len(lst)) if asc else range(len(lst) - 1, -1, -1)
    for i in indexes:
        o = lst[i]
        if lmd(i, o, lst) == True:
            return i
    return -1


def first(lmd, lst):
    return __first(lmd, lst, True)


def last(lmd, lst):
    return __first(lmd, lst, False)


def all(lmd, lst, asc=True):
    indexes = range(len(lst)) if asc else range(len(lst) - 1, -1, -1)
    results = []
    for i in indexes:
        o = lst[i]
        if lmd(i, o, lst) == True:
            results.append(i)
    return results


#----------------------------------------------------------------------------------------------------------
# IO Helpers
#----------------------------------------------------------------------------------------------------------


def csd():
    return os.path.dirname(os.path.abspath(sys.argv[0]))


def cwd():
    return os.path.abspath(os.getcwd())


def filename(path):
    return os.path.basename(path)


def find_pattern(pattern, relative_to=None):
    files = glob.iglob(pattern, recursive=True)
    if relative_to is not None:
        prefix = csd()
        files = list(map(lambda x: os.path.relpath(x, prefix), files))
    return files


def match_prefix(word=None, suffix='.py', folder='configurations/'):
    options, patterns, folder = [], None, csd()

    if word == '' or word is None:
        patterns = ['{}**/*{}'.format(folder, suffix)]
    else:
        patterns = ['{}{}**/*{}'.format(folder, word, suffix), '{}{}*{}'.format(folder, word, suffix)]

    for pattern in patterns:
        pattern = os.path.join(folder, pattern)
        for path in find_pattern(pattern, relative_to=folder):
            path = path.replace(folder, '')
            options.append(path)

    return options


def touch(path, relative_to=csd()):
    pathlib.Path(os.path.join(relative_to, path)).touch()


def mkdirp(path, relative_to=csd()):
    return os.makedirs(os.path.join(relative_to, path), exist_ok=True)


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


#----------------------------------------------------------------------------------------------------------
# Types
#----------------------------------------------------------------------------------------------------------


def is_int(i):
    return isinstance(i, int)


def is_float(i):
    return isinstance(i, float)


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
    else:
        return None


def is_num(i):
    return num(i) != None


def is_str(i):
    return isinstance(i, str)


def is_iter(i):
    return isinstance(i, collections.Iterable)


def is_dict(i):
    return isinstance(i, dict)


#----------------------------------------------------------------------------------------------------------
# Print and Log
#----------------------------------------------------------------------------------------------------------


def w(text=''):
    sys.stdout.write('{}'.format(text))
    sys.stdout.flush()


def p(text='', show_debug=True):
    if show_debug:
        print(
            '{} - {}() #{}: {}'.format(
                os.path.relpath(inspect.stack()[1][1], os.getcwd()),
                inspect.stack()[1][3],
                inspect.stack()[1][2], text
            )
        )
    else:
        print('{}'.format(text))


#----------------------------------------------------------------------------------------------------------
# Get & Has
#----------------------------------------------------------------------------------------------------------
def has_index(l, i):
    return l is not None and i >= 0 and i < len(l)


def has(o, *k):
    if len(k) == 1:
        if is_int(k[0]) and has_index(o, k[0]):
            return True
        elif is_str(k[0]) and hasattr(o, k[0]):
            return True
        elif is_dict(o) and k[0] in o:
            return True
    elif len(k) > 1:
        if is_int(k[0]) and has_index(o, k[0]):
            return has(o[k[0]], *k[1:])
        elif is_str(k[0]) and hasattr(o, k[0]):
            return has(get(o, k[0]), *k[1:])
        elif is_dict(o) and k[0] in o:
            return has(o[k[0]], *k[1:])
    return False


def get(o, *k, default=None):
    if len(k) == 0:
        return default
    elif len(k) == 1:
        if is_int(k[0]) and has_index(o, k[0]):
            return o[k[0]]
        elif is_str(k[0]) and hasattr(o, k[0]):
            return getattr(o, k[0])
        elif is_dict(o) and k[0] in o:
            return o[k[0]]
        else:
            return default
    else:
        if is_int(k[0]) and has_index(o, k[0]):
            return get(o[k[0]], *k[1:], default=default)
        elif is_str(k[0]) and hasattr(o, k[0]):
            return get(getattr(o, k[0]), *k[1:], default=default)
        elif is_dict(o) and k[0] in o:
            return get(o[k[0]], *k[1:], default=default)
        else:
            return default


#----------------------------------------------------------------------------------------------------------
# PyTorch helpers
#----------------------------------------------------------------------------------------------------------


def to_tensor(numpy_array):
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    if torch.cuda.is_available() and hasattr(tensor, 'cuda'):
        tensor = tensor.cuda()

    if not torch.is_tensor(tensor):
        return tensor
    return torch.autograd.Variable(tensor)


def init_parameters(cfg):

    def init_params(m):
        if type(m) == torch.nn.Linear:
            init_layer_parameters(m, cfg)
        elif type(m) == torch.nn.Conv1d or \
         type(m) == torch.nn.Conv2d or \
         type(m) == torch.nn.Conv3d:
            init_layer_parameters(m, cfg)

    return init_params


def init_layer_parameters(m, cfg):
    init_uniform = get(cfg, 'init_uniform') if has(cfg, 'uniform') else False
    if init_uniform:
        torch.nn.init.uniform(m.weight)

    init_normal = get(cfg, 'init_normal') if has(cfg, 'normal') else False
    if init_normal:
        torch.nn.init.normal(m.weight)

    init_constant = get(cfg, 'init_constant') if has(cfg, 'constant') else False
    if init_constant:
        torch.nn.init.constant(m.weight)

    init_eye = get(cfg, 'init_eye') if has(cfg, 'eye') else False
    if init_eye:
        torch.nn.init.eye(m.weight)

    init_dirac = get(cfg, 'init_dirac') if has(cfg, 'dirac') else False
    if init_dirac:
        torch.nn.init.dirac(m.weight)

    init_xavier_uniform = get(cfg, 'init_xavier_uniform') if has(cfg, 'xavier_uniform') else False
    if init_xavier_uniform:
        torch.nn.init.xavier_uniform(m.weight)

    init_xavier_normal = get(cfg, 'init_xavier_normal') if has(cfg, 'xavier_normal') else False
    if init_xavier_normal:
        torch.nn.init.xavier_normal(m.weight)

    init_kaiming_uniform = get(cfg, 'init_kaiming_uniform') if has(cfg, 'kaiming_uniform') else False
    if init_kaiming_uniform:
        torch.nn.init.kaiming_uniform(m.weight)

    init_kaiming_normal = get(cfg, 'init_kaiming_normal') if has(cfg, 'kaiming_normal') else False
    if init_kaiming_normal:
        torch.nn.init.kaiming_normal(m.weight)

    init_orthogonal = get(cfg, 'init_orthogonal') if has(cfg, 'orthogonal') else False
    if init_orthogonal:
        torch.nn.init.orthogonal(m.weight)

    init_zero_bias = get(cfg, 'init_zero_bias') if has(cfg, 'init_zero_bias') else False
    if init_zero_bias and hasattr(m, 'bias') and hasattr(m.bias, 'data'):
        m.bias.data.zero_()


def collate(data, axis=1, dim=2, mode='constant', value=0, min_len=None, concat_labels=False):
    axis, dim = axis - 1, dim - 1
    # axis and dim are on a per row basis
    data.sort(key=lambda x: len(x[0]), reverse=True)

    results = list(zip(*data))
    data, labels = results[0], results[1]

    lengths = [row.shape[axis] for row in data]
    max_len = max(lengths)

    if min_len is not None and max_len < min_len:
        max_len = min_len

    pad_locs = [(0, 0) * (dim - axis - 1) + (0, max_len - row.shape[axis]) for i, row in enumerate(data)]

    results[0] = torch.stack([F.pad(row, pad_locs[i], mode, value) for i, row in enumerate(data)])

    if concat_labels:
        results[1] = torch.cat(labels)
    else:
        results[1] = torch.stack(labels)
    one_hot = to_tensor(np.array([[1] * length + [0] * (max_len - length) for length in lengths]))

    return tuple(results + [lengths, one_hot])
