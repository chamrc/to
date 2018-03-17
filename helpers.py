import torch
import torch.nn.functional as F
import numpy as np
import math, os, csv, sys, importlib, inspect, glob, collections
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
	return num(s) != None

def mkdirp(path):
	return os.makedirs(path, exist_ok=True)

def touch(path):
	Path(path).touch()

def has_index(l, i):
	return l is not None and i >= 0 and i < len(l)

def has(o, *k):
	if len(k) == 0: return False
	return (hasattr(o, k[0]) or (isinstance(o, collections.Iterable) and k[0] in o)) and \
		(has(get(o, k[0]), *k[1:]) or len(k) == 1)

def get(o, *k):
	if len(k) == 0: return None
	elif len(k) == 1:
		if hasattr(o, k[0]): return getattr(o, k[0])
		elif k[0] in o: return o[k[0]]
		else: return None
	else:
		if hasattr(o, k[0]): return get(getattr(o, k[0]), *k[1:])
		elif k[0] in o: return get(o[k[0]], *k[1:])
		else: return None

def w(text=''):
	sys.stdout.write('{}'.format(text))
	sys.stdout.flush()

def p(text='', show_debug=True):
	if show_debug:
		print('{} - {}() #{}: {}'.format(os.path.relpath(inspect.stack()[1][1], os.getcwd()), inspect.stack()[1][3], inspect.stack()[1][2], text))
	else:
		print(''.format(text))

# Neural Network helper function
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
	if init_uniform: torch.nn.init.uniform(m.weight)

	init_normal = get(cfg, 'init_normal') if has(cfg, 'normal') else False
	if init_normal: torch.nn.init.normal(m.weight)

	init_constant = get(cfg, 'init_constant') if has(cfg, 'constant') else False
	if init_constant: torch.nn.init.constant(m.weight)

	init_eye = get(cfg, 'init_eye') if has(cfg, 'eye') else False
	if init_eye: torch.nn.init.eye(m.weight)

	init_dirac = get(cfg, 'init_dirac') if has(cfg, 'dirac') else False
	if init_dirac: torch.nn.init.dirac(m.weight)

	init_xavier_uniform = get(cfg, 'init_xavier_uniform') if has(cfg, 'xavier_uniform') else False
	if init_xavier_uniform: torch.nn.init.xavier_uniform(m.weight)

	init_xavier_normal = get(cfg, 'init_xavier_normal') if has(cfg, 'xavier_normal') else False
	if init_xavier_normal: torch.nn.init.xavier_normal(m.weight)

	init_kaiming_uniform = get(cfg, 'init_kaiming_uniform') if has(cfg, 'kaiming_uniform') else False
	if init_kaiming_uniform: torch.nn.init.kaiming_uniform(m.weight)

	init_kaiming_normal = get(cfg, 'init_kaiming_normal') if has(cfg, 'kaiming_normal') else False
	if init_kaiming_normal: torch.nn.init.kaiming_normal(m.weight)

	init_orthogonal = get(cfg, 'init_orthogonal') if has(cfg, 'orthogonal') else False
	if init_orthogonal: torch.nn.init.orthogonal(m.weight)

	init_zero_bias = get(cfg, 'init_zero_bias') if has(cfg, 'init_zero_bias') else False
	if init_zero_bias and hasattr(m, 'bias') and hasattr(m.bias, 'data'):
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
def collate_fn(axis=1, dim=2, mode='constant', value=0, min_len=None, buckets=None):
	def collate_data(data):
		results = list(zip(*data))
		data, labels = results[0], results[1]

		lengths = [row.shape[axis - 1] for row in data]
		max_len = max(lengths)
		if min_len is not None and max_len < min_len: max_len = min_len
		if buckets is not None:
			try:
				idx = next(i for i, v in enumerate(buckets) if v > max_len)
				max_len = buckets[idx]
			except:
				raise ValueError('Buckets too small for collate_fn(), should be at least {}.'.format(max_len))

		pad_locs = [tuple(sum([[0, max_len - row.shape[axis - 1]] if i == (axis - 1) else [0, 0] for i in range(dim - 1, -1, -1)], [])) for row in data]

		results[0] = torch.stack([F.pad(row, pad_locs[i], mode, value) for i, row in enumerate(data)])
		results[1] = torch.stack(labels)
		one_hot = to_tensor(np.array([[1] * length + [0] * (max_len - length) for length in lengths]))

		return tuple(results + [one_hot])

	return collate_data

# Write output data
def match_prefix(word=None, suffix='.py', folder='configurations/'):
	options, patterns = [], None

	if word == '' or word is None:
		patterns = ['{}**/*{}'.format(folder, suffix)]
	else:
		patterns = ['{}{}**/*{}'.format(folder, word, suffix),
			'{}{}*{}'.format(folder, word, suffix)]

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
