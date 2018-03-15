import numpy as np
import os
from enum import IntEnum, unique
from flare.helpers import p

DEV, TRAIN, TEST = range(3)


class WSJ():
    """ Load the WSJ speech dataset

    Ensure WSJ_PATH is path to directory containing
    all data files (.npy) provided on Kaggle.

    Example usage:
        loader = WSJ()
        trainX, trainY = loader.train
        assert(trainX.shape[0] == 24590)

"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.dev_set = None
        self.train_set = None
        self.test_set = None
        self.directory = os.environ['PWD'] + '/data'

    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = self.load_raw(self.directory, DEV)
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = self.load_raw(self.directory, TRAIN)
        return self.train_set

    @property
    def test(self):
        if self.test_set is None:
            self.test_set = self.load_raw(self.directory, TEST)
        return self.test_set

    def get_type_name(self, data_type):
        if data_type is DEV:
            return 'dev'
        elif data_type is TRAIN:
            return 'train'
        elif data_type is TEST:
            return 'test'
        else:
            return ''

    def get_path(self, data_type, array_type):
        if hasattr(self.cfg, '{}_{}_path'.format(
                self.get_type_name(data_type), array_type)):
            return getattr(self.cfg, '{}_{}_path'.format(
                self.get_type_name(data_type), array_type))

        return os.path.join(self.directory, '{}-{}.npy'.format(
            self.get_type_name(data_type), array_type))

    def load_path(self, data_type, array_type):
        path = self.get_path(data_type, array_type)
        if os.path.isfile(path):
            results = np.load(
                self.get_path(data_type, array_type), encoding='bytes')
            p('Dataset "{}" has {} records.'.format(path, len(results)))
            return results
        return None

    def save_path(self, data, data_type, array_type):
        path = self.get_path(data_type, array_type)
        np.save(path, data, allow_pickle=False)

    def load_raw(self, path, data_type):
        return (self.load_path(data_type, 'features'),
                self.load_path(data_type, 'labels'))

    def __len__(self):
        return 3

    def __getitem__(self, data_type):
        if data_type is DEV:
            return self.dev
        elif data_type is TRAIN:
            return self.train
        else:
            return self.test
