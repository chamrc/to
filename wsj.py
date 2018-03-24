import numpy as np
import os
from .options import *
from .helpers import *


class WSJ():

    def __init__(self, cfg):
        self.cfg = cfg
        self.dev_set = None
        self.train_set = None
        self.test_set = None
        default_directory = os.path.join(csd(), 'data')
        self.directory = get(self.cfg, WSJOptions.WSJ_FOLDER.value, default=default_directory)

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

    def __value_name(self, is_label):
        return 'labels' if is_label else 'features'

    def __get_path(self, data_type, is_label):
        options = [
            (WSJOptions.DEV_DATA_FILE, WSJOptions.DEV_LABELS_FILE), \
            (WSJOptions.TRAIN_DATA_FILE, WSJOptions.TRAIN_LABELS_FILE), \
            (WSJOptions.TEST_DATA_FILE, None)
        ]
        key = options[data_type][is_label]

        if key is not None and has(self.cfg, key.value):
            return os.path.join(self.directory, get(self.cfg, key.value))
        else:
            return os.path.join(
                self.directory, '{}-{}.npy'.format(data_type_name(data_type), self.__value_name(is_label))
            )

    def __load_path(self, data_type, is_label):
        path = self.__get_path(data_type, is_label)
        if os.path.isfile(path):
            results = np.load(path, encoding='bytes')
            p('Dataset "{}" has {} records.'.format(path, len(results)))
            return results
        return None

    def __save_path(self, data, data_type, is_label):
        path = self.__get_path(data_type, is_label)
        np.save(path, data, allow_pickle=False)

    def load_raw(self, path, data_type):
        return (self.__load_path(data_type, False), self.__load_path(data_type, True))

    def __len__(self):
        return 3

    def __getitem__(self, data_type):
        if data_type is DEV:
            return self.dev
        elif data_type is TRAIN:
            return self.train
        else:
            return self.test
