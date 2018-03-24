import time
from torch.utils.data import TensorDataset
from .wsj import *


class DataSet(TensorDataset):

    def __init__(self, cfg, data_type):
        self.cfg, self.data_type = cfg, data_type

        wsj = WSJ(self.cfg)

        p('Loading raw dataset "{}".'.format(data_type_name(data_type)))
        t0 = time.time()

        self.data, self.labels = wsj[data_type]

        p('Done loading raw data in {:.3} secs.'.format(time.time() - t0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return to_tensor(np.array(self.data[i])), to_tensor(np.array(self.labels[i]))
