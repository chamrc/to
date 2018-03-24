from enum import Enum

DEV, TRAIN, TEST = range(3)
data_type_name = lambda x: ['dev', 'train', 'test'][x]


class WSJOptions(Enum):
    WSJ_FOLDER = 'wsj_folder'
    DEV_DATA_FILE = 'dev_data_file'
    DEV_LABELS_FILE = 'dev_labels_file'
    TRAIN_DATA_FILE = 'train_data_file'
    TRAIN_LABELS_FILE = 'train_labels_file'
    TEST_DATA_FILE = 'test_data_file'


class NeuralNetworkOptions(Enum):
    LAYERS = 'layers'


class TrainerOptions(Enum):
    OPTIMIZER = 'optimizer'
    OPTIMIZER_ARGS = 'optimizer_args'
    LOSS_FN = 'loss_fn'
    LAYERS = 'layers'
    IN_CHANNELS = 'in_channels'
    OUT_CHANNELS = 'out_channels'
    AUTO_RELOAD_SAVED_MODEL = 'auto_reload_saved_model'


class TrainerEvents(Enum):
    COMPUTE_LOSS = 'compute_loss'
