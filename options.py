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
    IN_CHANNELS = 'in_channels'
    OUT_CHANNELS = 'out_channels'
    LAYERS = 'layers'


class TextModelOptions(Enum):
    EMBEDDING = 'embedding'
    PROJECTION = 'projection'
    PACK_PADDED = 'pack_padded'


class TrainerOptions(Enum):
    BATCH_SIZE = 'batch_size'
    OPTIMIZER = 'optimizer'
    OPTIMIZER_ARGS = 'optimizer_args'
    LOSS_FN = 'loss_fn'
    AUTO_RELOAD_SAVED_MODEL = 'auto_reload_saved_model'
    DEV_MODE = 'dev_mode'
    PRINT_INVERVAL = 'print_inverval'
    PRINT_ACCURACY = 'print_accuracy'


class TrainerEvents(Enum):
    # fn(x, y, extras, y_hat) => loss
    COMPUTE_LOSS = 'compute_loss'
    # fn(self.cfg, data_type, dataset) => dataloader
    CUSTOMIZE_DATALOADER = 'customize_dataloader'
    # fn(x, y, extras) => *args, **kwargs
    MODEL_EXTRA_ARGS = 'model_extra_args'
    # fn(x, y, extras) => x, y, extras
    PRE_PROCESS = 'pre_process'
    # fn(x, y, extras, y_hat) => y_hat
    POST_PROCESS = 'post_process'
    # fn(y, y_hat) => match_results (ndarray, 1 if correct, 0 otherwise)
    MATCH_RESULTS = 'match_results'
