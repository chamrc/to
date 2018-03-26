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
    # Defaults to 1, will convert (batch, #(labels)) to (batch, 1) for the output of the model
    GENERATE_AXIS = 'generate_axis'


class TrainerEvents(Enum):
    # Called when loss is being computed.
    # fn(x, y, extras, y_hat) => loss
    COMPUTE_LOSS = 'compute_loss'
    # Called when dataloader is being loaded to add collate_fn and sampler.
    # fn(self.cfg, data_type, dataset) => dataloader
    CUSTOMIZE_DATALOADER = 'customize_dataloader'
    # Called before calling model
    # fn(x, y, extras) => x, y, extras
    PRE_PROCESS = 'pre_process'
    # Called when calling model to get extra args and kwargss
    # fn(x, y, extras) => *args, **kwargs
    MODEL_EXTRA_ARGS = 'model_extra_args'
    # Called after calling model
    # fn(x, y, extras, y_hat) => y_hat
    POST_PROCESS = 'post_process'
    # Called to get percentage accuracy
    # fn(y, y_hat) => match_results (ndarray, 1 if correct, 0 otherwise)
    MATCH_RESULTS = 'match_results'
    # Called after processing in each batch in TEST mode to generate output to be written to output.
    # fn(x, y, extras, y_hat) => result (This will be written to CSV file)
    GENERATE = 'generate'
    # Called after test is completed before saving to a file.
    # fn(results) => results
    POST_TEST = 'post_test'
