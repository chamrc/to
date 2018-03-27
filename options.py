from enum import Enum

DEV, TRAIN, TEST = range(3)
data_type_name = lambda x: ['dev', 'train', 'test'][x]


class WSJOptions(Enum):
    WSJ_FOLDER = 'wsj_folder'  # Defaults to 'data'
    DEV_DATA_FILE = 'dev_data_file'  # Defaults to 'dev-features.npy'
    DEV_LABELS_FILE = 'dev_labels_file'  # Defaults to 'dev-labels.npy'
    TRAIN_DATA_FILE = 'train_data_file'  # Defaults to 'train-features.npy'
    TRAIN_LABELS_FILE = 'train_labels_file'  # Defaults to 'train-labels.npy'
    TEST_DATA_FILE = 'test_data_file'  # Defaults to 'test-features.npy'


class NeuralNetworkOptions(Enum):
    IN_CHANNELS = 'in_channels'  # *MUST*
    OUT_CHANNELS = 'out_channels'  # *MUST*
    LAYERS = 'layers'


class TextModelOptions(Enum):
    PACK_PADDED = 'pack_padded'  # Defaults to False


class TrainerOptions(Enum):
    BATCH_SIZE = 'batch_size'  # Defaults to 64
    OPTIMIZER = 'optimizer'  # Defaults to Adam
    OPTIMIZER_ARGS = 'optimizer_args'  # Defaults to { 'lr': 0.01 }
    SCHEDULER = 'scheduler'  # Defaults to schedule on validation loss
    SCHEDULER_ARGS = 'scheduler_args'
    SCHEDULE_VERBOSE = 'schedule_verbose'
    SCHEDULE_FIRST = 'schedule_first'  # Run scheduler before or after train, default: before
    SCHEDULE_BATCH_COUNT = 'schedule_batch_count'  # num of batch to use to get accuracy & loss from validation
    SCHEDULE_ON_BATCH = 'schedule_on_batch'  # Call scheduler for each batch, instead of epoch
    SCHEDULE_ON_ACCURACY = 'schedule_on_accuracy'  # Only works if print_accuracy is True
    SCHEDULE_ON_TRAIN_DATA = 'schedule_on_train_data'  # Schedule on train data instead
    LOSS_FN = 'loss_fn'  # Defaults to nn.CrossEntropyLoss
    AUTO_RELOAD_SAVED_MODEL = 'auto_reload_saved_model'
    DEV_MODE = 'dev_mode'  # Defaults to False
    PRINT_INVERVAL = 'print_inverval'  # Defaults to 100
    PRINT_ACCURACY = 'print_accuracy'  # Defaults to True
    CSV_FIELD_NAMES = 'csv_field_names'  # Defaults to ['id', 'label']
    # Generate test output (batch, 1) from y_hat (batch, classes)
    GENERATE_AXIS = 'generate_axis'  # Defaults to 1


class TrainerEvents(Enum):  # Events that can be binded
    # Called when loss is being computed.
    # fn(mode, x, y, extras, y_hat) => loss
    COMPUTE_LOSS = 'compute_loss'
    # Called when dataloader is being loaded to add collate_fn and sampler.
    # fn(self.cfg, data_type, dataset) => dataloader
    CUSTOMIZE_DATALOADER = 'customize_dataloader'
    # Called before calling model
    # fn(mode, x, y, extras) => x, y, extras
    PRE_PROCESS = 'pre_process'
    # Called when calling model to get extra args and kwargss
    # fn(mode, x, y, extras) => *args, **kwargs
    MODEL_EXTRA_ARGS = 'model_extra_args'
    # Called after calling model
    # fn(mode, x, y, extras, y_hat) => y_hat
    POST_PROCESS = 'post_process'
    # Called to get percentage accuracy
    # fn(mode, x, y, extras, y_hat) => match_results (ndarray, 1 if correct, 0 otherwise)
    MATCH_RESULTS = 'match_results'
    # Called after processing in each batch in TEST mode to generate output to be written to output.
    # fn(x, y, extras, y_hat) => result (This will be written to CSV file)
    GENERATE = 'generate'
    # Called after test is completed before saving to a file.
    # fn(results) => results
    POST_TEST = 'post_test'
