import numpy as np
import time
from .options import *
from .helpers import *


class BatchLogger(object):

    def __init__(self, trainer):
        super(BatchLogger, self).__init__()
        self.trainer = trainer

    def __reset(self, data_type=TRAIN):
        self.print_interval = get(self.trainer.cfg, TrainerOptions.PRINT_INVERVAL.value, default=100)
        self.print_accuracy = get(self.trainer.cfg, TrainerOptions.PRINT_ACCURACY.value, default=True)

        self.data_type = data_type
        self.t0 = time.time()
        self.__reset_batch()

    def __reset_batch(self, data_type=TRAIN):
        self.losses, self.batch_count = [], 0
        self.interval_correct, self.interval_count = 0, 0
        self.all_correct, self.all_count = 0, 0
        self.t1 = time.time()

    def start(self, data_type=TRAIN):
        self.__reset(data_type)

    def start_epoch(self):
        self.__reset_batch()

    def increment(self):
        self.batch_count += 1

    def log_loss(self, loss):
        self.losses.append(loss)

    def log_batch(self, x, y, extras, y_hat):
        self.interval_count += len(x)
        self.all_count += len(x)

        if self.print_accuracy and self.data_type is not TEST:
            match_results = self.trainer.__match(x, y, extras, y_hat)

            correct = match_results.sum()
            self.interval_correct += correct
            self.all_correct += correct

    def print_percentage(self):
        percentage = self.all_correct / max(self.all_count, 1) * 100
        if percentage == 0:
            percentage = None
        p('{0} / {1} ({2:.2f} %) correct!'.format(self.all_correct, max(self.all_count, 1), percentage))

    def print_batch(self):
        if self.batch_count % self.print_interval != 0:
            return

        curr_time = time.time()
        batch_time = curr_time - self.t1
        total_time = curr_time - self.t0
        self.t1 = time.time()

        if self.data_type == TEST:
            template = 'Done testing batch {} count {}. Time elapsed: {:.2f} | {:.2f} seconds.'
            p(template.format(self.batch_count, self.total_data, batch_time, total_time))
        else:
            percentage = (self.interval_correct / max(self.interval_count, 1)) * 100

            template = None
            if self.print_accuracy:
                template = 'Done {0} epoch {1} batch {2} count {3}. Time elapsed: {4:.2f} | {5:.2f} seconds. Accuracy: {6:.2f} %. Loss: {7:.8f} | {8:.8f}.'
            else:
                template = 'Done {0} epoch {1} batch {2} count {3}. Time elapsed: {4:.2f} | {5:.2f} seconds. Loss: {7:.8f} | {8:.8f}.'

            data_type_name = 'training' if self.data_type == TRAIN else 'validating'

            min_loss, mean_loss = self.get_loss()
            p(template.format(data_type_name, self.trainer.epoch_ran + 1, self.batch_count, self.all_count, \
                batch_time, total_time, percentage, min_loss, mean_loss))

    def get_percentage(self):
        percentage = self.all_correct / max(self.all_count, 1) * 100
        if percentage == 0:
            percentage = None
        return percentage

    def get_loss(self):
        losses = self.losses[-self.print_interval:]
        min_loss = np.asscalar(np.amin(losses))
        mean_loss = np.asscalar(np.mean(losses))
        return min_loss, mean_loss
