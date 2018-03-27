import numpy as np
import time
from .options import *
from .helpers import *
from .nn import Mode


class Logger(object):

    def __init__(self, trainer):
        super(Logger, self).__init__()
        self.trainer = trainer

    def __reset(self, mode=Mode.TRAIN):
        self.print_interval = get(self.trainer.cfg, TrainerOptions.PRINT_INVERVAL.value, default=100)
        self.print_accuracy = get(self.trainer.cfg, TrainerOptions.PRINT_ACCURACY.value, default=True)

        self.mode = mode
        self.t0 = time.time()
        self.__reset_epoch()

    def __reset_epoch(self):
        self.losses, self.batch_count = [], 0
        self.interval_correct, self.interval_count = 0, 0
        self.all_correct, self.all_count = 0, 0
        self.t1 = time.time()

    def start(self, mode=Mode.TRAIN):
        self.__reset(mode)

    def start_epoch(self):
        self.__reset_epoch()

    def increment(self):
        self.batch_count += 1

    def log_loss(self, loss):
        self.losses.append(loss)

    def log_batch(self, mode, x, y, extras, y_hat):
        self.interval_count += len(x)
        self.all_count += len(x)

        if self.print_accuracy and self.mode is not Mode.TEST:
            match_results = self.trainer._match(mode, x, y, extras, y_hat)

            correct = match_results.sum()
            self.interval_correct += correct
            self.all_correct += correct

    def print_summary(self):
        template, percentage = '', 0.0
        if self.print_accuracy:
            percentage = self.all_correct / max(self.all_count, 1) * 100
            if percentage == 0:
                percentage = None
            template += '{0} / {1} ({2:.2f} %) correct!'

        template += ' Loss is: {3:.8f} | {4:.8f} | {5:.8f}'
        min_loss, mean_loss, total_loss = self.get_loss()
        p(template.format(self.all_correct, max(self.all_count, 1), percentage, min_loss, mean_loss, total_loss))

    def print_batch(self, check_print_interval=True):
        if check_print_interval and self.batch_count % self.print_interval != 0:
            return

        curr_time = time.time()
        batch_time = curr_time - self.t1
        total_time = curr_time - self.t0
        self.t1 = time.time()

        # get_lr = getattr(self.trainer, '__get_lr')
        # lr = get_lr()
        lr = self.trainer._get_lr()
        if self.mode is Mode.TEST:
            template = 'lr {} => Done testing batch {} count {}. Time elapsed: {:.2f} | {:.2f} seconds.'
            p(template.format(lr, self.batch_count, self.all_count, batch_time, total_time), debug=False)
        else:
            percentage = (self.interval_correct / max(self.interval_count, 1)) * 100

            template = None
            if self.print_accuracy:
                template = 'lr {0} => Done {1} epoch {2} batch {3} count {4}. Time elapsed: {5:.2f} | {6:.2f} seconds. Accuracy: {7:.2f} %. Loss: {8:.8f} | {9:.8f}.'
            else:
                template = 'lr {0} => Done {1} epoch {2} batch {3} count {4}. Time elapsed: {5:.2f} | {6:.2f} seconds. Loss: {8:.8f} | {9:.8f}.'

            epoch = self.trainer.epoch_ran + 1 if self.mode is Mode.TRAIN else self.trainer.epoch_ran
            mode_name = 'training' if self.mode is Mode.TRAIN else 'validating'

            min_loss, mean_loss, _ = self.get_loss()
            p(template.format(lr, mode_name, epoch, self.batch_count, self.all_count, \
                batch_time, total_time, percentage, min_loss, mean_loss), debug=False)

    def get_percentage(self):
        percentage = self.all_correct / max(self.all_count, 1) * 100
        return percentage

    def get_loss(self):
        losses = self.losses[-self.print_interval:]
        min_loss, mean_loss, total_loss = float('inf'), float('inf'), float('inf')
        if len(losses):
            min_loss = np.asscalar(np.amin(losses))
            mean_loss = np.asscalar(np.mean(losses))
            total_loss = np.asscalar(np.mean(self.losses))
        return min_loss, mean_loss, total_loss
