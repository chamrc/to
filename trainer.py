import re
import os
import traceback
import importlib.util
import pprint

from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.validation import ValidationError
from colored import fg, bg, attr

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .helpers import *
from .options import *
from .nn import *
from .dataset import *
from .cli import *
from .batch_logger import *


class Trainer(object):

    #----------------------------------------------------------------------------------------------------------
    # Initialization
    #----------------------------------------------------------------------------------------------------------

    def __init__(self):
        super(Trainer, self).__init__()

        self.epoch_ran = 0
        self.logger = BatchLogger()
        self.name = sys.argv[0].replace('.py', '')
        self.commands = ['list', 'help', 'use', 'load', 'run', 'test', 'validate', 'set']
        # Configurations
        self.cfg_folder = 'configurations'
        self.default_cfg = 'default'
        self.current_cfg = self.default_cfg
        self.current_cfg_path = None
        # Models
        self.models_folder = 'models'
        self.Model = NeuralNetwork
        self.cuda_enabled = False
        # Events
        self.event_handlers = {}
        # Data
        self.DataLoader = None
        self.DataSet = DataSet
        # Submission
        self.submissions_folder = 'submissions'

        self.load_cfg('{}/{}.py'.format(self.cfg_folder, self.current_cfg))
        self.reset()

    def reset(self):
        self.__init_model()
        self.__init_optim()
        self.__init_loss_fn()

    def __init_model(self):
        self.model = self.Model(self.cfg)
        if torch.cuda.is_available():
            self.cuda_enabled = True
            self.model = self.model.cuda()

    def __init_optim(self):
        Optimizer = get(self.cfg, TrainerOptions.OPTIMIZER.value, default=optim.Adam)
        arguments = get(self.cfg, TrainerOptions.OPTIMIZER_ARGS.value, default={'lr': 0.01})
        self.optimizer = Optimizer(self.model.parameters(), **arguments)

    def __init_loss_fn(self):
        Fn = get(self.cfg, TrainerOptions.LOSS_FN.value, default=nn.CrossEntropyLoss)
        self.loss_fn = Fn()

    #----------------------------------------------------------------------------------------------------------
    # Folder
    #----------------------------------------------------------------------------------------------------------

    def set_models_folder(self, models_folder):
        self.models_folder = models_folder
        return self

    def set_submissions_folder(self, submissions_folder):
        self.submissions_folder = submissions_folder
        return self

    def set_configurations_folder(self, cfg_folder):
        self.cfg_folder = cfg_folder
        self.load('{}/{}.py'.format(self.cfg_folder, self.current_cfg))
        self.reset()
        return self

    #----------------------------------------------------------------------------------------------------------
    # Configuration
    #----------------------------------------------------------------------------------------------------------

    def load_cfg(self, cfg_file):
        path = os.path.join(csd(), cfg_file)

        try:
            spec = importlib.util.spec_from_file_location('configuration', path)
            self.cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.cfg)

            self.current_cfg = filename(path).replace('.py', '')
            self.current_cfg_path = path
        except Exception as e:
            raise Exception('Configuration file not found at "{}".'.format(path))

        return self

    #----------------------------------------------------------------------------------------------------------
    # Events
    #----------------------------------------------------------------------------------------------------------

    def bind(self, event, handler):
        if isinstance(event, TrainerEvents):
            self.event_handlers[event.value] = handler
        else:
            raise Exception('Event "{}" should be a TrainerEvents.'.format(event))

        return self

    #----------------------------------------------------------------------------------------------------------
    # DataSet and DataLoader
    #----------------------------------------------------------------------------------------------------------

    def set_dataloader(self, DataLoader):
        self.DataLoader = DataLoader
        return self

    def set_dataset(self, DataSet):
        self.DataSet = DataSet
        return self

    def __get_dataloader(self, data_type):
        if self.DataLoader is not None:
            return self.DataLoader(self.cfg, data_type)
        else:
            dataset = self.DataSet(self.cfg, data_type)

            if has(self.event_handlers, TrainerEvents.CUSTOMIZE_DATALOADER.value):
                return get(self.event_handlers, TrainerEvents.CUSTOMIZE_DATALOADER.value)(self.cfg, data_type, dataset)
            else:
                should_shuffle = data_type != TEST
                batch_size = get(self.cfg, TrainerOptions.BATCH_SIZE.value, default=64)
                return DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle)

    #----------------------------------------------------------------------------------------------------------
    # Model
    #----------------------------------------------------------------------------------------------------------

    def __adjust_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.cfg.learning_rate = new_lr
        if has(self.cfg, TrainerOptions.OPTIMIZER_ARGS.value, 'lr'):
            self.cfg.optim_args['lr'] = new_lr

        return self

    def set_model(self, Model):
        self.Model = Model
        self.reset()
        return self

    def save_model(self, percentage=None, loss=None):
        mkdirp(os.path.join(csd(), self.models_folder, self.name))

        path = '{}/{}/{} - {:03d}'.format(self.models_folder, self.name, self.current_cfg, self.epoch_ran)
        if percentage is not None:
            path += ' - {:.2f}%'.format(percentage)
        if loss is not None:
            path += ' - {:.6f}'.format(loss)
        path += '.model'
        path = os.path.join(csd(), path)

        p('Saving neural network "{}" using configuration "{}" to disk at "{}"'.format( \
            self.name, self.current_cfg, path))
        torch.save(self.model.state_dict(), path)

        return self

    def load_model(self, epoch=None):
        pattern = None

        epoch, path, files, versions = self.get_versions(epoch)
        if path is None and epoch is not None:  # Can't find the exact epoch, loading the highest.
            epoch, path, files, versions = self.get_versions()

        if epoch >= 0 and path is not None:
            p('Loading neural network "{}" using configuration "{}" and epoch "{}" at "{}"'.format( \
                self.name, self.current_cfg, epoch, path))
            try:
                if torch.cuda.is_available():
                    self.model.load_state_dict(torch.load(path))
                else:
                    self.model.load_state_dict(torch.load(path, lambda storage, loc: storage))

                self.epoch_ran = epoch
            except Exception as e:
                p('Failed to load model at path "{}"'.format(path))
                traceback.print_exc()

        return self

    def has_version(self, epoch):
        version, path, files, versions = self.get_versions(epoch)
        return version > 0 and version == epoch

    def get_versions(self, epoch=None):
        folder = csd()

        if epoch is not None:
            pattern = '{}/{}/{} - {:03d}*.model'.format(self.models_folder, self.name, self.current_cfg, epoch)
        else:
            pattern = '{}/{}/{}*.model'.format(self.models_folder, self.name, self.current_cfg)

        files = find_pattern(os.path.join(folder, pattern), relative_to=folder)
        if len(files) > 0:
            versions = [int(re.findall(' \d{3} |$', filename(f))[0]) for f in files]

            epoch = max(versions)
            i = versions.index(epoch)
            path = files[i]

            return epoch, path, files, versions

        return (0, None, files, [])

    #----------------------------------------------------------------------------------------------------------
    # CLI
    #----------------------------------------------------------------------------------------------------------

    def cli(self):
        print()
        print('----------------------------------------------------------')
        print('|                                                        |')
        print('|        Welcome to Flare Neural Network Trainer.        |')
        print('|                                                        |')
        print('----------------------------------------------------------')
        print()

        if get(self.cfg, TrainerOptions.AUTO_RELOAD_SAVED_MODEL.value, default=False):
            self.load_model()

        mkdirp('.flare')
        touch('.flare/history')

        while True:
            c = prompt(
                '> ',
                history=FileHistory('.flare/history'),
                auto_suggest=AutoSuggestFromHistory(),
                completer=CommandCompleter(self),
                validator=CommandValidator(self)
            )
            try:
                self.process_command(c)
            except Exception as e:
                traceback.print_exc()

        return self

    def process_command(self, c):
        parts = list(filter(None, c.split(' ')))
        command = parts[0]

        if command == 'list':
            self.list()
        elif command == 'help':
            self.help()
        elif command == 'use':
            self.load_configuration(parts[1])
        elif command == 'load':
            if len(parts) == 2:
                self.load_model(int(parts[1]))
            else:
                self.load_model(0)
        elif command == 'run':
            if len(parts) == 1:
                self.run()
            else:
                self.run(int(parts[1]))
        elif command == 'set':
            parts = list(filter(None, c.split(' ', 2)))
            self.set(parts[1], parts[2])
        elif command == 'test' or command == 'validate':
            fn = self.test if command == 'test' else self.validate

            if len(parts) == 1:
                fn()
            else:
                locs = list(map(int, parts[1].split(':')))
                if len(locs) == 1:
                    if self.load_model(locs[0]):
                        fn()
                    else:
                        p('Skipping test because epoch {} cannot be loaded correctly.'.format(locs[0]))
                else:
                    for i in range(*locs):
                        if self.load_model(i):
                            fn()
                        else:
                            p('Skipping test because epoch {} cannot be loaded correctly.'.format(i))

    #----------------------------------------------------------------------------------------------------------
    # Commands
    #----------------------------------------------------------------------------------------------------------

    def list(self):
        color = fg(45)
        parameter = fg(119)
        reset = attr('reset')

        print('Module:\t\t\t{}{}{}'.format(color, self.name, reset))
        print('Epoch:\t\t\t{}{}{}'.format(color, self.epoch_ran, reset))
        print('Configuration:\t\t{}{}{}'.format(color, self.current_cfg, reset))
        print('Configuration Path:\t{}{}.py{}'.format(color, self.current_cfg_path, reset))
        print()
        configs = []
        for k in list(filter(lambda x: not x.startswith('__'), dir(self.cfg))):
            v = getattr(self.cfg, k)
            configs.append((k, v))
        max_key_len = max([len(k) for k, _ in configs])
        pp = pprint.PrettyPrinter(indent=4, compact=True)
        for k, v in configs:
            w('{}{} :    {}'.format(k, ' ' * (max_key_len - len(k)), parameter))
            w(re.sub('^    ', ' ' * (max_key_len + 10), pp.pformat(v), flags=re.M))
            print(reset)

        return self

    def help(self):
        command = fg(45)
        parameter = fg(119)
        sample = fg(105)
        reset = attr('reset')
        indent = '  '
        print(
            indent + """
            {0}python {1}<PYTHON>{3} {1}[CONFIG]{3} {1}[EPOCH]{3}{3}
            You can to specify the configuration file path and epoch count to load at script
            launch where {1}<PYTHON>{3} is the location of your python file, {1}[CONFIG]{3} is the
            location of your configuration file and {1}[EPOCH]{3} is the epoch count you wish to
            load.
            e.g: {2}python nn.py default 2{3}

            {0}list:{3}
                  Usage: {0}list{3}
                  List current module, epoch count and configuration file path.
                  e.g: {2}list{3}
            {0}help:{3}
                  Usage: {0}help{3}
                  Print help message.
                  e.g: {2}help{3}
            {0}use:{3}
                  Usage: {0}use{3} {1}<PATH>{3}
                  Switch to configuration file located at {1}<PATH>{3}.
                  e.g: {2}use default{3}
            {0}load:{3}
                  Usage: {0}load{3} {1}<EPOCH>{3}
                  Load previously trained model at epoch {1}<EPOCH>{3}.
                  e.g: {2}load 10{3}
            {0}run:{3}
                  Usage: {0}run{3} {1}[COUNT]{3}
                  Run training, optionally {1}[COUNT]{3} times
                  e.g: {2}run{3} OR {2}run 10{3}
            {0}set:{3}
                  Usage: {0}set{3} {1}<ATTR> <VALUE>{3}
                  Set the value in configuration dynamically, this does NOT overwrite the
                  configuration file.
                  e.g: {2}set learn_rate 0.01{3}
            {0}test:{3}
                  Usage: {0}test{3} {1}[EPOCH]{3}
                  Test using the model trained, optionally using at epoch {1}[EPOCH]{3}.
                  {1}[EPOCH]{3} can be a range input to range() or an integer.
                  e.g: {2}test 10{3} OR {2}test 1:10:2{3}
            {0}validate:{3}
                  Usage: {0}validate{3} {1}[EPOCH]{3}
                  Validate using the model trained, optionally using at epoch {1}[EPOCH]{3}.
                  {1}[EPOCH]{3} can be a range input to range() or an integer.
                  e.g: {2}validate 10{3} OR {2}validate 1:10:2{3}
            """.format(command, parameter, sample, reset).replace('\t\t\t', indent).strip()
        )

        return self

    def set(self, key, val):
        p('Setting configuration key "{}" to "{}"'.format(key, val))

        if key == 'learning_rate':
            self.__adjust_learning_rate(num(val))
        else:
            cmd = 'self.cfg.{} = {}'.format(key, val)
            try:
                exec(cmd)
                self.reset()
            except Exception as e:
                p('Failed to set configuration key "{}" to "{}"'.format(key, val))

        return self

    #----------------------------------------------------------------------------------------------------------
    # Neural Network
    #----------------------------------------------------------------------------------------------------------

    def match(self, y_hat, y):
        predictions = y_hat.data.max(1, keepdim=True)[1]
        expectations = y.long()

        if torch.cuda.is_available():
            return predictions.eq(expectations.cuda())
        else:
            return predictions.cpu().eq(expectations)

    def run(self, epochs=1):
        dev_mode = get(self.cfg, TrainerOptions.DEV_MODE.value, default=False)
        train_type = DEV if dev_mode else TRAIN
        print_interval = get(self.cfg, TrainerOptions.PRINT_INVERVAL.value, default=100)
        print_accuracy = get(self.cfg, TrainerOptions.PRINT_ACCURACY.value, default=True)

        t0 = time.time()
        dataloader = self.__get_dataloader(train_type)
        for epoch in range(epochs):
            losses, batch_count = [], 0
            interval_correct, interval_count = 0, 0
            all_correct, all_count = 0, 0
            t1 = time.time()

            for batch in dataloader:
                x, y, extras = batch[0], batch[1], batch[2:]
                self.optimizer.zero_grad()

                if has(self.event_handlers, TrainerEvents.PRE_PROCESS.value):
                    x, y, extras = get(self.event_handlers, TrainerEvents.PRE_PROCESS.value)(x, y, extras)

                y_hat = None
                self.model.train()
                if has(self.event_handlers, TrainerEvents.MODEL_EXTRA_ARGS.value):
                    args, kwargs = get(self.event_handlers, TrainerEvents.MODEL_EXTRA_ARGS.value)(x, y, extras)
                    y_hat = self.model(to_variable(x), *args, **kwargs)
                else:
                    y_hat = self.model(to_variable(x))

                if has(self.event_handlers, TrainerEvents.POST_PROCESS.value):
                    y_hat = get(self.event_handlers, TrainerEvents.POST_PROCESS.value)(x, y, extras, y_hat)

                loss = None
                if has(self.event_handlers, TrainerEvents.COMPUTE_LOSS.value):
                    loss = get(self.event_handlers, TrainerEvents.COMPUTE_LOSS.value)(x, y, extras, y_hat)
                else:
                    loss = self.loss_fn(y_hat, to_variable(y).long().squeeze())  # Compute losses

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                self.optimizer.step()
                batch_count += 1

                interval_count += len(x)
                all_count += len(x)
                if print_accuracy:
                    match_results = None
                    if has(self.event_handlers, TrainerEvents.MATCH_RESULTS.value):
                        match_results = get(self.event_handlers, TrainerEvents.MATCH_RESULTS.value)(x, y, extras, y_hat)
                    else:
                        match_results = self.match(y_hat, y)  # Compute losses

                    correct = match_results.sum()
                    interval_correct += correct
                    all_correct += correct

                if batch_count % print_interval == 0:
                    curr_time = time.time()
                    batch_time = curr_time - t1
                    total_time = curr_time - t0
                    t1 = time.time()

                    percentage = (interval_correct / max(interval_count, 1)) * 100

                    template = None
                    if print_accuracy:
                        template = 'Done training epoch {0} batch {1} count {2}. Time elapsed: {3:.2f} | {4:.2f} seconds. Accuracy: {5:.2f} %. Loss: {6:.12f}.'
                    else:
                        template = 'Done training epoch {0} batch {1} count {2}. Time elapsed: {3:.2f} | {4:.2f} seconds. Loss: {6:.12f}.'

                    p(template.format(self.epoch_ran + 1, batch_count, all_count, \
                        batch_time, total_time, percentage, np.asscalar(losses[-1])))
                    interval_correct, interval_count = 0, 0

            self.epoch_ran += 1
            percentage = all_correct / max(all_count, 1) * 100
            if percentage == 0:
                percentage = None
            loss = np.asscalar(np.mean(losses))

            self.save_model(percentage, loss)

        return self

    def validate(self):
        return self.test(DEV)

    def test(self, data_type=TEST):
        print_interval = get(self.cfg, TrainerOptions.PRINT_INVERVAL.value, default=100)
        print_accuracy = get(self.cfg, TrainerOptions.PRINT_ACCURACY.value, default=True)

        losses, batch_count = [], 0
        interval_correct, interval_count = 0, 0
        all_correct, all_count = 0, 0
        results = []
        t0 = t1 = time.time()

        dataloader = self.__get_dataloader(TEST)
        for batch in dataloader:
            x, y, extras = batch[0], batch[1], batch[2:]
            self.optimizer.zero_grad()

            if has(self.event_handlers, TrainerEvents.PRE_PROCESS.value):
                x, y, extras = get(self.event_handlers, TrainerEvents.PRE_PROCESS.value)(x, y, extras)

            y_hat = None
            self.model.eval()
            if has(self.event_handlers, TrainerEvents.MODEL_EXTRA_ARGS.value):
                args, kwargs = get(self.event_handlers, TrainerEvents.MODEL_EXTRA_ARGS.value)(x, y, extras)
                y_hat = self.model(to_variable(x), *args, **kwargs)
            else:
                y_hat = self.model(to_variable(x))

            if has(self.event_handlers, TrainerEvents.POST_PROCESS.value):
                y_hat = get(self.event_handlers, TrainerEvents.POST_PROCESS.value)(x, y, extras, y_hat)

            interval_count += len(x)
            all_count += len(x)
            if data_type == TEST:
                result = None
                if has(self.event_handlers, TrainerEvents.POST_TEST.value):
                    y_hat, result = get(self.event_handlers, TrainerEvents.POST_TEST.value)(x, y, extras, y_hat)
                else:
                    labels_axis = get(self.cfg, TrainerOptions.LABELS_AXIS.value, default=1)
                    result = predictions.data.max(1, keepdim=True)[1].cpu().numpy().flatten()
                results += list(result)

                if batch_count % print_interval == 0:
                    curr_time = time.time()
                    batch_time = curr_time - t1
                    total_time = curr_time - t0
                    t1 = time.time()

                    template = 'Done testing batch {} count {}. Time elapsed: {:.2f} | {:.2f} seconds.'
                    p(template.format(batch_count, total_data, batch_time, total_time))
            else:
                if print_accuracy:
                    match_results = None
                    if has(self.event_handlers, TrainerEvents.MATCH_RESULTS.value):
                        match_results = get(self.event_handlers, TrainerEvents.MATCH_RESULTS.value)(x, y, extras, y_hat)
                    else:
                        match_results = self.match(y_hat, y)  # Compute losses

                    correct = match_results.sum()
                    interval_correct += correct
                    all_correct += correct

                if batch_count % print_interval == 0:
                    curr_time = time.time()
                    batch_time = curr_time - t1
                    total_time = curr_time - t0
                    t1 = time.time()

                    percentage = (interval_correct / max(interval_count, 1)) * 100

                    template = None
                    if print_accuracy:
                        template = 'Done training epoch {0} batch {1} count {2}. Time elapsed: {3:.2f} | {4:.2f} seconds. Accuracy: {5:.2f} %. Loss: {6:.12f}.'
                    else:
                        template = 'Done training epoch {0} batch {1} count {2}. Time elapsed: {3:.2f} | {4:.2f} seconds. Loss: {6:.12f}.'

                    p(template.format(self.epoch_ran + 1, batch_count, all_count, \
                        batch_time, total_time, percentage, np.asscalar(losses[-1])))
                    interval_correct, interval_count = 0, 0

            if data_type == TEST:
                # TODO: print to CSV
                pass
            else:
                percentage = all_correct / max(all_count, 1) * 100
                if percentage == 0:
                    percentage = None
                p('{0} / {1} ({2:.2f} %) correct!'.format(all_correct, max(all_count, 1), percentage))

        return self
