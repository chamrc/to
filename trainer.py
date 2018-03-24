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


class Trainer(object):

    #----------------------------------------------------------------------------------------------------------
    # Initialization
    #----------------------------------------------------------------------------------------------------------

    def __init__(self):
        super(Trainer, self).__init__()

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
        self.epoch_ran = 0
        self.__init_model()
        self.__init_optim()
        self.__init_loss_fn()

    def __init_model(self):
        self.model = self.Model(self.cfg)

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
    # DataSet and DataLoader
    #----------------------------------------------------------------------------------------------------------

    def set_dataloader(self, DataLoader):
        self.DataLoader = DataLoader
        return self

    def set_dataset(self, DataSet):
        self.DataSet = DataSet
        return self

    def __get_dataloader(self, data_type):
        pass

    #----------------------------------------------------------------------------------------------------------
    # Model
    #----------------------------------------------------------------------------------------------------------

    def __adjust_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.cfg.learning_rate = new_lr
        if has(self.cfg, TrainerOptions.OPTIMIZER_ARGS.value, 'lr'):
            self.cfg.optim_args['lr'] = new_lr

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

    def run(self):
        return self

    def validate(self):
        return self.test(DEV)

    def test(self, data_type=TEST):
        return self
