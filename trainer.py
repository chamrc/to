import torch
import glob
import re
import time
import pprint
from flare import *
from flare.helpers import *
from flare.commands import *
from flare.nn import *
from torch.utils.data import TensorDataset
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.validation import ValidationError
from colored import fg, bg, attr
from scipy import stats
import shlex as lex
import time
import traceback


class DataSet(TensorDataset):
    def __init__(self, cfg, wsj, data_type):
        self.cfg, self.wsj, self.data_type = cfg, wsj, data_type

        p('Loading raw dataset "{}".'.format(wsj.get_type_name(data_type)))
        t0 = time.time()

        self.data, self.labels = wsj[data_type]

        p('Done loading raw data in {:.3} secs.'.format(time.time() - t0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return to_tensor(np.array(self.data[i])), to_tensor(
            np.array(self.labels[i]))


class DataSource():
    def __init__(self, cfg, DataSet=DataSet):
        self.data, self.wsj, self.cfg, self.DataSet = [
            None, None, None
        ], WSJ(cfg), cfg, DataSet

    def get_dataset(self, data_type):
        return self.DataSet(self.cfg, self.wsj, data_type)

    def get_dataloader(self, data_type, dataset):
        should_shuffle = data_type is not TEST
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=should_shuffle)

    def get_data(self, data_type):
        if self.data[data_type] is None:
            dataloadaer = None
            should_shuffle = data_type is not TEST

            dataset = self.get_dataset(data_type)
            dataloader = self.get_dataloader(data_type, dataset)

            self.data[data_type] = (dataset, dataloader)
        return self.data[data_type]


class Trainer():
    def __init__(self,
                 DataSource=DataSource,
                 DataSet=DataSet,
                 Model=NeuralNetwork):
        self.commands = [
            'list', 'help', 'use', 'load', 'run', 'test', 'validate', 'set',
            'vote'
        ]
        self.DataSource, self.DataSet, self.Model = DataSource, DataSet, Model
        self.epoch_ran = 0

        if torch.cuda.is_available():
            p('GPU Enabled.')

        self.init_configuration()
        self.init()

    def init(self):
        self.data_source = self.DataSource(self.cfg, self.DataSet)
        self.cnn = self.Model(self.cfg).apply(init_xavier)
        self.optim = self.cfg.optim(self.cnn.parameters(),
                                    **self.cfg.optim_args)
        self.loss_fn = self.cfg.loss_fn()
        self.epoch_ran = 0

        if torch.cuda.is_available():
            self.cnn = self.cnn.cuda()
            self.loss_fn = self.loss_fn.cuda()

    def reset(self):
        p('Reconfiguring network using new setting.')
        epoch = self.epoch_ran
        self.init()
        self.load_model(epoch if epoch else None)

    def init_configuration(self):
        name, cwd = None, os.getcwd()

        if len(sys.argv) >= 2 and os.path.isfile('configurations/{}.py'.format(
                sys.argv[1])):
            self.load_configuration(sys.argv[1])
        elif os.path.isfile('configurations/default.py'):
            self.load_configuration('default')
        else:
            raise Exception(
                'No configuration files found under ./configurations')

        try:
            cfg.epoch = sys.argv[2]
        except Exception as e:
            pass

        if not hasattr(self, 'cfg'):
            raise Exception('Cannot load configuration file')

    def dataset(self, data_type):
        return self.data_source.get_data(data_type)[0]

    def dataloader(self, data_type):
        return self.data_source.get_data(data_type)[1]

    def has_configuration(self, config_path):
        if not config_path.startswith('configurations/'):
            config_path = 'configurations/' + config_path
        return os.path.isfile(config_path)

    def has_version(self, epoch):
        version, path = self.get_version(epoch=epoch)
        return version > 0 and version == epoch

    def get_all_versions(self, prefix=None, greater_than=0):
        pattern, prefix = None, str(prefix)
        pattern = 'models/{}/{}*.model'.format(self.cfg.name,
                                               self.cfg.config_name)

        try:
            greater_than = int(greater_than)
        except Exception as e:
            greater_than = 0

        paths = glob.glob(pattern)
        versions = [int(re.findall(' \d{3} |$', f)[0]) for f in paths]

        results = [
            (v, p)
            for v, p in sorted(zip(versions, paths), key=lambda pair: pair[0])
        ]

        if prefix is not None and (prefix != '' or greater_than != 0):
            filtered = []
            for pair in results:
                if str(pair[0]).startswith(prefix) and pair[0] > greater_than:
                    filtered.append(pair)
            results = filtered

        return results

    def get_version(self, epoch=None, pattern=None):
        version, path = 0, None
        if pattern is not None:
            paths = glob.glob(pattern)

            if len(paths):
                versions = [int(re.findall(' \d{3} |$', f)[0]) for f in paths]
                latest = max(versions)
                i = versions.index(latest)
                if latest > 0 and has(paths, i) and os.path.isfile(paths[i]):
                    version, path = latest, paths[i]
        else:
            if epoch is not None:
                pattern = 'models/{}/{} - {:03d}*.model'.format(
                    self.cfg.name, self.cfg.config_name, int(epoch))
                version, path = self.get_version(pattern=pattern)
                if version > 0:
                    return version, path

            pattern = 'models/{}/{}*.model'.format(self.cfg.name,
                                                   self.cfg.config_name)
            version, path = self.get_version(pattern=pattern)

        return version, path

    def match(self, y_hat, y):
        predictions = y_hat.data.max(1, keepdim=True)[1]
        expectations = y.long()

        if torch.cuda.is_available():
            return predictions.eq(expectations.cuda())
        else:
            return predictions.cpu().eq(expectations)

    def load_configuration(self, path):
        if not path.startswith('configurations/'):
            path = 'configurations/' + path
        if not path.endswith('.py'):
            path = path + '.py'

        module = path.replace('/', '.').replace('.py', '')

        p('Loading configuration file at "{}"'.format(path))
        try:
            self.cfg = importlib.import_module(module)
            self.cfg.name = sys.argv[0].replace('.py', '')
            self.cfg.config_name = module.replace('configurations.', '', 1)
            self.cfg.config_path = path

            self.reset()
        except Exception as e:
            raise e

    def adjust_learning_rate(self, new_lr):
        for param_group in self.optim.param_groups:
            param_group['lr'] = new_lr

        self.cfg.optim_args['lr'] = new_lr
        self.cfg.learn_rate = new_lr

    def save_model(self, percentage=0.0, loss_rate=None):
        mkdirp('models/{}'.format(self.cfg.name))
        path = ''
        if loss_rate is not None:
            path = 'models/{}/{} - {:03d} - {:.2f}% - {:.6f}.model'.format(
                self.cfg.name, self.cfg.config_name, self.epoch_ran,
                percentage, loss_rate)
        else:
            path = 'models/{}/{} - {:03d} - {:.2f}%.model'.format(
                self.cfg.name, self.cfg.config_name, self.epoch_ran,
                percentage)
        p('Saving neural network "{}" to disk at "{}"'.format(
            self.cfg.config_name, path))
        torch.save(self.cnn.state_dict(), path)

    def load_model(self, epoch=None):
        if epoch == 0:
            p('Resetting parameters using xavier.')
            self.epoch_ran = 0
            self.cnn.apply(init_xavier)
            return True

        epoch, path = self.get_version(epoch=epoch)

        if epoch > 0 and path is not None and os.path.isfile(path):
            p('Loading neural network "{}" epoch "{}" at "{}"'.format(
                self.cfg.config_name, epoch, path))

            try:
                if torch.cuda.is_available():
                    self.cnn.load_state_dict(torch.load(path))
                else:
                    self.cnn.load_state_dict(
                        torch.load(path, lambda storage, loc: storage))

                if epoch:
                    self.epoch_ran = epoch

                return True
            except Exception as e:
                p('Failed to load network "{}" epoch "{}" at "{}"'.format(
                    self.cfg.config_name, epoch, path))

        return False

    def run(self):
        print()
        print('----------------------------------------------------------')
        print('|                                                        |')
        print('|        Welcome to Flare Neural Network Trainer.        |')
        print('|                                                        |')
        print('----------------------------------------------------------')
        print()

        if self.cfg.auto_reload_saved_model:
            epoch = self.cfg.epoch if hasattr(
                self.cfg, 'epoch') and self.cfg.epoch is not None else None
            self.load_model(epoch)

        mkdirp('.flare')
        touch('.flare/history')

        while True:
            c = prompt(
                '> ',
                history=FileHistory('.flare/history'),
                auto_suggest=AutoSuggestFromHistory(),
                completer=CommandCompleter(self),
                validator=CommandValidator(self))
            try:
                self.process_command(c)
            except Exception as e:
                traceback.print_exc()

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
            self.load_model(0)
        elif command == 'run':
            if len(parts) == 1:
                self.train()
            else:
                self.train(int(parts[1]))
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
                        p('Skipping test because epoch {} cannot be loaded correctly.'.
                          format(locs[0]))
                else:
                    for i in range(*locs):
                        if self.load_model(i):
                            fn()
                        else:
                            p('Skipping test because epoch {} cannot be loaded correctly.'.
                              format(i))
        elif command == 'vote':
            parts = list(filter(None, c.split(' ', 1)))
            if len(parts) == 1:
                files = match_prefix(folder='submissions/', suffix='.csv')
                self.vote(files)
            else:
                files = lex.split(parts[1])
                self.vote(files)

    def list(self):
        color = fg(45)
        parameter = fg(119)
        reset = attr('reset')

        print('Module:\t\t\t{}{}{}'.format(color, self.cfg.name, reset))
        print('Epoch:\t\t\t{}{}{}'.format(color, self.epoch_ran, reset))
        print('Configuration:\t\t{}{}{}'.format(color, self.cfg.config_name,
                                                reset))
        print('Configuration Path:\t{}{}.py{}'.format(
            color, self.cfg.config_path, reset))
        print()
        configs = []
        for k in list(filter(lambda x: not x.startswith('__'), dir(self.cfg))):
            v = getattr(self.cfg, k)
            configs.append((k, v))
        max_key_len = max([len(k) for k, _ in configs])
        pp = pprint.PrettyPrinter(indent=4, compact=True)
        for k, v in configs:
            w('{}{} :    {}'.format(k, ' ' * (max_key_len - len(k)),
                                    parameter))
            w(
                re.sub(
                    '^    ',
                    ' ' * (max_key_len + 10),
                    pp.pformat(v),
                    flags=re.M))
            print(reset)

    def help(self):
        command = fg(45)
        parameter = fg(119)
        sample = fg(105)
        reset = attr('reset')
        indent = '  '
        print(indent + """
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
            """.format(command, parameter, sample, reset).replace('\t\t\t', indent)
              .strip())

    def train(self, epoch=1):
        p('Running trainer for {} {} in "{}" mode'.format(
            epoch, 'epoch' if epoch == 1 else 'epochs', 'dev'
            if self.cfg.dev_mode else 'train'))

        t0 = time.time()

        for e in range(epoch):
            losses, batch_count = [], 0
            correct_count, total_count, total_data = 0, 0, 0
            percentages = []

            data_type = DEV if self.cfg.dev_mode else TRAIN
            loader = self.dataloader(data_type)
            for batch in loader:
                data, labels, extras = batch[0], batch[1], batch[2:]
                self.optim.zero_grad()

                if hasattr(self.data_source, 'preprocess'):
                    data, labels = self.data_source.preprocess(
                        data_type, data, labels, extras)

                self.cnn.train()
                predictions = self.cnn(to_variable(data))

                if hasattr(self.data_source, 'postprocess'):
                    data, labels, predictions = self.data_source.postprocess(
                        data_type, data, labels, extras, predictions)

                # expectations = torch.zeros(len(labels), self.cfg.out_channels)
                # for i in range(len(labels)):
                #   expectations[i][int(labels[i])] = 1

                results = self.match(predictions, labels)
                correct_count += results.sum()
                total_count += len(data)
                total_data += len(data)

                loss = self.loss_fn(
                    predictions,
                    to_variable(labels).long().squeeze())  # Compute losses
                loss.backward()
                losses.append(loss.data.cpu().numpy())

                self.optim.step()

                batch_count += 1

                if batch_count % 100 == 0:
                    t = time.time() - t0
                    percentage = (correct_count / max(total_count, 1)) * 100
                    percentages.append(percentage)
                    p('Done training epoch {0} batch {1} count {2}. Time elapsed: {3:.2f} seconds. \
                        Accuracy: {4:.2f} %. Loss: {5:.12f}.'.
                      format(self.epoch_ran + 1, batch_count, total_data, t,
                             percentage, np.asscalar(losses[-1])))
                    correct_count, total_count = 0, 0

                if hasattr(
                        self.cfg,
                        'break_at_one_batch') and self.cfg.break_at_one_batch:
                    break

            self.epoch_ran += 1
            if correct_count:
                percentage = (correct_count / max(total_count, 1)) * 100
                percentages.append(percentage)

            if hasattr(self.cfg, 'lr_decay_interval') and hasattr(
                    self.cfg, 'lr_decay_rate'):
                interval, rate = self.cfg.lr_decay_interval, self.cfg.lr_decay_rate
                if self.epoch_ran % interval == 0:
                    self.adjust_learning_rate(self.cfg.learn_rate * rate)

            percentages = percentages[-5:]
            loss_rate = np.asscalar(np.mean(losses))

            self.save_model(sum(percentages) / len(percentages), loss_rate)
            p("Epoch {} Loss: {:.12f}".format(self.epoch_ran, loss_rate))

    def set(self, key, val):
        p('Setting configuration key "{}" to "{}"'.format(key, val))
        if key == 'learn_rate':
            self.adjust_learning_rate(num(val))
        else:
            cmd = 'self.cfg.{} = {}'.format(key, val)
            try:
                exec(cmd)
                self.reset()
            except Exception as e:
                p('Failed to set configuration key "{}" to "{}"'.format(
                    key, val))

    def test(self, data_type=TEST):
        correct_count, total_count, total_data, batch_count, results, t0 = 0, 0, 0, 0, [], time.time()

        loader = self.dataloader(data_type)
        for batch in loader:
            data, labels, extras = batch[0], batch[1], batch[2:]
            batch_count += 1

            if hasattr(self.data_source, 'preprocess'):
                data, labels = self.data_source.preprocess(
                    data_type, data, labels, extras)

            self.cnn.eval()
            predictions = self.cnn(to_variable(data))

            if hasattr(self.data_source, 'postprocess'):
                data, labels, predictions = self.data_source.postprocess(
                    data_type, data, labels, extras, predictions)

            if data_type == TEST:
                result = predictions.data.max(
                    1, keepdim=True)[1].cpu().numpy().flatten()
                results += list(result)
                total_data += len(result)

                if batch_count % 10000 == 0:
                    t = time.time() - t0
                    p('Done testing batch {0} count {1}. Time elapsed: {2:.2f} seconds.'.
                      format(batch_count, total_data, t))
            else:
                result = self.match(predictions, labels)

                correct_count += result.sum()
                total_count += len(data)
                total_data += len(data)
                results = (correct_count, total_count)

                if batch_count % 100 == 0:
                    t = time.time() - t0
                    p('Done validating batch {0} count {1}. Time elapsed: {2:.2f} seconds. Accuracy: {3:.2f} %.'.
                      format(batch_count, total_data, t,
                             (correct_count / total_count) * 100))

            if hasattr(self.cfg,
                       'break_at_one_batch') and self.cfg.break_at_one_batch:
                break

        if data_type == TEST:
            if hasattr(self.data_source, 'posttest'):
                results = self.data_source.posttest(results)

            mkdirp('submissions/{}'.format(self.cfg.name))
            path = 'submissions/{}/{} - {:03d}.csv'.format(
                self.cfg.name, self.cfg.config_name, self.epoch_ran)
            write_to_csv(results, path)
            p('Test data saved to "{}"'.format(path))
        else:
            p('{0} / {1} ({2:.2f} %) correct!'.format(
                correct_count, total_count,
                100 * (correct_count / total_count)))

        return results

    def validate(self):
        return self.test(DEV)

    def vote(self, files):
        count, data = -1, []

        p('Voting for most common predictions.')

        for f in files:
            if f == 'mode.csv':
                p('Skipping submission/mode.csv.')
                continue

            f = re.sub('^"', '', re.sub('"$', '', f))
            path = os.path.join('submissions/', f)
            file_count, file_data = read_from_csv(path)

            if count < 0:
                count = file_count
            if count != file_count:
                raise ValidationError(
                    'vote: Count for submission "{}" didn\'t match {}.'.format(
                        f, count))

            data.append(file_data)

        mode = stats.mode(np.array(data), axis=0)
        output_path = 'submissions/mode.csv'
        try:
            write_to_csv(mode[0][0], output_path)
            p('Successfully wrote voted results to "{}"'.format(output_path))
        except Exception as e:
            p('Failed to write voted results to "{}"'.format(output_path))
