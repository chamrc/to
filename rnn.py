import torch.nn as nn
from .nn import *
from .helpers import *


class TextModel(NeuralNetwork):

    def __init__(self, cfg):
        super(TextModel, self).__init__(cfg)

        self.in_features = get(self.cfg, NeuralNetworkOptions.IN_CHANNELS.value, default=40)
        self.out_features = get(self.cfg, NeuralNetworkOptions.OUT_CHANNELS.value, default=47)

    def __forward(self, h, lengths):
        is_RNN = lambda x: isinstance(x, nn.modules.rnn.RNNBase)

        starts = all(lambda i, o, A: i >= 0 and not is_RNN(get(A, i - 1)) and is_RNN(o), self.layers)
        ends = all(lambda i, o, A: i <= len(A) - 1 and is_RNN(o) and not is_RNN(get(A, i + 1)), self.layers)

        should_pack_padded = get(self.cfg, TextModelOptions.PACK_PADDED.value, default=False) and \
            len(starts) >= 0 and len(ends) >= 0

        for i, layer in enumerate(self.layers):
            if is_RNN(layer):
                if should_pack_padded and i in starts:
                    h = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=False)

                layer.flatten_parameters()
                h, _ = layer(h)

                if should_pack_padded and i in ends:
                    h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=False)
            else:
                h = layer(h)

        return h

    def forward(self, inputs, lengths, forward=0, stochastic=False):
        h = inputs.float()  # (n, t)

        h = self.__forward(h, lengths)

        if stochastic:
            gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
            h += gumbel

        logits = h
        if forward > 0:
            outputs = []

            h = torch.max(logits[:, -1:, :], dim=2)[1] + 1
            for i in range(forward):
                h = self.__forward(h)

                if stochastic:
                    gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
                    h += gumbel

                outputs.append(h, lengths)

                h = torch.max(h, dim=2)[1] + 1
            logits = torch.cat([logits] + outputs, dim=1)

        return logits
