import torch
import torch.nn as nn
import math


class PackPaddedLayer(torch.nn.Module):

    def __init__(self, batch_first=False):
        super(PackPaddedLayer, self).__init__()
        self.batch_first = batch_first

    def forward(self, h, lengths):
        h = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=self.batch_first)
        return h


class PadPackedLayer(torch.nn.Module):

    def __init__(self, batch_first=False):
        super(PadPackedLayer, self).__init__()
        self.batch_first = batch_first

    def forward(self, h):
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=False)
        return h


class Flatten(torch.nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class MeanPoolingLayer(torch.nn.Module):

    def __init__(self):
        super(MeanPoolingLayer, self).__init__()

    def forward(self, input, dim=2):
        length = input.shape[2]
        return torch.sum(input, dim=2) / length

    def __repr__(self):
        return self.__class__.__name__ + '()'
