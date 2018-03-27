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


class SpatialPyramidPoolingLayer(torch.nn.Module):

    def __init__(self, out_pool_size, num_sample=1):
        super(SpatialPyramidPoolingLayer, self).__init__()
        self.out_pool_size = out_pool_size
        self.num_sample = 1

    def forward(self, previous_conv, previous_conv_size):
        """
                previous_conv: a tensor vector of previous convolution layer
                num_sample: an int number of image in the batch
                previous_conv_size: an int vector [height, width] of the matrix features size of previous
                convolution layer.
                out_pool_size: a int vector of expected output size of max pooling layer

                returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        """
        for i in range(len(self.out_pool_size)):
            h_wid = int(math.ceil(previous_conv_size[0] / self.out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / self.out_pool_size[i]))
            h_pad = (h_wid * self.out_pool_size[i] - previous_conv_size[0] + 1) / 2
            w_pad = (w_wid * self.out_pool_size[i] - previous_conv_size[1] + 1) / 2

            maxpool = torch.nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)

            if (i == 0):
                spp = x.view(self.num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(self.num_sample, -1)), 1)

        return spp

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MeanPoolingLayer(torch.nn.Module):

    def __init__(self):
        super(MeanPoolingLayer, self).__init__()

    def forward(self, input, dim=2):
        length = input.shape[2]
        return torch.sum(input, dim=2) / length

    def __repr__(self):
        return self.__class__.__name__ + '()'
