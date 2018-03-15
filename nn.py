import torch
from torch import nn
from flare.layers import *


class NeuralNetwork(torch.nn.Module):
    def __init__(self, cfg):
        super(NeuralNetwork, self).__init__()
        self.cfg = cfg
        self.layers = torch.nn.ModuleList(self.cfg.layers)

    def forward(self, data):
        for layer in self.layers:
            if isinstance(layer, SpatialPyramidPoolingLayer):
                data = layer(x, x.shape[2:])
            else:
                data = layer(data)
        return data
