import torch
from torch import nn
from .helpers import *
from .options import *


class NeuralNetwork(torch.nn.Module):

    def __init__(self, cfg):
        super(NeuralNetwork, self).__init__()
        self.cfg = cfg

        layers = get(self.cfg, NeuralNetworkOptions.LAYERS.value, default=[])
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, data):
        for layer in self.layers:
            data = layer(data)
        return data
