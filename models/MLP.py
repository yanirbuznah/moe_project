import torch
from torch import nn

from . import utils as ut


class MLP(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.output_size = output_size
        self._init_params(config['model_params'])
        self._init_layers()
        # TODO: add save and load policy

    def _init_params(self, config):
        self.output_size = config.get('output_size', self.output_size)
        self.hidden_sizes = config.get('hidden_sizes', [128, 128])
        self.activation = config.get('activation', 'relu')
        self.dropout = config.get('dropout', 0.2)
        self.batch_norm = config.get('batch_norm', False)
        self.bias = config.get('bias', True)
        self.dtype = ut.get_dtype(config.get('dtype', 'float32'))
        self.wight_init = ut.get_weight_initialize_technique(config.get('wight_init', 'xavier'))

    def _init_layers(self):
        self.layers = nn.ModuleList()
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.layers.append(nn.LazyLinear(hidden_size, bias=self.bias, dtype=self.dtype))
            if self.batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(ut.get_activation(self.activation))
            self.layers.append(nn.Dropout(self.dropout))
            self.input_size = hidden_size
        self.layers.append(nn.LazyLinear(self.output_size, bias=self.bias))
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.wight_init(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def reset_parameters(self, input):
        self.forward(input)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.wight_init(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
