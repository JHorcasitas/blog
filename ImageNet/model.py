from collections import namedtuple

import torch
import torch.nn as nn


ConvLayer = namedtuple('ConvLayer', 'kernel_size number_of_filters stride')
PoolLayer = namedtuple('PoolLayer', 'kernel_size stride')

CONFIGURATION = [
    ConvLayer(7, 64, 2),
    PoolLayer(2, 2),
    ConvLayer(3, 3, 1),
    PoolLayer(2, 2),
    ConvLayer(1, 128, 1),
    ConvLayer(3, 256, 1),
    ConvLayer(1, 256, 1),
    ConvLayer(3, 512, 1),
    PoolLayer(2, 2),
    ConvLayer(1, 256, 1),
    ConvLayer(3, 512, 1),
    ConvLayer(1, 256, 1),
    ConvLayer(3, 512, 1),
    ConvLayer(1, 256, 1),
    ConvLayer(3, 512, 1),
    ConvLayer(1, 256, 1),
    ConvLayer(3, 512, 1),
    ConvLayer(1, 512, 1),
    ConvLayer(3, 1024, 1),
    PoolLayer(2, 2),
    ConvLayer(1, 512, 1),
    ConvLayer(3, 1024, 1),
    ConvLayer(1, 512, 1),
    ConvLayer(3, 1024, 1)]


def model_factory(kind: str) -> nn.Module:
    """Retrieves a model.
    :param kind: One of {'classification', 'detection'}
    """
    if kind == 'classification':
        return ImageNetClassifier()
    elif kind == 'detection':
        raise NotImplementedError()
    else:
        raise ValueError()


class ImageNetClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._base_layers = self.make_layers()
        self._avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=1_024, out_features=1_000, bias=True)

    def forward(self, x):
        out = self._base_layers(x)
        out = self._avg_pool(out)
        out = torch.squeeze(out)
        return self.fc(out)

    def make_layers(self):
        layers = []
        in_channels = 3
        for layer in CONFIGURATION:
            if isinstance(layer, ConvLayer):
                conv_layer = nn.Conv2d(in_channels=in_channels,
                                       out_channels=layer.number_of_filters,
                                       kernel_size=layer.kernel_size,
                                       stride=layer.stride,
                                       padding=((layer.kernel_size - 1) // 2))
                in_channels = layer.number_of_filters
                layers.extend([conv_layer, nn.LeakyReLU()])
            elif isinstance(layer, PoolLayer):
                pool_layer = nn.MaxPool2d(kernel_size=layer.kernel_size,
                                          stride=layer.stride)
                layers.append(pool_layer)
            else:
                raise ValueError()
        return nn.Sequential(*layers)
