from mxnet.gluon import nn
from .convolution import _conv2d, DetectionBlock


class MapView(nn.HybridBlock):
    def __init__(self):
        """
        MLP in mxnet with input by Lidar and Distance sensor
        """
        super(MapView, self).__init__()
        with self.name_scope():
            layers = [1, 1, 1]
            channels = [16, 32, 64]
            # first 3x3 conv
            self.features = nn.HybridSequential()
            for n_layer, channel in zip(layers, channels):
                for _ in range(n_layer):
                    self.features.add(_conv2d(channel, 2, 0, 1))
                    self.features.add(nn.MaxPool2D((2, 2), strides=1))
            self.decode = nn.Dense(256)

    def hybrid_forward(self, F, x):
        features = self.features(x)
        return self.decode(features)
