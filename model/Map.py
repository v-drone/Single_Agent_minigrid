from mxnet.gluon import nn
from .convolution import _conv2d, DetectionBlock


class AgentView(nn.HybridBlock):
    def __init__(self):
        """
        MLP in mxnet with input by Lidar and Distance sensor
        """
        super(AgentView, self).__init__()
        with self.name_scope():
            layers = [1]
            channels = [32]
            # first 3x3 conv
            self.features = nn.HybridSequential()
            self.features.add(_conv2d(10, 3, 1, 1))
            for n_layer, channel in zip(layers, channels):
                self.features.add(_conv2d(channel, 3, 1, 1))
                for _ in range(n_layer):
                    self.features.add(DetectionBlock(channel // 2))

    def hybrid_forward(self, F, x):
        return self.features(x)


class WholeView(nn.HybridBlock):
    def __init__(self):
        """
        MLP in mxnet with input by Lidar and Distance sensor
        """
        super(WholeView, self).__init__()
        with self.name_scope():
            layers = [1, 2]
            channels = [32, 128]
            # first 3x3 conv
            self.features = nn.HybridSequential()
            self.features.add(_conv2d(10, 3, 1, 1))
            for n_layer, channel in zip(layers, channels):
                self.features.add(_conv2d(channel, 3, 1, 2))
                for _ in range(n_layer):
                    self.features.add(DetectionBlock(channel // 2))

    def hybrid_forward(self, F, x):
        return self.features(x)
