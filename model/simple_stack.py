from mxnet.gluon import nn
from mxnet import nd
from gluoncv.model_zoo.mobilenetv3 import mobilenet_v3_small


class ConvBlock(nn.Sequential):
    def __init__(self, c, k):
        super(ConvBlock, self).__init__()
        with self.name_scope():
            self.add(nn.Conv2D(c, k, use_bias=False, layout="NCHW"))
            self.add(nn.BatchNorm())


class LBBlock(nn.Sequential):
    def __init__(self, c, k, t):
        super(LBBlock, self).__init__()
        with self.name_scope():
            self.add(nn.Conv2D(c * t, 1, use_bias=False, layout="NCHW"))
            self.add(nn.BatchNorm())
            self.add(nn.Activation("relu"))
            self.add(nn.Conv2D(c * t, k, use_bias=False, layout="NCHW"))
            self.add(nn.BatchNorm())
            self.add(nn.Activation("relu"))
            self.add(nn.Conv2D(c, 1, use_bias=False, layout="NCHW"))
            self.add(nn.BatchNorm())


class MapBlock(nn.Sequential):
    def __init__(self):
        t = 3
        super(MapBlock, self).__init__()
        c = [24, 32, 32, 64, 64, 128, 128]
        k = [3, 3, 3, 3, 3, 3, 3]
        with self.name_scope():
            self.add(nn.Conv2D(32, 3, use_bias=False, layout="NCHW"))
            self.add(nn.BatchNorm())
            self.add(nn.Activation("relu"))
            self.add(nn.Conv2D(1, 1, use_bias=False, layout="NCHW"))
            self.add(nn.BatchNorm())
            for i, j in zip(c, k):
                self.add(LBBlock(i, j, t))


class SimpleStack(nn.Block):
    def __init__(self):
        self.memory_size = 10
        self.map_size = 20
        # _hidden = (((self.memory_size - (self.k[0] - 1)) // 2) - sum([i - 1 for i in self.k[:-1]]))
        # _hidden = _hidden * _hidden * self.c[-1]
        super(SimpleStack, self).__init__()
        with self.name_scope():
            self.map = mobilenet_v3_small(name_prefix="map").features[0:14]
            self.memory = mobilenet_v3_small(name_prefix="memory").features[0:14]
            # self.LSTM = rnn.LSTM(_hidden, self.memory_size)
            self.embedding = mobilenet_v3_small(name_prefix="embedding").features[14:18]
            self.out = nn.Sequential()
            self.out.add(nn.Dense(64, activation="relu"))
            self.out.add(nn.Dense(3, activation="relu"))

    def forward(self, income, *args):
        _view, _map, _memory, _battery = income
        _battery = nd.expand_dims(_battery, axis=1)
        _map = self.map(nd.transpose(_map, [0, 3, 1, 2]))
        _memory = self.memory(nd.transpose(_memory, [0, 3, 1, 2]))
        embedding = self.embedding(nd.concat(_map, _memory, dim=1))
        return self.out(nd.concat(*[embedding.flatten(), _battery.flatten()]))
