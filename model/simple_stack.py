from mxnet.gluon import nn, rnn
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
        super(SimpleStack, self).__init__()
        with self.name_scope():
            self.map = mobilenet_v3_small(name_prefix="map").features[0:18]
            self.LSTM = rnn.LSTM(577, self.memory_size, bidirectional=False)
            # self.embedding = mobilenet_v3_small(name_prefix="embedding").features[14:18]
            self.out = nn.Sequential()
            self.out.add(nn.Dense(3))
            self.out.add(nn.BatchNorm())
            self.out.add(nn.PReLU())

    def begin_state(self, batch_size=1):
        return self.LSTM.begin_state(batch_size)

    def random_state(self, batch_size=1):
        return self.LSTM.begin_state(batch_size, func=nd.random_normal)

    def forward(self, income, hidden, *args):
        # _view, _map, _memory, _battery = income
        _memory, _battery = income
        _battery = nd.expand_dims(_battery, axis=1)
        _b, _m, _h, _w, _c = _memory.shape
        _embedding = self.map(_memory.reshape([_b * _m, _h, _w, _c]).transpose((0, 3, 1, 2))).flatten()
        # image part
        _embedding = _embedding.reshape(_b, _m, -1).flatten()
        # battery part
        _battery = _battery.transpose([0, 2, 1]).flatten()
        _embedding = nd.concat(_embedding, _battery).reshape(_b, _m, -1)
        _output, _hidden = self.LSTM(_embedding.transpose([1, 0, 2]), states=hidden)
        return self.out(_output.transpose([1, 0, 2])), _hidden
