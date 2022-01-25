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
    def __init__(self, name_prefix):
        t = 1
        super(MapBlock, self).__init__()
        c = [32, 64, 64]
        k = [8, 4, 3]
        s = [4, 2, 1]
        with self.name_scope():
            for i, j, z in zip(c, k, s):
                self.add(nn.Conv2D(channels=i * t, kernel_size=j, strides=z, padding=0, use_bias=False, layout="NCHW"))
                self.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
                self.add(nn.Activation("relu"))
            self.add(nn.Flatten())


class SimpleStack(nn.Block):
    def __init__(self, actions, frames, channel=3):
        self.frames = frames
        self.channel = channel
        super(SimpleStack, self).__init__()
        with self.name_scope():
            self.map = MapBlock(name_prefix="map")
            self.out = nn.Sequential()
            self.out.add(nn.Dense(512))
            self.out.add(nn.Activation("relu"))
            self.out.add(nn.Dense(actions))
            self.out.add(nn.Activation("relu"))

    def forward(self, income, *args):
        _memory, _battery = income
        _b, _c, _h, _w = _memory.shape
        _memory = _memory.reshape([_b * self.frames, self.channel, _h, _w])
        # image part
        _features = self.map(_memory).reshape([_b, self.frames, -1])
        # # battery part
        # _battery = nd.expand_dims(_battery, axis=1)
        # _battery = _battery.transpose([0, 2, 1])
        # _embedding = nd.concat(_memory, _battery, dim=2)
        return self.out(_features)
