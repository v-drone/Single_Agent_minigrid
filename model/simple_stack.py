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


class SimpleStack(nn.Block):
    def __init__(self, actions, frames, channel=3):
        self.frames = frames
        self.channel = channel
        super(SimpleStack, self).__init__()
        c = [64, 64]
        k = [4, 3]
        s = [2, 1]
        with self.name_scope():
            self.map = nn.Sequential()
            self.out = nn.Sequential()
            with self.map.name_scope():
                self.map.add(nn.Conv2D(channels=32, kernel_size=8, strides=4, padding=0, layout="NCHW"))
                self.map.add(nn.Activation("tanh"))
                # self.add(nn.MaxPool2D(2, 2))
                self.map.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
                for i, j, z in zip(c, k, s):
                    self.map.add(nn.Conv2D(channels=i, kernel_size=j, strides=z, padding=0, layout="NCHW"))
                    self.map.add(nn.BatchNorm(axis=1, momentum=0.1, center=True))
                    self.map.add(nn.Activation("tanh"))
                    # self.add(nn.MaxPool2D(2, 2))
                self.map.add(nn.Flatten())
            with self.out.name_scope():
                self.out.add(nn.Dense(512, activation="tanh"))
                self.out.add(nn.Dense(actions))

    def forward(self, memory, battery, *args):
        _b, _c, _h, _w = memory.shape
        # image part
        _features = self.map(memory).reshape([_b, self.frames, -1])
        # # battery part
        # _battery = nd.expand_dims(_battery, axis=1)
        # _battery = _battery.transpose([0, 2, 1])
        return self.out(_features)
