from mxnet.gluon import nn
from mxnet import nd


class ConvBlock(nn.Sequential):
    def __init__(self, channels=256, kernel_size=1):
        super().__init__()
        self.add(nn.Conv2D(channels, kernel_size=kernel_size, use_bias=False, layout="NCHW"))
        self.add(nn.BatchNorm())
        self.add(nn.Activation('sigmoid'))


class ViewBlock(nn.Sequential):
    def __init__(self):
        super(ViewBlock, self).__init__()
        with self.name_scope():
            c = [32, 64, 128]
            k = [1, 2, 2]
            for i in range(len(k)):
                self.add(nn.Conv2D(c[i], k[i], use_bias=False, layout="NCHW"))
            for i in [128]:
                self.add(nn.Dense(i, "sigmoid"))


class MapBlock(nn.Sequential):
    def __init__(self):
        super(MapBlock, self).__init__()
        with self.name_scope():
            c = [32, 128]
            k = [1, 2]
            for i, j in zip(c[:-1], k[:-1]):
                self.add(nn.Conv2D(i, j, use_bias=False, layout="NCHW"))
                self.add(nn.AvgPool2D())
            self.add(nn.Conv2D(c[-1], k[-1], use_bias=False, layout="NCHW"))


class MemoryBlock(nn.Sequential):
    def __init__(self):
        super(MemoryBlock, self).__init__()
        with self.name_scope():
            c = [32, 128]
            k = [1, 2]
            for i, j in zip(c[:-1], k[:-1]):
                self.add(nn.Conv2D(i, j, use_bias=False, layout="NCHW"))
                self.add(nn.AvgPool2D())
            self.add(nn.Conv2D(c[-1], k[-1], use_bias=False, layout="NCHW"))


class SimpleStack(nn.Block):
    def __init__(self):
        super(SimpleStack, self).__init__()
        with self.name_scope():
            # self.view = ViewBlock()
            self.map = MapBlock()
            self.memory = MemoryBlock()
            self.decision_making = nn.Sequential()
            self.decision_making.add(nn.Dense(3, "sigmoid"))

    def forward(self, income, *args):
        _view, _map, _battery = income
        _battery = nd.expand_dims(_battery, axis=1)
        # _view = self.view(_view)
        _map = nd.transpose(_map, [1, 0, 2, 3])
        _map, _memory = _map
        _memory = 1 - _memory
        _map = nd.one_hot(_map, 9, 1).transpose([0, 3, 1, 2])
        _memory = nd.expand_dims(_memory, axis=1)
        _map = self.map(_map)
        _memory = self.memory(_memory)
        _features = [_map.flatten(), _memory.flatten()]
        _features = nd.concat(*_features)
        result = self.decision_making(_features)
        return result
