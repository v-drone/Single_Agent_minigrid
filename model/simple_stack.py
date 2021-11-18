from mxnet.gluon import nn
from mxnet import nd


class ConvBlock(nn.Sequential):
    def __init__(self, channels=256, kernel_size=1):
        super().__init__()
        self.add(nn.Conv2D(channels, kernel_size=kernel_size, use_bias=False, layout="NCHW"))
        self.add(nn.BatchNorm())
        self.add(nn.Activation('tanh'))


class ViewBlock(nn.Sequential):
    def __init__(self):
        super(ViewBlock, self).__init__()
        with self.name_scope():
            c = [64, 128, 128]
            k = [1, 2, 2]
            for i in range(len(k)):
                self.add(ConvBlock(c[i], k[i]))
            for i in [128]:
                self.add(nn.Dense(i, "tanh"))


class MapBlock(nn.Sequential):
    def __init__(self):
        super(MapBlock, self).__init__()
        with self.name_scope():
            self.add(nn.Conv2D(32, 1, use_bias=False, layout="NCHW"))
            self.add(nn.AvgPool2D(2, 2))
            c = [64, 128, 128, 128]
            k = [2, 2, 2, 2]
            for i in range(len(k)):
                self.add(ConvBlock(c[i], k[i]))
            for i in [128]:
                self.add(nn.Dense(i, "tanh"))


class MemoryBlock(nn.Sequential):
    def __init__(self):
        super(MemoryBlock, self).__init__()
        with self.name_scope():
            self.add(nn.Conv2D(32, 1, use_bias=False, layout="NCHW"))
            self.add(nn.AvgPool2D(2, 2))
            c = [64, 128, 128, 128]
            k = [2, 2, 2, 2]
            for i in range(len(k)):
                self.add(ConvBlock(c[i], k[i]))
            for i in [128]:
                self.add(nn.Dense(i, "tanh"))


class SimpleStack(nn.Block):
    def __init__(self):
        super(SimpleStack, self).__init__()
        with self.name_scope():
            self.view = ViewBlock()
            self.map = MapBlock()
            self.memory = MemoryBlock()
            self.decision_making = nn.Sequential()
            for i in [32]:
                self.decision_making.add(nn.Dense(i))
            self.decision_making.add(nn.Dense(3, "tanh"))

    def forward(self, income, *args):
        _view, _map, _battery = income
        _battery = nd.expand_dims(_battery, axis=1)
        _view = self.view(_view)
        _map = nd.transpose(_map, [1, 0, 2, 3])
        _map, _memory = _map
        _map = nd.one_hot(_map, 7).transpose([0, 3, 1, 2])
        # _map = nd.expand_dims(_map, axis=1)
        _memory = nd.expand_dims(_memory, axis=1)
        _map = self.map(_map)
        _memory = self.memory(_memory)
        _features = [_view.flatten(), _map.flatten(), _memory.flatten(), _battery]
        _features = nd.concat(*_features)
        result = self.decision_making(_features)
        return result
