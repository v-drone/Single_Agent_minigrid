from mxnet.gluon import nn
from mxnet import nd


class ConvBlock(nn.Sequential):
    def __init__(self, channels=256, kernel_size=1):
        super().__init__()
        self.add(nn.Conv2D(channels, kernel_size=kernel_size, use_bias=False, layout="NCHW"))
        self.add(nn.BatchNorm())
        self.add(nn.Activation('softrelu'))


class ViewBlock(nn.Sequential):
    def __init__(self):
        super(ViewBlock, self).__init__()
        with self.name_scope():
            c = [64, 128, 128]
            k = [1, 2, 2]
            for i in range(len(k)):
                self.add(nn.Conv2D(c[i], k[i], use_bias=False, layout="NCHW"))


class MapBlock(nn.Sequential):
    def __init__(self):
        super(MapBlock, self).__init__()
        with self.name_scope():
            self.add(nn.Conv2D(64, 1, use_bias=False, layout="NCHW"))
            self.add(nn.AvgPool2D(2, 2))
            c = [64, 128, 128]
            k = [2, 2, 2]
            for i in range(len(k)):
                self.add(nn.Conv2D(c[i], k[i], use_bias=False, layout="NCHW"))


class MemoryBlock(nn.Sequential):
    def __init__(self):
        super(MemoryBlock, self).__init__()
        with self.name_scope():
            self.add(nn.Conv2D(64, 1, use_bias=False, layout="NCHW"))
            self.add(nn.AvgPool2D(2, 2))
            c = [64, 128, 128]
            k = [2, 2, 2]
            for i in range(len(k)):
                self.add(nn.Conv2D(c[i], k[i], use_bias=False, layout="NCHW"))


class SimpleStack(nn.Block):
    def __init__(self):
        super(SimpleStack, self).__init__()
        with self.name_scope():
            self.view = ViewBlock()
            self.map = MapBlock()
            self.memory = MemoryBlock()
            self.decision_making = nn.Sequential()
            for i in [64]:
                self.decision_making.add(nn.Dense(i))
            self.decision_making.add(nn.Dense(3))
            self.decision_making.add(nn.LeakyReLU(0.1))

    def forward(self, income, *args):
        _view, _map, battery = income
        _view = self.view(_view)
        _map = nd.transpose(_map, [1, 0, 2, 3])
        _map, _memory = _map
        _map = nd.expand_dims(_map, axis=1)
        _memory = nd.expand_dims(_memory, axis=1)
        _map = self.map(_map)
        _memory = self.memory(_memory)
        _features = [_view.flatten(), _map.flatten(), _memory.flatten(), battery]
        _features = nd.concat(*_features)
        result = self.decision_making(_features)
        return result
