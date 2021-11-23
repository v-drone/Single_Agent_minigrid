from mxnet.gluon import nn, rnn
from mxnet import nd


class ConvBlock(nn.Sequential):
    def __init__(self, channels=256, kernel_size=1):
        super().__init__()
        self.add(nn.Conv2D(channels, kernel_size=kernel_size, use_bias=False, layout="NCHW"))
        self.add(nn.Activation('relu'))


class ViewBlock(nn.Sequential):
    def __init__(self):
        super(ViewBlock, self).__init__()
        with self.name_scope():
            c = [32, 32, 32]
            k = [2, 2, 2]
            for i in range(len(k)):
                self.add(nn.Conv2D(c[i], k[i], use_bias=False, layout="NCHW"))
            for i in [128]:
                self.add(nn.Dense(i, "sigmoid"))


class MapBlock(nn.Sequential):
    def __init__(self, c, k):
        super(MapBlock, self).__init__()
        with self.name_scope():
            for i, j in zip(c, k):
                self.add(nn.Conv2D(i, j, use_bias=False, layout="NCHW"))
                self.add(nn.ELU())


class SimpleStack(nn.Block):
    def __init__(self):
        self.memory_size = 10
        self.map_size = 20
        self.c = [32, 32, 32, 32, 32]
        self.k = [3, 2, 2, 2, 2]
        _hidden = (((self.memory_size - (self.k[0] - 1)) // 2) - sum([i - 1 for i in self.k[:-1]]))
        _hidden = _hidden * _hidden * self.c[-1]
        super(SimpleStack, self).__init__()
        with self.name_scope():
            # self.view = ViewBlock()
            self.map = MapBlock(self.c, self.k)
            self.memory = MapBlock(self.c, self.k)
            # self.LSTM = rnn.LSTM(_hidden, self.memory_size)
            self.decision_making = nn.Sequential()
            self.decision_making.add(nn.Dense(1024, activation='relu'))
            self.decision_making.add(nn.Dense(3))

    def forward(self, view, grid_data, grid_memory, battery, hidden=None, *args):
        battery = nd.expand_dims(battery, axis=1)
        # view = self.view(view)
        # revert memory
        grid_memory = 1 - grid_memory
        grid_data = self.map(grid_data)
        grid_memory = self.memory(grid_memory)
        embedding = grid_data * grid_memory
        # decision layer
        result = self.decision_making(embedding)
        return result
