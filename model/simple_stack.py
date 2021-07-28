from mxnet.gluon import nn
from mxnet import nd


class ConvBlock(nn.Sequential):
    def __init__(self, channels=256, kernel_size=1):
        super().__init__()
        self.add(nn.Conv2D(channels, kernel_size=kernel_size, strides=1,
                           use_bias=False, layout="NCHW"))
        self.add(nn.BatchNorm())
        self.add(nn.Activation('softrelu'))


class SimpleStack(nn.Block):
    def __init__(self, agent_view, whole_map, channels=256):
        """
        MLP in mxnet with input by Lidar and Distance sensor
        """
        super(SimpleStack, self).__init__()
        with self.name_scope():
            self.view_decode = nn.Sequential()
            self.view_decode.add(ConvBlock(128, 1))
            self.view_decode.add(ConvBlock(128, 2))
            self.view_decode.add(ConvBlock(512, 1))
            self.map_decode = nn.Sequential()
            self.map_decode.add(ConvBlock(512, 1))
            self.map_decode.add(ConvBlock(256, 3))
            self.map_decode.add(ConvBlock(256, 3))
            self.map_decode.add(ConvBlock(256, 3))
            self.decision_making = nn.Sequential()
            self.decision_making.add(nn.Dense(3, "tanh"))
        self.agent_view = agent_view
        self.whole_map = whole_map

    def forward(self, income, *args):
        view, whole_map, attitude = income
        view = self.view_decode(view).flatten()
        whole_map = self.map_decode(whole_map).flatten()
        # relative angle, distance to goal, distance sensor result
        all_features = [view, whole_map, attitude]
        all_features = nd.concat(*all_features)
        result = self.decision_making(all_features)
        return result
