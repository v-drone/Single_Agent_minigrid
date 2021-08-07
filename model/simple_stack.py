from mxnet.gluon import nn
from mxnet import nd


class ConvBlock(nn.Sequential):
    def __init__(self, channels=256, kernel_size=1):
        super().__init__()
        self.add(nn.Conv2D(channels, kernel_size=kernel_size,
                           use_bias=False, layout="NCHW"))
        self.add(nn.BatchNorm())
        self.add(nn.Activation('softrelu'))


class SimpleStack(nn.Block):
    def __init__(self):
        """
        MLP in mxnet with input by Lidar and Distance sensor
        """
        super(SimpleStack, self).__init__()
        with self.name_scope():
            self.view = nn.Sequential()
            self.view.add(
                nn.Conv2D(256, kernel_size=1, use_bias=False, layout="NCHW"))
            self.view.add(
                nn.Conv2D(128, kernel_size=2, use_bias=False, layout="NCHW"))
            self.view.add(
                nn.Conv2D(128, kernel_size=2, use_bias=False, layout="NCHW"))
            self.map = nn.Sequential()
            self.map.add(
                nn.Conv2D(256, kernel_size=1, use_bias=False, layout="NCHW"))
            self.map.add(
                nn.Conv2D(256, kernel_size=3, use_bias=False, layout="NCHW"))
            self.map.add(
                nn.Conv2D(128, kernel_size=3, use_bias=False, layout="NCHW"))
            self.map.add(
                nn.Conv2D(128, kernel_size=3, use_bias=False, layout="NCHW"))
            self.map.add(
                nn.Conv2D(128, kernel_size=3, use_bias=False, layout="NCHW"))
            self.map.add(
                nn.Conv2D(128, kernel_size=3, use_bias=False, layout="NCHW"))
            self.decision_making = nn.Sequential()
            self.decision_making.add(nn.Dense(1024, "sigmoid"))
            self.decision_making.add(nn.Dense(64, "sigmoid"))
            self.decision_making.add(nn.Dense(3, "sigmoid"))

    def forward(self, income, *args):
        view, whole_map, attitude = income
        view = self.view(view).flatten()
        whole_map = self.map(whole_map).flatten()
        # relative angle, distance to goal, distance sensor result
        all_features = [view, whole_map, attitude]
        all_features = nd.concat(*all_features)
        result = self.decision_making(all_features)
        return result
