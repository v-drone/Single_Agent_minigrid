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
        super(SimpleStack, self).__init__()
        with self.name_scope():
            self.view = nn.Sequential()
            c = [256, 128, 128]
            k = [1, 2, 2]
            for i in range(len(k)):
                self.view.add(nn.Conv2D(c[i], k[i], use_bias=False,  layout="NCHW"))
            c = [256, 256, 128, 128, 128, 128]
            k = [1, 3, 3, 3, 3, 3]
            # self.map = nn.Sequential()
            # for i in range(len(k)):
            #     self.map.add(nn.Conv2D(c[i], k[i], use_bias=False,layout="NCHW"))
            self.decision_making = nn.Sequential()
            self.decision_making.add(nn.Dense(1024, "sigmoid"))
            self.decision_making.add(nn.Dense(64, "sigmoid"))
            self.decision_making.add(nn.Dense(3, "sigmoid"))
        self.agent_view = None
        self.whole_map = None

    def forward(self, income, *args):
        view, whole_map, attitude = income
        view = self.view(view).flatten()
        # whole_map = self.map(whole_map).flatten()
        # relative angle, distance to goal, distance sensor result
        # all_features = [view, whole_map, attitude]
        all_features = [view, attitude]
        all_features = nd.concat(*all_features)
        result = self.decision_making(all_features)
        return result
