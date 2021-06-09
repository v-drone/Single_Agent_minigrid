from mxnet.gluon import nn
from mxnet import nd


class SimpleStack(nn.Block):
    def __init__(self, agent_view, whole_map):
        """
        MLP in mxnet with input by Lidar and Distance sensor
        """
        super(SimpleStack, self).__init__()
        with self.name_scope():
            self.view_decode = nn.Sequential()
            self.view_decode.add(nn.Dense(64, activation="tanh"))
            self.view_decode.add(nn.Dense(64, activation="tanh"))
            self.view_decode.add(nn.Dense(64, activation="tanh"))
            self.view_decode.add(nn.Dense(32, activation="tanh"))
            self.view_decode.add(nn.Dense(16, activation="tanh"))
            self.map_decode = nn.Sequential()
            self.map_decode.add(nn.Dense(512, activation="tanh"))
            self.map_decode.add(nn.Dense(256, activation="tanh"))
            self.map_decode.add(nn.Dense(128, activation="tanh"))
            self.map_decode.add(nn.Dense(64, activation="tanh"))
            self.map_decode.add(nn.Dense(32, activation="tanh"))
            self.map_decode.add(nn.Dense(16, activation="tanh"))
            self.decision_making = nn.Sequential()
            self.decision_making.add(nn.Dense(8, activation="tanh"))
            self.decision_making.add(nn.Dense(3, activation="sigmoid"))
        self.agent_view = agent_view
        self.whole_map = whole_map

    def forward(self, income, *args):
        view, whole_map, attitude = income
        view = self.view_decode(view).flatten()
        whole_map = self.map_decode(whole_map).flatten()
        # relative angle, distance to goal, distance sensor result
        all_features = [view, whole_map, attitude]
        all_features = nd.concat(*all_features)
        return self.decision_making(all_features)
