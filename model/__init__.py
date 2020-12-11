from mxnet.gluon import nn
from .Map import MapView
from mxnet import nd


class Stack(nn.Block):
    def __init__(self):
        """
        MLP in mxnet with input by Lidar and Distance sensor
        """
        super(Stack, self).__init__()
        with self.name_scope():
            # self.agent_view = MapView()
            self.whole_view = MapView()
            self.decision_making = nn.Sequential()
            self.decision_making.add(nn.Dense(64, activation="tanh"))
            self.decision_making.add(nn.Dense(3, activation="sigmoid"))

    def forward(self, income, *args):
        # agent_in = income[:, 0:225].reshape(-1, 1, 15, 15)
        whole_map_in = income[:, 225:450].reshape(-1, 1, 15, 15).astype('float32') / 255
        location_in = income[:, 450: 452]
        attitude_in = income[:, -1]
        # relative angle, distance to goal, distance sensor result
        # all_features = [agent_in.flatten(), whole_map_in.flatten(), location_in.flatten(), attitude_in.flatten()]
        all_features = [location_in.flatten(), attitude_in.flatten()]
        # all_features = []
        # agent_view = self.agent_view(agent_in).flatten()
        whole_map = self.whole_view(whole_map_in).flatten()
        all_features.append(whole_map)
        # all_features.append(whole_map)
        return self.decision_making(nd.concat(*all_features))


class SimpleStack(nn.Block):
    def __init__(self):
        """
        MLP in mxnet with input by Lidar and Distance sensor
        """
        super(SimpleStack, self).__init__()
        with self.name_scope():
            self.map_decode = nn.Sequential()
            self.map_decode.add(nn.Dense(64, activation="tanh"))
            self.map_decode.add(nn.Dense(12, activation="tanh"))
            self.decision_making = nn.Sequential()
            self.decision_making.add(nn.Dense(3, activation="sigmoid"))

    def forward(self, income, *args):
        agent_in = income[:, 0:225].reshape(-1, 1, 15, 15)
        whole_map_in = income[:, 225:450].reshape(-1, 1, 15, 15) * 100
        location_in = income[:, 225: 227]
        attitude_in = income[:, -1]
        map_feature = self.map_decode(whole_map_in.flatten())
        # relative angle, distance to goal, distance sensor result
        # all_features = [agent_in.flatten(), whole_map_in.flatten(), location_in.flatten(), attitude_in.flatten()]
        all_features = [map_feature, location_in.flatten(), attitude_in.flatten()]
        all_features = nd.concat(*all_features)
        return self.decision_making(all_features)
