from mxnet.gluon import nn
from .Map import AgentView, WholeView
from mxnet import nd


class Stack(nn.Block):
    def __init__(self):
        """
        MLP in mxnet with input by Lidar and Distance sensor
        """
        super(Stack, self).__init__()
        with self.name_scope():
            self.agent_view = AgentView()
            self.whole_view = WholeView()
            self.decision_making = nn.Sequential()
            self.decision_making.add(nn.Dense(12, activation="relu"))
            self.decision_making.add(nn.BatchNorm())
            self.decision_making.add(nn.Dense(3, activation="relu"))

    def forward(self, income, *args):
        agent_in = income[:, 0:225].reshape(-1, 1, 15, 15)
        whole_map_in = income[:, 225:450].reshape(-1, 1, 15, 15)
        location_in = income[:, 450: 452]
        attitude_in = income[:, -1]
        # relative angle, distance to goal, distance sensor result
        all_features = [agent_in.flatten(), whole_map_in.flatten(), location_in.flatten(), attitude_in.flatten()]
        agent_view = self.agent_view(agent_in).flatten()
        whole_map = self.whole_view(whole_map_in).flatten()
        all_features.append(agent_view)
        all_features.append(whole_map)
        return self.decision_making(nd.concat(*all_features))
