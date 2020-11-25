from .DQN import DQN
import numpy as np
from mxnet import nd


class AbstractModel(object):
    def __init__(self, action_space, models, ctx):
        self.models = models
        self.ctx = ctx
        self.action_space = action_space
        pass

    def get_action(self, state, poss):
        if np.random.random() < poss:
            by = "Random"
            action = np.random.randint(0, self.action_space)
        else:
            by = "Model"
            action = self.models[-1](nd.array([np.concatenate(state)], ctx=self.ctx))
            action = int(nd.argmax(action, axis=1).asnumpy()[0])
        return action, by

    def train(self):
        raise NotImplementedError
