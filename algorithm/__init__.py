import numpy as np
from mxnet import nd
from utils import translate_state


class AbstractAlgorithm(object):
    def __init__(self, models, action_max, ctx):
        self.models = models
        self.ctx = ctx
        self.action_max = action_max
        self.online = models[0]
        self.offline = models[1]

    def get_action(self, state, poss):
        # epsilon greedy policy
        # with probability select a random action
        if np.random.random() < poss:
            by = "Random"
            action = np.random.randint(0, self.action_max)
        else:
            by = "Model"
            state = nd.array([translate_state(state)])
            state = state.as_in_context(self.ctx)
            action = self.offline(state)
            action = int(nd.argmax(action, axis=1).asnumpy()[0])
        return action, by

    def train(self):
        raise NotImplementedError
