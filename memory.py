import numpy as np
from mxnet import nd
from collections import namedtuple


class Memory(object):
    def __init__(self, memory_length=2048, memory=None):
        """
        dataset in mxnet case
        :param memory_length: int
        memory_length
        memory_size of ('state', 'action', 'next_state', 'reward','finish'， 'battery')
        """
        if memory is None:
            memory = []
        self.memory_length = memory_length
        self.memory = memory
        self.position = 0

    def push(self, old, action, new, reward, finish, battery):
        _ = {"state": old, "state_next": new, "action": action, "reward": reward, "finish": finish, "battery": battery}
        if len(self.memory) < self.memory_length:
            self.memory.append(None)
        self.memory[self.position] = _
        self.position = (self.position + 1) % self.memory_length

    def sample(self, bz, ctx):
        index = np.random.choice(len(self.memory), bz)
        memory = [self.memory[i] for i in index]
        state = nd.concat(*[nd.expand_dims(i["state"], 0) for i in memory], dim=0).as_in_context(ctx)
        state_next = nd.concat(*[nd.expand_dims(i["state_next"], 0) for i in memory], dim=0).as_in_context(ctx)
        action = nd.array([i.get("action") for i in memory], ctx)
        finish = nd.array([int(i.get("finish")) for i in memory], ctx)
        reward = nd.array([int(i.get("reward")) for i in memory], ctx)
        battery = nd.array([int(i.get("battery")) for i in memory], ctx)
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'battery'))
        return Transition(state, action, state_next, reward, finish, battery)
