import random
from mxnet import nd
from collections import namedtuple


class Memory(object):
    def __init__(self, memory_length=2048, frame_len=4, channel=3):
        """
        dataset in mxnet case
        :param memory_length: int
        memory_length
        memory_size of ('state', 'action'', 'reward','finish'， 'battery', 'initial)
        """
        self.memory = []
        self.memory_length = memory_length
        self.frame_len = frame_len
        self.channel = channel
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.memory_length:
            self.memory.append(None)
        Transition = namedtuple('Transition', ('state', 'action', 'state_next', 'reward', 'finish', 'battery'))
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.memory_length

    def sample(self, size, ctx):
        Transition = namedtuple('Transition', ('state', 'action', 'state_next', 'reward', 'finish', 'battery'))
        data = random.sample(self.memory, size)
        # if len(set([i.state.shape for i in data])) > 1:
        #     raise
        # if len(set([i.state_next.shape for i in data])) > 1:
        #     raise
        state = nd.concat(*[nd.expand_dims(i.state, 0) for i in data], dim=0).as_in_context(ctx)
        state_next = nd.concat(*[nd.expand_dims(i.state_next, 0) for i in data], dim=0).as_in_context(ctx)
        finish = nd.concat(*[nd.array([i.finish]) for i in data], dim=0).as_in_context(ctx)
        reward = nd.concat(*[nd.array([i.reward]) for i in data], dim=0).as_in_context(ctx)
        action = nd.concat(*[nd.array([i.action]) for i in data], dim=0).as_in_context(ctx)
        battery = nd.concat(*[nd.array([i.battery]) for i in data], dim=0).as_in_context(ctx)
        return Transition(*[state, action, state_next, reward, finish, battery])
