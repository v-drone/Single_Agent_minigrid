import random
from collections import namedtuple


class Memory(object):
    def __init__(self, memory_length=2048, frame_len=4, channel=3):
        """
        dataset in mxnet case
        :param memory_length: int
        memory_length
        memory_size of ('state', 'state_next', 'action'', 'reward','finish'， 'battery', 'initial)
        """
        self.memory = []
        self.memory_length = memory_length
        self.frame_len = frame_len
        self.channel = channel
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.memory_length:
            self.memory.append(None)
        Transition = namedtuple('Transition', ('state', 'state_next', 'action', 'reward', 'finish', 'battery'))
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.memory_length

    def sample(self):
        return self.memory[random.randint(0, len(self.memory) -1)]
