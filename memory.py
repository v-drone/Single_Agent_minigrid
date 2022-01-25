import random
from collections import namedtuple


class Memory(object):
    def __init__(self, memory_length=2048, frame_len=4, memory=None):
        """
        dataset in mxnet case
        :param memory_length: int
        memory_length
        memory_size of ('state', 'action'', 'reward','finish'， 'battery', 'initial)
        """
        if memory is None:
            memory = []
        self.memory_length = memory_length
        self.memory = memory
        self.frame_len = frame_len
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.memory_length:
            self.memory.append(None)
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'finish', 'battery', 'initial'))
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.memory_length

    def sample(self, bz, state, state_next, reward, action, done, battery):
        ctx = state.context
        for i in range(bz):
            j = random.randint(self.frame_len - 1, len(self.memory) - 2)
            for x in range(self.frame_len):
                _state = self.memory[j - x].state[0].as_in_context(ctx).astype('float32') / 255.
                _state_next = self.memory[j - x + 1].state[0].as_in_context(ctx).astype('float32') / 255.
                state[i, self.frame_len - 1 - x] = _state
                state_next[i, self.frame_len - 1 - x] = _state_next
                if self.memory[j - x].initial:
                    for y in range(self.frame_len - x - 1):
                        _state = self.memory[j - x].state[0].as_in_context(ctx).astype('float32') / 255.
                        _state_next = self.memory[j - x].state[0].as_in_context(ctx).astype('float32') / 255.
                        state[i, self.frame_len - 2 - y] = _state
                        state_next[i, self.frame_len - 2 - y] = _state_next
                    break
            if self.memory[j].finish:
                state_next[i, self.frame_len - 1] = state[i, self.frame_len - 1]
            reward[i] = self.memory[j].reward
            action[i] = self.memory[j].action
            done[i] = self.memory[j].finish
            battery[i] = self.memory[j].battery
