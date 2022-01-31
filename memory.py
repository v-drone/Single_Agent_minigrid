import random
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
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'finish', 'battery', 'initial'))
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.memory_length

    def sample(self, size, state, state_next, reward, action, done, battery):
        ctx = state.context

        def _process(row):
            return row.as_in_context(ctx)

        for i in range(size):
            j = random.randint(self.frame_len - 1, len(self.memory) - 2)
            for jj in range(self.frame_len):
                for jjj in range(self.channel):
                    state[i, self.frame_len - 1 - jj + jjj] = _process(self.memory[j - jj].state)[jjj]
                    state_next[i, self.frame_len - 1 - jj + jjj] = _process(self.memory[j - jj + 1].state)[jjj]
                    if self.memory[j - jj].initial:
                        for kk in range(self.frame_len - jj - 1):
                            state[i, self.frame_len - 2 - kk + jjj] = _process(self.memory[j - jj].state)[jjj]
                            state_next[i, self.frame_len - 2 - kk + jjj] = _process(self.memory[j - jj].state)[jjj]
                        break
            if self.memory[j].finish:
                state_next[i, self.frame_len - 1] = state[i, self.frame_len - 1]
            reward[i] = self.memory[j].reward
            action[i] = self.memory[j].action
            done[i] = self.memory[j].finish
            battery[i] = self.memory[j].battery
