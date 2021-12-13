import numpy as np
from utils import create_input, translate_state


class Memory(object):
    def __init__(self, memory_length=2048, memory=None):
        """
        dataset in mxnet case
        :param memory_length: int
        memory_length
        memory_size of ('state', 'action', 'next_state', 'reward','finish')
        """
        if memory is None:
            memory = []
        self.memory_length = memory_length
        self.memory = memory
        self.position = 0

    def add(self, old, new, action, reward, finish):
        _ = {"state": old, "state_next": new, "action": action,
             "reward": reward, "finish": finish}
        if len(self.memory) < self.memory_length:
            self.memory.append(None)
        self.memory[self.position] = _
        self.position = (self.position + 1) % self.memory_length

    def next_batch(self, bz):
        index = np.random.choice(len(self.memory), bz)
        memory = [self.memory[i] for i in index]
        state = [translate_state(i.get("state")) for i in memory]
        state_next = [translate_state(i.get("state_next")) for i in memory]
        hidden = [i["state"]["hidden"] for i in memory]
        hidden_next = [i["state_next"]["hidden"] for i in memory]
        action = [i.get("action") for i in memory]
        finish = [int(i.get("finish")) for i in memory]
        reward = [i.get("reward") for i in memory]
        result = {
            "state": create_input(state),
            "state_next": create_input(state_next),
            "action": action,
            "finish": finish,
            "reward": reward,
            "hidden_state": hidden,
            "hidden_state_next": hidden_next,
        }
        return result
