import numpy as np
from utils import create_input
from mxnet import nd


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
        _ = {"state": old, "state_next": new, "action": action, "reward": reward, "finish": finish}
        if len(self.memory) < self.memory_length:
            self.memory.append(None)
        self.memory[self.position] = _
        self.position = (self.position + 1) % self.memory_length

    def next_batch(self, bz):
        index = np.random.choice(len(self.memory), bz)
        memory = [self.memory[i] for i in index]
        batch_state = [i["state"] for i in memory]
        batch_state = [nd.concat(*[nd.expand_dims(i[0], 0) for i in batch_state], dim=0),
                       nd.concat(*[nd.expand_dims(i[1], 0) for i in batch_state], dim=0)]
        batch_state_next = [i["state_next"] for i in memory]
        batch_state_next = [nd.concat(*[nd.expand_dims(i[0], 0) for i in batch_state_next], dim=0),
                            nd.concat(*[nd.expand_dims(i[1], 0) for i in batch_state_next], dim=0)]
        action = nd.array([i.get("action") for i in memory], batch_state_next[0].context)
        finish = nd.array([int(i.get("finish")) for i in memory], batch_state_next[0].context)
        reward = nd.array([int(i.get("reward")) for i in memory], batch_state_next[0].context)
        result = {
            "state": batch_state,
            "state_next": batch_state_next,
            "action": action,
            "finish": finish,
            "reward": reward
        }
        return result
