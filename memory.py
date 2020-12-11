import numpy as np
from utils import translate_state


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
        result = {
            "state": [],
            "state_next": [],
            "action": [],
            "finish": [],
            "reward": []
        }
        index = np.random.choice(len(self.memory), bz)
        _ = [self.memory[i] for i in index]

        for i in _:
            result["state"].extend(translate_state(i["state"]))
            result["state_next"].extend(translate_state(i["state_next"]))
            result["action"].extend(np.array([i["action"]]))
            result["finish"].extend(np.array([i["finish"]]))
            result["reward"].extend(np.array([i["reward"]]))
        result["state"] = np.array(result["state"]).reshape((bz, -1))
        result["state_next"] = np.array(result["state_next"]).reshape((bz, -1))
        result["action"] = np.array(result["action"])
        result["finish"] = np.array(result["finish"])
        result["reward"] = np.array(result["reward"])
        return result
