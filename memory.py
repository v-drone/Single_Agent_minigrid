import numpy as np
from utils import translate_state


class Memory(object):
    def __init__(self, memory_length=2048):
        """
        dataset in mxnet case
        :param memory_length: int
        memory_length
        memory_size of ('state', 'action', 'next_state', 'reward','finish')
        """
        self.memory_length = memory_length
        self.memory = []
        self.index = 0

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
        import pdb
        pdb.set_trace()
        return result
