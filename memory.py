import random
import numpy as np
from mxnet import nd
from collections import namedtuple


class Memory(object):
    def __init__(self, memory_length=2048):
        """
        dataset in mxnet case
        :param memory_length: int
        memory_length
        memory_size of ('state', 'state_next', 'action'', 'reward','finish'， 'battery', 'initial)
        """
        self.memory = []
        self.memory_length = memory_length
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.memory_length:
            self.memory.append(None)
        Transition = namedtuple('Transition', ('state', 'state_next', 'action', 'reward', 'finish', 'battery'))
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.memory_length

    def sample(self, n):
        return random.sample(self.memory, n)

    def to_file(self, fn):
        _ = [list(i) for i in self.memory]
        _state = np.concatenate([i[0].asnumpy() for i in _])
        _state_next = np.concatenate([i[1].asnumpy() for i in _])
        _action = np.concatenate([i[2].asnumpy() for i in _])
        _reward = np.concatenate([i[3].asnumpy() for i in _])
        _finish = np.concatenate([i[4].asnumpy() for i in _])
        _battery = np.concatenate([i[5].asnumpy() for i in _])
        np.save("%s_state" % fn, _state)
        np.save("%s_state_next" % fn, _state_next)
        np.save("%s_action" % fn, _action)
        np.save("%s_reward" % fn, _reward)
        np.save("%s_finish" % fn, _finish)
        np.save("%s_battery" % fn, _battery)

    def load_file(self, fn, bz):
        Transition = namedtuple('Transition', ('state', 'state_next', 'action', 'reward', 'finish', 'battery'))
        _state = np.load("%s_state.npy" % fn)
        _state_next = np.load("%s_state_next.npy" % fn)
        _action = np.load("%s_action.npy" % fn)
        _reward = np.load("%s_reward.npy" % fn)
        _finish = np.load("%s_finish.npy" % fn)
        _battery = np.load("%s_battery.npy" % fn)
        _state = [nd.array(i).astype('uint8') for i in np.split(_state, _state.shape[0] / bz)]
        _state_next = [nd.array(i).astype('uint8') for i in np.split(_state_next, _state_next.shape[0] / bz)]
        _action = [nd.array(i).astype('uint8') for i in np.split(_action, _action.shape[0] / bz)]
        _reward = [nd.array(i) for i in np.split(_reward, _reward.shape[0] / bz)]
        _finish = [nd.array(i) for i in np.split(_finish, _finish.shape[0] / bz)]
        _battery = [nd.array(i) for i in np.split(_battery, _battery.shape[0] / bz)]
        self.memory = list(zip(*[_state, _state_next, _action, _reward, _finish, _battery]))
        self.memory = [Transition(*i) for i in self.memory]
        self.position = len(self.memory)
