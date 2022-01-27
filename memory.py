import random
from collections import namedtuple


class Memory(object):
    def __init__(self, memory_length=2048, frame_len=4):
        """
        dataset in mxnet case
        :param memory_length: int
        memory_length
        memory_size of ('state', 'action'', 'reward','finish'， 'battery', 'initial)
        """
        self.memory = []
        self.memory_length = memory_length
        self.frame_len = frame_len
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.memory_length:
            self.memory.append(None)
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'finish', 'battery', 'initial'))
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.memory_length

    def sample(self, batch_size,batch_state,batch_state_next,batch_reward,batch_action,batch_done, batch_battery):
        ctx = batch_size.context
        for i in range(batch_size):
            j = random.randint(self.frame_len-1,len(self.memory)-2)
            for jj in range(self.frame_len):
                batch_state[i,self.frame_len-1-jj] = self.memory[j-jj].state[0].as_in_context(ctx).astype('float32')/255.
                batch_state_next[i,self.frame_len-1-jj] = self.memory[j-jj+1].state.as_in_context(ctx)[0].astype('float32')/255.
                if self.memory[j-jj].initial_state:
                    for kk in range(self.frame_len-jj-1):
                        batch_state[i,self.frame_len-2-kk] =  self.memory[j-jj].state[0].as_in_context(ctx).astype('float32')/255.
                        batch_state_next[i,self.frame_len-2-kk] = self.memory[j-jj].state.as_in_context(ctx)[0].astype('float32')/255.
                    break
            if self.memory[j].done:
                batch_state_next[i,self.frame_len-1] = batch_state[i,self.frame_len-1]
            batch_reward[i] = self.memory[j].reward
            batch_action[i] = self.memory[j].action
            batch_done[i] = self.memory[j].done
            batch_battery[i] = self.memory[j].battery
