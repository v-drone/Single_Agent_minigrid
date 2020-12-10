import numpy as np
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from utils import translate_state
import time


class DQN(object):
    def __init__(self, models, ctx, lr, gamma, pool, action_max, temporary_model):
        """
        mxnet DQN algorithm train case
        :param models: two model
        [online model, off line model]
        :param ctx: mxnet ctx
        :param lr: float
        learning rate
        :param gamma float
        """
        self.action_max = action_max
        self.temporary_model = temporary_model
        self.batch_size = 512
        self.training_counter = 0
        self.lr = lr
        self.gamma = gamma
        self.dataset = pool
        self.online = models[0]
        self.offline = models[1]
        self.trainer = gluon.Trainer(self.offline.collect_params(), 'sgd', {'learning_rate': lr, 'wd': 0.0001})
        self.online.collect_params().zero_grad()
        self.ctx = ctx
        self.loss_func = gluon.loss.L2Loss()

    def reload(self):
        self.offline.save_parameters(self.temporary_model)
        self.online.load_parameters(self.temporary_model, self.ctx)

    def get_action(self, state, poss):
        # epsilon greedy policy
        # with probability select a random action
        if np.random.random() < poss:
            by = "Random"
            action = np.random.randint(0, self.action_max)
        else:
            by = "Model"
            state = nd.array([translate_state(state)])
            state = state.as_in_context(self.ctx)
            action = self.offline(state)
            action = int(nd.argmax(action, axis=1).asnumpy()[0])
        return action, by

    def train(self):
        # Sample random mini batch of transitions
        if len(self.dataset.memory) > self.batch_size:
            bz = self.batch_size
        else:
            bz = len(self.dataset.memory)
        for_train = self.dataset.next_batch(bz)
        batch_state = nd.array(for_train["state"], self.ctx)
        batch_state_next = nd.array(for_train["state_next"], self.ctx)
        batch_action = nd.array(for_train["action"], self.ctx).astype('uint8')
        batch_reward = nd.array(for_train["reward"], self.ctx)
        batch_finish = nd.array(for_train["finish"], self.ctx)

        with autograd.record(train_mode=True):
            # non final
            mask = nd.ones(bz, ctx=self.ctx) - batch_finish
            # next state V(s_{t+1})
            batch_state_next = batch_state_next * mask.expand_dims(1)
            q_target = self.online(batch_state_next).detach()
            q_target = nd.max(q_target, axis=1)
            q_target = batch_reward + self.gamma * q_target
            # Q(s_t, a) - the model computes Q(s_t)
            q_eval = self.offline(batch_state)
            q_eval = nd.pick(q_eval, batch_action, 1)
            loss = nd.mean(self.loss_func(q_eval, q_target))
        loss.backward()
        self.trainer.step(bz)
        return nd.mean(loss).asscalar()
