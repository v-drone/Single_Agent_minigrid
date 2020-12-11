import numpy as np
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from . import AbstractAlgorithm


class DQN(AbstractAlgorithm):
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
        super(DQN, self).__init__(models, action_max, ctx)
        self.temporary_model = temporary_model
        self.batch_size = 512
        self.training_counter = 0
        self.lr = lr
        self.gamma = gamma
        self.dataset = pool
        self.trainer = gluon.Trainer(self.offline.collect_params(), 'sgd', {'learning_rate': lr, 'wd': 0.0001})
        self.online.collect_params().zero_grad()
        self.loss_func = gluon.loss.L2Loss()

    def reload(self):
        self.offline.save_parameters(self.temporary_model)
        self.online.load_parameters(self.temporary_model, self.ctx)

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
            Q_sp = nd.max(self.online(batch_state_next), axis=1)
            Q_sp = Q_sp * (nd.ones(bz, ctx=self.ctx) - batch_finish)
            Q_s_array = self.offline(batch_state)
            Q_s = nd.pick(Q_s_array, batch_action, 1)
            loss = nd.mean(self.loss_func(Q_s, (batch_reward + self.gamma * Q_sp)))
        loss.backward()
        self.trainer.step(bz)
        return loss.asscalar()
