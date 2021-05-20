from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from . import AbstractAlgorithm


class DQN(AbstractAlgorithm):
    def __init__(self, models, ctx, lr, gamma, pool, action_max,
                 temporary_model, bz=32):
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
        self.batch_size = bz
        self.training_counter = 0
        self.lr = lr
        self.gamma = gamma
        self.dataset = pool
        self.trainer = gluon.Trainer(self.offline.collect_params(), 'adam',
                                     {'learning_rate': lr})
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
        with autograd.record(train_mode=True):
            q_sp = nd.max(self.online(for_train["state_next"]), axis=1)
            q_sp = q_sp * (nd.ones(bz, ctx=self.ctx) - for_train["finish"])
            q_s_array = self.offline(for_train["state"])
            q_s = nd.pick(q_s_array, for_train["action"], 1)
            loss = nd.mean(
                self.loss_func(q_s, (for_train["reward"] + self.gamma * q_sp)))
        loss.backward()
        self.trainer.step(bz)
        return loss.asscalar()
