from config import *
import os
import numpy as np
import mxnet as mx
from model.simple_stack import SimpleStack
from utils import check_dir
from memory import Memory
from environments.SimpleEnv import SimpleEnv
from utils import create_input, translate_state
from evaluation import evaluate
from mxnet import gluon, nd, autograd

if os.path.exists(summary):
    os.remove(summary)
ctx = mx.cpu()
for i in ["model_save", "data_save"]:
    check_dir(i)
# build models
online_model = SimpleStack(agent_view, map_size)
offline_model = SimpleStack(agent_view, map_size)
online_model.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
offline_model.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
offline_model.collect_params().zero_grad()
# create env
env = SimpleEnv(display=False)
env.reset_env()
memory_pool = Memory(memory_length, ctx=ctx)
annealing = 0
total_reward = np.zeros(num_episode)
eval_result = []
loss_func = gluon.loss.L2Loss()
trainer = gluon.Trainer(offline_model.collect_params(), 'adam',
                        {'learning_rate': lr})
for epoch in range(num_episode):
    env.reset_env()
    finish = 0
    cum_clipped_dr = 0
    if epoch == 100:
        print("Model Structure: ")
        print(offline_model)
    if sum(env.step_count) > replay_start:
        print('annealing and learning are started')
    while not finish:
        if sum(env.step_count) > replay_start:
            annealing += 1
        eps = np.maximum(1 - sum(env.step_count) / annealing_end, epsilon_min)
        if np.random.random() < eps:
            by = "Random"
            action = np.random.randint(0, action_max)
        else:
            by = "Model"
            data = create_input([translate_state(env.map.state())], ctx)
            action = offline_model(data)
            action = int(nd.argmax(action, axis=1).asnumpy()[0])
        old, new, reward_get, finish = env.step(action)
        memory_pool.add(old, new, action, reward_get, finish)
        if finish and epoch > 50:
            cum_clipped_dr += env.detect_rate[-1]
            dr_50 = float(np.mean(env.detect_rate[-50:]))
            dr_all = float(np.mean(env.detect_rate))
            if epoch % 50 == 0:
                text = "DR: %f(50), %f(all), eps: %f" % (dr_50, dr_all, eps)
                print(text)
                with open(summary, "a") as f:
                    f.writelines(text + "\n")
            if epoch % 100 == 0 and annealing > replay_start:
                eval_result.extend(evaluate(offline_model, 5, ctx))
        # save model and replace online model each update_step
        if annealing > replay_start and annealing % update_step == 0:
            offline_model.save_parameters(temporary_model)
            online_model.load_parameters(temporary_model, ctx)
    #  train every 4 epoch
    if annealing > replay_start and epoch % 4 == 0:
        # Sample random mini batch of transitions
        if len(memory_pool.memory) > batch_size:
            bz = batch_size
        else:
            bz = len(memory_pool.memory)
        for_train = memory_pool.next_batch(bz)
        with autograd.record(train_mode=True):
            q_sp = nd.max(online_model(for_train["state_next"]), axis=1)
            q_sp = q_sp * (nd.ones(bz, ctx=ctx) - for_train["finish"])
            q_s_array = offline_model(for_train["state"])
            q_s = nd.pick(q_s_array, for_train["action"], 1)
            loss = nd.mean(
                loss_func(q_s, (for_train["reward"] + gamma * q_sp)))
        loss.backward()
        trainer.step(bz)
    total_reward[int(epoch) - 1] = cum_clipped_dr
