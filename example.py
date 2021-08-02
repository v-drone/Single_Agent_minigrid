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
online = SimpleStack()
offline = SimpleStack()
online.collect_params().initialize(mx.init.MSRAPrelu(), ctx=ctx)
offline.collect_params().initialize(mx.init.MSRAPrelu(), ctx=ctx)
offline.collect_params().zero_grad()
# create env
env = SimpleEnv(display=False, agent_view=agent_view, map_size=map_size)
env.reset_env()
memory_pool = Memory(memory_length, ctx=ctx)
annealing = 0
total_reward = np.zeros(num_episode)
eval_result = []
loss_func = gluon.loss.L2Loss()
trainer = gluon.Trainer(offline.collect_params(), 'adam',
                        {'learning_rate': lr})
print_text = True
last_dr_50 = 0
for epoch in range(num_episode):
    env.reset_env()
    finish = 0
    cum_clipped_dr = 0
    if epoch == 51:
        print("Model Structure: ")
        print(offline)
    if sum(env.step_count) > replay_start and print_text:
        print('annealing and learning are started')
        print_text = False
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
            action = offline(data)
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
                eval_result.extend(evaluate(ctx, offline, env, 1))
            # save model and replace online model each update_step
            if annealing % update_step == 0 and annealing > replay_start:
                if dr_50 >= last_dr_50:
                    last_dr_50 = dr_50
                    offline.save_parameters(temporary_model)
                    online.load_parameters(temporary_model, ctx)
    #  train every 2 epoch
    if annealing > replay_start and epoch % 2 == 0:
        # Sample random mini batch of transitions
        if len(memory_pool.memory) > batch_size:
            bz = batch_size
        else:
            bz = len(memory_pool.memory)
        for_train = memory_pool.next_batch(bz)
        with autograd.record(train_mode=True):
            q_sp = nd.max(online(for_train["state_next"]), axis=1)
            q_sp = q_sp * (nd.ones(bz, ctx=ctx) - for_train["finish"])
            q_s_array = offline(for_train["state"])
            q_s = nd.pick(q_s_array, for_train["action"], 1)
            loss = nd.mean(
                loss_func(q_s, (for_train["reward"] + gamma * q_sp)))
        loss.backward()
        trainer.step(bz)
    total_reward[int(epoch) - 1] = cum_clipped_dr
