import os
import json
import func_timeout
import numpy as np
import mxnet as mx
from algorithm import DQN
from memory import Memory
from model import Stack
from datetime import datetime
import time

ctx = mx.cpu(0)

# build models
online_model = Stack()
offline_model = Stack()

# read from database
memory_pool = Memory(memory_length)

memory_pool.memory = data
print("overwrite memory with length ", len(memory_pool.memory))

# DQN
dqn = DQN(models=[online_model, offline_model], ctx=ctx, lr=lr, gamma=gamma, pool=memory_pool)
waiting = []


@func_timeout.func_set_timeout(100)
def step(poss, counter):
    old_state = env.state()
    action, by = dqn.get_action(old_state, poss)
    try:
        reward, finish, text, new_state = env.step(action, counter)
    except func_timeout.exceptions.FunctionTimedOut:
        reward = -100
        finish = -1
        text = "collision"
        new_state = old_state
        by = "Error"
    _data = {
        "old": old_state,
        "new": new_state,
        "action": action,
        "finish": finish,
        "reward": reward
    }
    print(text, by, poss, action)
    # store transition (st, at, rt, st+1) in memory
    waiting.append(_data)
    result = {
        "time": datetime.now().strftime("%m-%d-%H-%M-%S"),
        "action": action,
        "reward": reward,
        "state": new_state,
        "finish": finish,
        "by": by,
        "goal": env.aim
    }
    return result, finish, reward


all_step_counter = 0

while True:
    goal = read_one_goal(goal_list)
    _goal = [int(i) for i in goal["X;Y"].split(";")]
    rewards = 0
    all_steps = []
    print("goal: %d, %d, succeed: %d, fail: %d" % (_goal[0], _goal[1], goal["success"], goal["fail"]))
    # initialise sequence
    env = EnvClient(_goal, step_limit)
    env.reset_env()
    _finish = 0
    step_counter = 0
    while _finish == 0:
        all_step_counter += 1
        try:
            _result, _finish, _reward = step(np.maximum(1 - all_step_counter / annealing_end, epsilon_min),
                                             step_counter)
            rewards += _reward
            all_steps.append(state_to_json(_result))
            step_counter += 1
        except func_timeout.exceptions.FunctionTimedOut:
            _finish = -1
        #  train 5 step once
        if all_step_counter % 5 == 0:
            dqn.train()
        # save model and replace online model each 20 steps
        if all_step_counter % 20 == 0:
            dqn.offline.save_parameters(temporary_model)
            dqn.online.load_parameters(temporary_model, dqn.ctx)
        if all_step_counter % 500 == 0:
            try:
                # add to database
                write(waiting)
                waiting = []
                # overwrite from database
                memory_pool = Memory(memory_length)
                data = read(memory_length)
                memory_pool.memory = data
                print("overwrite memory with length ", len(memory_pool.memory))
            except:
                pass
    # write result
    with open("./data_save/{}.json".format(datetime.now().strftime("%m-%d-%H-%M-%S")), "w") as f:
        to_save = {
            "steps": all_steps,
            "goal": goal["X;Y"],
            "order": int(goal["success"]),
            "finish": _finish
        }
        json.dump(to_save, f)
    if _finish == 1 and all_step_counter > 20:
        goal["success"] += 1
        goal["wait"] -= 1
        # save model when success
        dqn.offline.save_parameters("./{}/{}.params".format(model_save, read_succeed(goal_list)))
    else:
        goal["fail"] += 1
    write_goal(goal, goal_list)
    try:
        env.reset_env()
    except func_timeout.exceptions.FunctionTimedOut:
        env.stop_client()
        time.sleep(1)
        os.system("python ./airsim_operation.py")
        time.sleep(40)
