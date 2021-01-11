import time
# action max
action_max = 2
# learning rate
model_save = "./model_save/"
lr = 0.01
# start play
replay_start_size = 10000
# update step
update_step = 500
# gamma in q-loss calculation
gamma = 0.99
# memory pool size
memory_length = 100000
# file to save train log
summary = "./test_{}".format(str(time.time()))
# the number of step it take to linearly anneal the epsilon to it min value
annealing_end = 100000
# min level of stochastically of policy (epsilon)-greedy
epsilon_min = 0.2
# temporary files
temporary_model = "./{}/model.params".format(model_save)
temporary_pool = "./{}/pool".format(model_save)