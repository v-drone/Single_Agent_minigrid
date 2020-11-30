import time
# action max
action_max = 3
# learning rate
model_save = "./model_save/"
lr = 0.005
# gamma in q-loss calculation
gamma = 0.9
# memory pool size
memory_length = 100000
# each goal repeat number
replace_target_iter = 100
# maximum step avoid
step_limit = 50
# file to save train log
result_saver = "./test_{}".format(str(time.time()))
# the number of step it take to linearly anneal the epsilon to it min value
annealing_end = 100
# min level of stochastically of policy (epsilon)-greedy
epsilon_min = 0.15
# temporary files
temporary_model = "./{}/model.params".format(model_save)
temporary_pool = "./{}/pool".format(model_save)
