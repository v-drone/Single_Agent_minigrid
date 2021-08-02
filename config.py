# training cases
order = "TEST"
# batch size
batch_size = 512
# agent view
agent_view = 7
map_size = 10
# action max
action_max = 3
# learning rate
model_save = "./model_save/"
lr = 0.001
num_episode = 1000000
# start play
replay_start = 10000
# update step
update_step = 1000
# gamma in q-loss calculation
gamma = 0.99
# memory pool size
memory_length = 100000
# file to save train log
summary = "./{}_Reward.csv".format(order)
eval_statistics = "./{}_CSV.csv".format(order)
# the number of step it take to linearly anneal the epsilon to it min value
annealing_end = 200000
# min level of stochastically of policy (epsilon)-greedy
epsilon_min = 0.2
# temporary files
temporary_model = "./{}/{}.params".format(model_save, order)
temporary_pool = "./{}/{}.pool".format(model_save, order)
