import random
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from IPython import display
from memory import Memory
from utils import preprocess
from model.simple_stack import SimpleStack

# frame channel
channel = 1
# The size of the batch to learn the Q-function
batch_size = 16
# Resize the raw input frame to square frame of size 80 by 80
image_size = 84
# Skip 4-1 raw frames between steps
skip_frame = 4
# Skip 4-1 raw frames between skipped frames
internal_skip_frame = 4
# The size of replay buffer; set it to size of your memory (.5M for 50G available memory)
replay_buffer_size = 100000
# With Freq of 1/4-step update the Q-network
learning_frequency = 4
# Each state is formed as a concatenation 4 step frames [f(t-12),f(t-8),f(t-4),f(t)]
frame_len = 4
# Update the target network each 10000 steps
target_update = 10000
# Minimum level of stochasticity of policy (epsilon)-greedy
epsilon_min = 0.01
# The number of step it take to linearly anneal the epsilon to it min value
annealing_end = 1000000.
# The discount factor
gamma = 0.99
# Start to back propagated through the network, learning starts
replay_start_size = 50000
# Run uniform policy for first 30 times step of the beginning of the game
no_op_max = 8
# Number episode to run the algorithm
num_episode = 10000000
max_frame = 200000000
# RMSprop learning rate
lr = 0.00025
# RMSprop gamma1
gamma1 = 0.95
# RMSprop gamma2
gamma2 = 0.95
# RMSprop epsilon bias
rms_eps = 0.01
# Enables gpu if available, if not, set it to mx.cpu()
ctx = mx.gpu()

env_name = 'AssaultNoFrameskip-v4'
env = gym.make(env_name)
num_action = env.action_space.n
manualSeed = 1
mx.random.seed(manualSeed)

dqn = SimpleStack(env.action_space.n, frame_len, channel=channel)
dqn.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
trainer = gluon.Trainer(dqn.collect_params(), 'RMSProp',
                        {'learning_rate': lr, 'gamma1': gamma1, 'gamma2': gamma2, 'epsilon': rms_eps, 'centered': True})
dqn.collect_params().zero_grad()

target_dqn = SimpleStack(env.action_space.n, frame_len, channel=channel)
target_dqn.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)


def rew_clipper(rew_clip):
    if rew_clip > 0.:
        return 1.
    elif rew_clip < 0.:
        return -1.
    else:
        return 0


def render_image(frame, render):
    if render:
        plt.imshow(frame)
        plt.show()
        display.clear_output(wait=True)
        time.sleep(.1)


loss_f = mx.gluon.loss.L2Loss(batch_axis=0)
# Counts the number of steps so far
frame_counter = 0.
# Counts the number of annealing steps
annealing_count = 0.
# Counts the number episodes so far
epis_count = 0.
# Initialize the replay buffer
replay_memory = Memory(replay_buffer_size)
tot_clipped_reward = []
tot_reward = []
frame_count_record = []
moving_average_clipped = 0.
moving_average = 0.

# Whether to render Frames and show the game
_render = False
while epis_count < max_frame:
    cum_clipped_reward = 0
    cum_reward = 0
    next_frame = env.reset()
    state, current_frame = preprocess(next_frame, image_size, channel, frame_len, initial_state=True)
    t = 0.
    done = False
    initial_state = True
    while not done:
        previous_state = state
        # show the frame
        render_image(next_frame, _render)
        sample = random.random()
        if frame_counter > replay_start_size:
            annealing_count += 1
        if frame_counter == replay_start_size:
            print('annealing and learning are started ')
        eps = np.maximum(1. - annealing_count / annealing_end, epsilon_min)

        effective_eps = eps
        if t < no_op_max:
            effective_eps = 1.
        # epsilon greedy policy
        if sample < effective_eps:
            action = random.randint(0, num_action - 1)
        else:
            data = [nd.array(state.reshape([1, frame_len, image_size, image_size]), ctx), nd.array([100 - t], ctx)]
            action = int(nd.argmax(dqn(data), axis=1).as_in_context(mx.cpu()).asscalar())
        # Skip frame
        rew = 0
        for skip in range(skip_frame - 1):
            next_frame, reward, done, _ = env.step(action)
            render_image(next_frame, _render)
            cum_clipped_reward += rew_clipper(reward)
            rew += reward
            for internal_skip in range(internal_skip_frame - 1):
                _, reward, done, _ = env.step(action)
                cum_clipped_reward += rew_clipper(reward)
                rew += reward

        next_frame_new, reward, done, _ = env.step(action)
        render_image(next_frame, _render)
        cum_clipped_reward += rew_clipper(reward)
        rew += reward
        cum_reward += rew

        # Reward clipping

        reward = rew_clipper(rew)
        next_frame = np.maximum(next_frame_new, next_frame)
        state, current_frame = preprocess(next_frame, image_size, channel, frame_len, current_state=state)
        replay_memory.push(previous_state, action, state, reward, done)

        # Train
        if frame_counter > replay_start_size:
            if frame_counter % learning_frequency == 0:
                batch = replay_memory.sample(batch_size, ctx)
                batch_state = batch.state
                batch_state_next = batch.state_next
                batch_battery  = batch.battery
                batch_reward = batch.reward
                batch_action = batch.action.astype('uint8')
                batch_done = batch.finish.astype('uint8')
                with autograd.record():
                    argmax_Q = nd.argmax(dqn(batch_state_next, batch_battery),axis = 1).astype('uint8')
                    Q_sp = nd.pick(target_dqn(batch.state_next, batch.battery),argmax_Q,1)
                    Q_sp = Q_sp*(nd.ones(batch_size,ctx = ctx)-batch_done)
                    Q_s_array = dqn([batch_state, batch_battery])
                    Q_s = nd.pick(Q_s_array,batch_action,1)
                    loss = nd.mean(loss_f(Q_s ,  (batch_reward + gamma *Q_sp)))
                loss.backward()
                trainer.step(batch_size)
        t += 1
        frame_counter += 1
        # Save the model and update Target model
        if frame_counter >replay_start_size:
            if frame_counter % target_update == 0:
                check_point = frame_counter / (target_update * 100)
                filename = './data/target_%s_%d' % (env_name, int(check_point))
                dqn.save_params(filename)
                target_dqn.load_params(filename, ctx)
                filename = './data/clipped_rew_DDQN_%s' % env_name
                np.save(filename, tot_clipped_reward)
                filename = './data/tot_rew_DDQN_%s' % env_name
                np.save(filename, tot_reward)
                filename = './data/frame_count_DDQN_%s' % env_name
                np.save(filename, frame_count_record)
        if done:
            if epis_count % 50. == 0. :
                print('epis[%d],eps[%.4f],durat[%d],fnum=%d, cum_cl_rew = %d, cum_rew = %.4f,tot_cl = %.4f , tot = %.4f'\
                  %(epis_count,eps,t+1,frame_counter,cum_clipped_reward,cum_reward,moving_average_clipped,moving_average))
    epis_count += 1
    tot_clipped_reward = np.append(tot_clipped_reward, cum_clipped_reward)
    tot_reward = np.append(tot_reward, cum_reward)
    frame_count_record = np.append(frame_count_record,frame_counter)
    if epis_count > 100.:
        moving_average_clipped = np.mean(tot_clipped_reward[int(epis_count)-1-100:int(epis_count)-1])
        moving_average = np.mean(tot_reward[int(epis_count)-1-100:int(epis_count)-1])