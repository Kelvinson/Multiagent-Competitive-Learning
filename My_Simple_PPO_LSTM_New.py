"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import copy
import os
import gym_compete

EP_MAX = 1
EP_LEN = 64
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 137, 8

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class Policy(object):
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        # should return act, info
        raise NotImplementedError()

class RunningMeanStd(object):
    def __init__(self, scope="running", reuse=False, epsilon=1e-2, shape=()):
        with tf.variable_scope(scope, reuse=reuse):
            self._sum = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="sum", trainable=False)
            self._sumsq = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="sumsq", trainable=False)
            self._count = tf.get_variable(
                dtype=tf.float32,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=False)
            self.shape = shape

            self.mean = tf.to_float(self._sum / self._count)
            var_est = tf.to_float(self._sumsq / self._count) - tf.square(self.mean)
            self.std = tf.sqrt(tf.maximum(var_est, 1e-2))

def switch(condition, if_exp, else_exp):
    x_shape = copy.copy(if_exp.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: if_exp,
                lambda: else_exp)
    x.set_shape(x_shape)
    return x

def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret

class DiagonalGaussian(object):
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def mode(self):
        return self.mean

class PPO():
    def __init__(self, scope, *, ob_space, ac_space, hiddens, reuse=False, normalize=False):
        self.recurrent = True
        self.normalized = normalize
        with tf.variable_scope(scope, reuse = reuse):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box)

            self.observation_ph = tf.placeholder(tf.float32, [None, None] + list(ob_space.shape), name="observation")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, ac_space.shape[0]],
                                                  name="taken_action")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
            self.zero_state = []
            self.state_in_ph = []
            self.state_out = []
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmv_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmv_h"))
            initial_state = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2], self.state_in_ph[-1])
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=initial_state, scope="lstmv")
            self.state_out.append(state_out)

            self.vpredz = tf.contrib.layers.fully_connected(last_out, 1, activation_fn=None)[:, :, 0]
            self.vpred = self.vpredz
            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmp_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmp_h"))
            initial_state = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2], self.state_in_ph[-1])
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=initial_state, scope="lstmp")
            self.state_out.append(state_out)

            mean = tf.contrib.layers.fully_connected(last_out, ac_space.shape[0], activation_fn=None)
            logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]], initializer=tf.zeros_initializer())

            self.pd = DiagonalGaussian(mean, logstd)
            # self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())
            self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())

            self.zero_state = np.array(self.zero_state)
            self.state_in_ph = tuple(self.state_in_ph)
            self.state = self.zero_state

            for p in self.get_trainable_variables():
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.square(p)))

    def make_feed_dict(self, observation, state_in, taken_action):
        return {
            self.observation_ph: observation,
            self.state_in_ph: list(np.transpose(state_in, (1, 0, 2))),
            self.taken_action_ph: taken_action
        }

    def act(self, observation, stochastic=True):
        outputs = [self.sampled_action, self.vpred, self.state_out]
        a, v, s = tf.get_default_session().run(outputs, {
            self.observation_ph: observation[None, None],
            self.state_in_ph: list(self.state[:, None, :]),
            self.stochastic_ph: stochastic})
        self.state = []
        for x in s:
            self.state.append(x.c[0])
            self.state.append(x.h[0])
        self.state = np.array(self.state)
        return a[0, 0], {'vpred': v[0, 0], 'state': self.state}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def reset(self):
        self.state = self.zero_state


env = gym.make("sumo-ants-v0").unwrapped
len1 = len(env.agents)
tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
sess = tf.Session(config=tf_config)
sess.__enter__()
ppo = []
for i in range(len1):
    scope = "agent" + str(i)
    ppo.append(PPO(scope=scope, reuse=False,
                                     ob_space=env.observation_space.spaces[i],
                                     ac_space=env.action_space.spaces[i],
                                     hiddens=[128, 128], normalize=True))


sess.run(tf.variables_initializer(tf.global_variables()))
all_ep_r = [[] for i in range(len1)]
for ep in range(EP_MAX):
    # in the environment of "ante-sumo" s of environment produced is a tuple of two ob arrays
    # remember s is a tuple
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [[] for i in range(len1)], [[] for i in range(len1)], [[] for i in range(len1)]
    ep_r = [0.0 for i in range(len1)]
    for t in range(EP_LEN):    # in one episode
        # env.render()
        # a is the action for one step of two agents
        a = [ppo[i].act(stochastic=True, observation=s[i])[0]
                        for i in range(len1)]
        info_dict = [ppo[i].act(stochastic=True, observation=s[i])[1]
                        for i in range(len1)]
        print("the action value for the two agents are {} and {} respectively".format(a[0], a[1]))
        # now in the environment of "ant-sumo", all the returned info is tuple
        # in order to feed the step function, 'a' should be tuple type so
        # just save a in a temple tuple variable a2tuple
        a2tuple = tuple(a)
        # the returned state info s_, rewared r, done info done are all tuple of two
        s_, r, done, info = env.step(a2tuple)

        for i in range(len1):


            buffer_s[i].append(s[i])
            buffer_a[i].append(a[i])
            # exploration_reward = r[i][0] * (1-t * 0.002)
            # if t == EP_LEN -1:
            #     competition_reward = r[i][1] * t * 0.002
            # else :
            #     competition_reward = 0
            # r[i] = exploration_reward + competition_reward
            # center_reward = info['reward_center']
            # ctrl_cost = info['reward_ctrl']
            # contact_cost  = info['reward_contact']
            # survive = info['reward_survive']
            exploration_reward = info[i]['reward_move'] * (1 - t*0.002)
            competition_reward = info[i]['reward_remaining'] * t * 0.002
            rewrd= exploration_reward + competition_reward
            # buffer_r[i].append((rewrd+8)/8)    # normalize reward, find to be useful
            ep_r[i] += rewrd

        print("At step {} in episode {} reward for two agents are {} and  {} respectively".format(t, ep, ep_r[0], ep_r[1]))
        s = s_

        # update ppo
        v_s_ = [0.0 for i in range(len1)]
        discounted_r = [[] for i in range(len1)]
        # bs, ba, br = [np.zeros(3,1) for i in range(len1)], [np.zeros(3,1) for i in range(len1)],[np.zeros(3,1) for i in range(len1)]
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            # print(s)
            for i in range(len1):
                v_s_[i] = info_dict[i]['vpred']
                discounted_r[i] = []
                # note: a[::-1] return the reversed array a
                # but a[:-1] return the array a but the last element of a
                for r in buffer_r[i][::-1]:
                    # r is the returned reward by the environment through env.step
                    v_s_[i] = r + GAMMA * v_s_[i]
                    discounted_r[i].append(v_s_[i])
                discounted_r[i].reverse()
                #Stack arrays in sequence vertically (row wise).
                # Take a sequence of arrays and stack them vertically
                # to make a single array. Rebuild arrays divided by vsplit.

                #TODO: before this step, first normalize the input state to (-5.0,5.0) as follows:

                #obz = self.observation_ph
                # if self.normalized:
                #   obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                bs, ba, br= np.vstack(buffer_s[i]), np.vstack(buffer_a[i]), np.array(discounted_r[i])[:, np.newaxis]
                # buffer_s, buffer_a, buffer_r = [[] for i in range(len1)], [[] for i in range(len1)], [[] for i in range(len1)]
                bs = (bs - np.mean(bs)) / (np.std(bs))
                # bs = np.clip(bs, -5.0, 5.0)


                buffer_s[i], buffer_a[i], buffer_r[i] = [], [], []
                # for i in range(len1):
                ppo[i].update(bs, ba, br)
    if ep == 0:
        for i in range(len1):
            all_ep_r[i].append(ep_r[i])
            print(all_ep_r[i])
    else:
        for i in range(len1):
            print(all_ep_r[i][-1])
            all_ep_r[i].append(all_ep_r[i][-1]*0.9 + ep_r[i]*0.1)
            print(all_ep_r[i])
    print(all_ep_r)
    print(
        'Ep: %i' % ep,
        "|Ep_r_0: %f" % ep_r[0],
        "|Ep-r_1: %f" % ep_r[1],
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

# for xe, ye in zip(np.arange(len(all_ep_r[0])), all_ep_r):
#     plt.scatter([xe] * len(ye), ye)
#
# plt.xticks([1,2])
# plt.axes().set_xticklabels(['agent0', 'agent1'])
# plt.savefig('train-multiagent1.png')
plt.subplot(1,2,1)
plt.plot(np.arange(len(all_ep_r[0])), all_ep_r[0])
plt.xlabel('Episode');
plt.ylabel('Moving averaged episode reward for agent0');
plt.subplot(1,2,2)
plt.plot(np.arange(len(all_ep_r[1])), all_ep_r[1])
plt.xlabel('Episode');
plt.ylabel('Moving averaged episode reward for agent0');
plt.show()
plt.close()