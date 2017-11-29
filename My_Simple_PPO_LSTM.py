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

class RunningMeanStd(object):
    def __init__(self, scope= "running", reuse=False, epsilon=1e-2, shape=()):
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


class DiagonalGaussian(object):
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def mode(self):
        return self.mean

class PPO(object):

    def __init__(self, scope):
        with tf.variable_scope(scope):
            self.sess = tf.Session()
            ob_shape = (137,) # it's a tuple of one element
            ac_space = (8,)  # it is a tuple of one element
            hiddens = [128, 128]
            self.tfs = tf.placeholder(tf.float32, [None, None] + list(ob_shape), 'observation')
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, ac_space[0]],name="taken_action")

            #first try to normalize the the observation
            self.ret_rms = RunningMeanStd(scope = "retfilter")
            self.ob_rms = RunningMeanStd(shape = ob_shape, scope="obsfilter")

            obz = tf.clip_by_value((self.tfs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)



            self.zero_state = []
            self.state_in_ph = []
            self.state_out = []
            # critic
            with tf.variable_scope('critic'):
                last_out = obz
                for hidden in hiddens[:-1]:
                    last_out = tf.contrib.layers.fully_connected(last_out, hidden)

                cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=False)
                size = cell.state_size
                self.zero_state.append(np.zeros(size.c, dtype=np.float32))
                self.zero_state.append(np.zeros(size.h, dtype=np.float32))
                self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmv_c"))
                self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmv_h"))
                initial_state = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2], self.state_in_ph[-1])
                last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=initial_state, scope="lstmv")
                self.state_out.append(state_out)

                self.vpredz = tf.contrib.layers.fully_connected(last_out, 1, activation_fn=None)[:, :, 0]
                self.v = self.vpredz * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

                # l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
                # self.v = tf.layers.dense(l1, 1)
                #

                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

            # actor
            pi,pi_norm, pi_params = self._build_anet('pi', trainable=True)
            oldpi, oldpi_normal, oldpi_params = self._build_anet('oldpi', trainable=False)
            with tf.variable_scope('sample_action'):
                self.sample_op = tf.squeeze(pi, axis=0)       # choosing action
            with tf.variable_scope('update_oldpi'):
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
            with tf.variable_scope('loss'):
                with tf.variable_scope('surrogate'):
                    # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                    # ratio = tf.distributions.Normal.prob(self.tfa) / tf.distributions.Normal.prob(self.tfa)
                    ratio = pi_norm.prob(self.tfa) / oldpi_normal.prob(self.tfa)
                    surr = ratio * self.tfadv
                if METHOD['name'] == 'kl_pen':
                    self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                    kl = tf.distributions.kl_divergence(oldpi, pi)
                    self.kl_mean = tf.reduce_mean(kl)
                    self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
                else:   # clipping method, find this is better
                    self.aloss = -tf.reduce_mean(tf.minimum(
                        surr,
                        tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

            with tf.variable_scope('atrain'):
                self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

            tf.summary.FileWriter("log/", self.sess.graph)

            self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s[None, None], self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s[None], self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # some time explode, this is my method
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s[None], self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s[None], self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            hiddens = [128, 128]
            obz = tf.clip_by_value((self.tfs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=False)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            # self.zero_state = np.concatenate(self.zero_state, np.zeros(size.c, dtype=np.float32))
            # self.zero_state = np.concatenate(self.zero_state, np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmp_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmp_h"))

            initial_state = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2], self.state_in_ph[-1])
            # print("type of last_out is",type(last_out))
            # print("last_out is ", last_out)
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=initial_state, scope="lstmp")
            self.state_out.append(state_out)

            self.mean = tf.contrib.layers.fully_connected(last_out, A_DIM, activation_fn=None)
            self.logstd = tf.get_variable(name="logstd", shape=A_DIM, initializer=tf.zeros_initializer())

            # print("in the policy network, the mean and logstd is {} {}".format(mean, logstd))
            pd = DiagonalGaussian(self.mean, self.logstd)
            # print("in the policy network, the output pd is {}".format(self.pd))

            # def switch(condition, if_exp, else_exp):
            # if stochastic is true then return sampled pd else return pd.mode()
            # first cast stochastic_ph to a bool value if it is true then sampled_action is pd.sample
            # else sampled_aciton is pd.mode
            # self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())
            sampled_action = pd.mean + pd.std * tf.random_normal(tf.shape(pd.mean))

            # to make it consistent with the original code
            norm_dist = tf.distributions.Normal(loc=self.mean, scale=self.logstd)
            # l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            # mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            # sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            # norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
            self.zero_state = np.array(self.zero_state)
            self.state_in_ph = tuple(self.state_in_ph)
            self.state = self.zero_state
            return sampled_action, norm_dist, params

    def choose_action(self, s):
        outputs = [self.sample_op, self.state_out]
        a, s = tf.get_default_session().run(outputs, {
            self.tfs: s[None, None],
            self.state_in_ph: list(self.state[:, None, :])
        })

        self.state  = []
        for x in s:
            self.state.append(x.c[0])
            self.state.append(x.h[0])
        self.state = np.array(self.state)
        # s = s[np.newaxis, :]
        # a = self.sess.run(self.sample_op, {self.tfs: s[None], self.state_in_ph: list(self.state[:, None,:])})
        # self.state = []
        # for x in
        # return np.clip(a, -2, 2)
        return a

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

env = gym.make("sumo-ants-v0").unwrapped
len1 = len(env.agents)
ppo = []
for i in range(len1):
    scope = "agent" + str(i)
    ppo.append(PPO(scope))

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
        a = [ppo[i].choose_action(s[i]) for i in range(len1)]
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
                v_s_[i] = ppo[i].get_v(s_[i])
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