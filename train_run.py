#!/usr/bin/env python
from baselines.common import set_global_seeds, Dataset, explained_variance, fmt_row, zipsame
from baselines import bench
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

from baselines.ppo1.mlp_policy import MlpPolicy
from policy import MlpPolicyValue
import os.path as osp
import pickle
import gym
import gym_compete
import sys
import argparse
import logging
from baselines import logger
import tensorflow as tf
import numpy as np
import sys
import copy
import time


def train(args):
    if args.env == "sumo-ants":
        env = gym.make("sumo-ants-v0")
    else:
        print("right now I only support sumo-ants-v0")
        sys.exit()
    seed = args.seed
    num_timesteps = 10
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)

    # policy = []
    # for i in range(2):
    #     scope = "policy" + str(i)
    #     policy.append(MlpPolicyValue(scope=scope, reuse=False,
    #                                  ob_space=env.observation_space.spaces[i],
    #                                  ac_space=env.action_space.spaces[i],
    #                                  hiddens=[64, 64], normalize=True))

    def policy_fn(pi_name, i, ob_space, ac_space):
        scope = pi_name + str(i)
        return MlpPolicyValue(scope=scope, reuse=False,
                                     ob_space=env.observation_space.spaces[i],
                                     ac_space=env.action_space.spaces[i],
                                     hiddens=[64, 64], normalize=True)

    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    compete_learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()


def compete_learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = []
    ac_space = []
    pi = []
    oldpi = []
    atarg = []
    ret = []
    lrmult = []
    clip_param = []
    ob = []
    ac = []
    kloldnew = []
    ent = []
    meankl = []
    meanent = []
    pol_entpen = []
    ratio = []
    surr1 = []
    surr2 = []
    pol_surr = []
    vf_loss = []
    total_loss = []
    losses = []
    loss_names = []
    var_list = []
    lossandgrad = []
    adam = []
    for i in range(2):

        ob_space.append(env.observation_space.spaces[i])
        ac_space.append(env.action_space.spaces[i])
        pi.append(policy_func("pi", i, ob_space, ac_space)) # Construct network for new policy
        oldpi.append(policy_func("oldpi",i, ob_space, ac_space)) # Network for old policy
        atarg.append(tf.placeholder(dtype=tf.float32, shape=[None])) # Target advantage function (if applicable)
        ret.append(tf.placeholder(dtype=tf.float32, shape=[None])) # Empirical return

        lrmult.append(tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])) # learning rate multiplier, updated with schedule
        clip_param.append(clip_param * lrmult[i]) # Annealed cliping parameter epislon

        # def get_placeholder(name, dtype, shape):
        #     if name in _PLACEHOLDER_CACHE:
        #         out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        #         assert dtype1 == dtype and shape1 == shape
        #         return out
        #     else:
        #         out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        #         _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
        #         return out
        #
        # def get_placeholder_cached(name):
        #     return _PLACEHOLDER_CACHE[name][0]


        ob = U.get_placeholder_cached(name="ob")
        ac = pi[i].pdtype.sample_placeholder([None])

        kloldnew.append(oldpi[i].pd.kl(pi[i].pd))
        ent.append(pi[i].pd.entropy())
        meankl.append(U.mean(kloldnew[i]))
        meanent.append(U.mean(ent[i]))
        pol_entpen.append((-entcoeff) * meanent[i])

        ratio.append(tf.exp(pi[i].pd.logp(ac[i]) - oldpi[i].pd.logp(ac[i]))) # pnew / pold
        surr1.append(ratio[i] * atarg[i]) # surrogate from conservative policy iteration
        surr2.append(U.clip(ratio[i], 1.0 - clip_param[i], 1.0 + clip_param[i]) * atarg[i]) #
        pol_surr.append(- U.mean(tf.minimum(surr1[i], surr2[i]))) # PPO's pessimistic surrogate (L^CLIP)
        vf_loss.append(U.mean(tf.square(pi[i].vpred - ret[i])))
        total_loss.append(pol_surr[i] + pol_entpen[i] + vf_loss[i])
        losses.append([pol_surr[i], pol_entpen[i], vf_loss[i], meankl[i], meanent[i]])
        loss_names.append(["pol_surr" + str(i), "pol_entpen" + str(i), "vf_loss" + str(i), "kl" + str(i), "ent" + str(i)])

        var_list.append(pi[i].get_trainable_variables())
        lossandgrad.append(U.function([ob[i], ac[i], atarg[i], ret[i], lrmult[i]], losses[i] + [U.flatgrad(total_loss[i], var_list[i])]))
        adam.append(MpiAdam(var_list[i], epsilon=adam_epsilon))

        assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldpi[i].get_variables(), pi[i].get_variables())])
        compute_losses = U.function([ob[i], ac[i], atarg[i], ret[i], lrmult[i]], losses[i])

        U.initialize()
        adam[i].sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        vpredbefore = []
        tdlamret = []
        for i in range(2):
            ob[i], ac[i], atarg[i], tdlamret_temp= seg["ob"][i], seg["ac"][i], seg["adv"][i], seg["tdlamret"][i]
            tdlamret.append(tdlamret_temp)
            vpredbefore.append(seg["vpred"][i]) # predicted value function before udpate
            atarg[i] = (atarg[i] - atarg[i].mean()) / atarg[i].std() # standardized advantage function estimate
            d = Dataset(dict(ob=ob[i], ac=ac[i], atarg=atarg[i], vtarg=tdlamret[i]), shuffle=not pi[i].recurrent)
            optim_batchsize = optim_batchsize or ob[i].shape[0]

            if hasattr(pi[i], "ob_rms"): pi[i].ob_rms.update(ob[i]) # update running mean/std for policy

            assign_old_eq_new() # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, loss_names[i]))
        # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses[i] = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam[i].update(g, optim_stepsize * cur_lrmult)
                    losses[i].append(newlosses)
                logger.log(fmt_row(13, np.mean(losses[i], axis=0)))

            logger.log("Evaluating losses...")
            losses[i] = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses[i] = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses[i].append(newlosses)
            meanlosses,_,_ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names[i]):
                logger.record_tabular("loss_"+name, lossval)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore[i], tdlamret[i]))
            lrlocal = (seg["ep_lens"][i], seg["ep_rets"][i]) # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            if MPI.COMM_WORLD.Get_rank()==0:
                logger.dump_tabular()



# note after I revised this function, the trajectory segment generator will receive
# the double-view observation and output the joint actions of the competitive two agents
def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    # not used, just so we have the datatype, here ac is the random actions taken
    ac = env.action_space.sample() # ac is a tuple of length of two stands for the two agents' actions
    new = True # marks if we're on first timestep of an episode
    ob = env.reset() # env.reset() returns an initial observation, it is also a tuple of two

    cur_ep_ret = []
    cur_ep_len = []
    ep_lens = []
    ep_rets = []
    obs = []
    rews = []
    vpreds = []
    news = []
    acs = []
    prevacs = []
    for i in range(2):

        cur_ep_ret.append(0) # return in current episode
        cur_ep_len.append(0) # len of current episode
        ep_rets.append([]) # returns of completed episodes in this segment
        ep_lens.append([]) # lengths of completed episodes in this segement

    # Initialize history arrays
        obs.append(np.array([ob[i] for _ in range(horizon)])) # observations
        rews.append(np.zeros(horizon, 'float32')) # rewards
        vpreds.append(np.zeros(horizon, 'float32')) # previous values
        news.append(np.zeros(horizon, 'int32')) # the mark arrays
        acs.append(np.array([ac[i] for _ in range(horizon)])) # the action arrays
        prevacs.append(acs[i].copy()) # previous action arrays

    prevac = []
    ac = []
    vpred = []
    while True:
        for i in range(2):
            prevac.append(ac[i]) #copy action arrays to previous action arrays
            ac_temp, vpred_temp = pi[i].act(stochastic, ob[i]) # in the first time, input initial observations and execute random actions
            ac.append(ac_temp)
            vpred.append(vpred_temp)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets[i] = []
            ep_lens[i] = []

        for j in range(2):
            i = t % horizon
            obs[j][i] = ob[j]
            vpreds[j][i] = vpred[j]
            news[j][i] = new[j]
            acs[j][i] = ac[j]
            prevacs[j][i] = prevac[j]

        # observation, reward, done, info = env.step(action)
        # the done flag means whether this episode ends

        ob_temp, rew_temp, new_temp, _ = env.step(ac)
        for j in range(2):
            i = t % horizon
            ob[j] = ob_temp[j]
            rews[j][i] = rew_temp[j]
            new[j] = rew_temp[j]

            cur_ep_ret[j] += rew_temp[j]
            cur_ep_len[j] += 1

            if new[0] or new[1]:
                ep_rets[j].append(cur_ep_ret[j])
                ep_lens[j].append(cur_ep_len[j])
                cur_ep_ret[j] = 0
                cur_ep_len[j] = 0
                ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = []
    vpred = []
    gaelam = []
    rew = []
    for j in range(2):
        new.append(np.append(seg["new"][j], 0)) # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred.append(np.append(seg["vpred"][j], seg["nextvpred"][j]))
        T = len(seg["rew"][j])
        gaelam.append(np.empty(T,'float32'))
        seg["adv"][j] = gaelam[j]
        rew.append(seg["rew"][j])
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-new[j][t+1]
            delta = rew[j][t] + gamma * vpred[j][t+1] * nonterminal - vpred[j][t]

            lastgaelam[j] = delta + gamma * lam * nonterminal * lastgaelam[j]
            gaelam[j][t] = lastgaelam[j]
        seg["tdlamret"][j] = seg["adv"][j] + seg["vpred"][j]


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params


def setFromFlat(var_list, flat_params):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape)  )
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="Environments for Multi-agent competition")
    parser.add_argument("--env", default="sumo-ants", type=str, help="competitive environment: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)


    args = parser.parse_args()
    train(args)