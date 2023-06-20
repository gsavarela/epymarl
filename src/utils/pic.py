""" Credits
    -------
    
    auxiliary functions from PIC

    source:
        PIC/replay_memory.py
        maddpg/utils.py
        maddpg/eval.py
        maddpg/ckpt_plots/plot_curve.py
* https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
"""
import contextlib
from collections import namedtuple
import csv
import os
import random
import sys
import time

import gym
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Agg')
import numpy as np
import torch as th


Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def shuffle(self):
        random.shuffle(self.memory)

    def __len__(self):
        return len(self.memory)

def dict2csv(output_dict, f_name):
    with open(f_name, mode='w') as f:
        writer = csv.writer(f, delimiter=",")
        for k, v in output_dict.items():
            v = [k] + v
            writer.writerow(v)

def n_actions(action_spaces):
    """
    :param action_space: list
    :return: n_action: list
    """
    n_actions = []
    from gym import spaces
    from multiagent.environment import MultiDiscrete
    for action_space in action_spaces:
        if isinstance(action_space, spaces.discrete.Discrete):
            n_actions.append(action_space.n)
        elif isinstance(action_space, MultiDiscrete):
            total_n_action = 0
            one_agent_n_action = 0
            for h, l in zip(action_space.high, action_space.low):
                total_n_action += int(h - l + 1)
                one_agent_n_action += int(h - l + 1)
            n_actions.append(one_agent_n_action)
        else:
            raise NotImplementedError
    return n_actions

def copy_actor_policy(s_agent, t_agent):
    if hasattr(s_agent, 'actors'):
        for i in range(s_agent.n_group):
            state_dict = s_agent.actors[i].state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            t_agent.actors[i].load_state_dict(state_dict)
        t_agent.actors_params, t_agent.critic_params = None, None
    else:
        state_dict = s_agent.actor.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        t_agent.actor.load_state_dict(state_dict)
        t_agent.actor_params, t_agent.critic_params = None, None

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_model_q(test_q, done_training, args):
    plot = {'good_rewards': [], 'adversary_rewards': [], 'rewards': [], 'steps': [], 'q_loss': [], 'gcn_q_loss': [],
            'p_loss': [], 'final': [], 'abs': []}
    best_eval_reward = -100000000
    while True:
        if not test_q.empty():
            print('=================== start eval ===================')
            # MPE
            # eval_env = make_env(args.scenario, args)
            eval_env = gym.make(args.scenario)
            eval_env.seed(args.seed + 10)
            eval_rewards = []
            good_eval_rewards = []
            agent, tr_log = test_q.get()
            with temp_seed(args.seed):
                for n_eval in range(args.num_eval_runs):
                    obs_n = eval_env.reset()
                    episode_reward = 0
                    episode_step = 0
                    # n_agents = eval_env.n
                    n_agents = len(eval_env.action_space)
                    agents_rew = [[] for _ in range(n_agents)]
                    while True:
                        action_n = agent.select_action(th.Tensor(obs_n), action_noise=True,
                                                       param_noise=False).squeeze().cpu().numpy()
                        next_obs_n, reward_n, done_n, _ = eval_env.step(action_n)
                        episode_step += 1
                        terminal = (episode_step >= args.num_steps)
                        episode_reward += np.sum(reward_n)
                        for i, r in enumerate(reward_n):
                            agents_rew[i].append(r)
                        obs_n = next_obs_n
                        if done_n[0] or terminal:
                            eval_rewards.append(episode_reward)
                            agents_rew = [np.sum(rew) for rew in agents_rew]
                            good_reward = np.sum(agents_rew)
                            good_eval_rewards.append(good_reward)
                            if n_eval % 100 == 0:
                                print('test reward', episode_reward)
                            break
                if np.mean(eval_rewards) > best_eval_reward:
                    best_eval_reward = np.mean(eval_rewards)
                    th.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents_best.ckpt'))

                plot['rewards'].append(np.mean(eval_rewards))
                plot['steps'].append(tr_log['total_numsteps'])
                plot['q_loss'].append(tr_log['value_loss'])
                plot['p_loss'].append(tr_log['policy_loss'])
                print("========================================================")
                print(
                    "Episode: {}, total numsteps: {}, {} eval runs, total time: {} s".
                        format(tr_log['i_episode'], tr_log['total_numsteps'], args.num_eval_runs,
                               time.time() - tr_log['start_time']))
                print("GOOD reward: avg {} std {}, average reward: {}, best reward {}".format(np.mean(eval_rewards),
                                                                                              np.std(eval_rewards),
                                                                                              np.mean(plot['rewards'][
                                                                                                      -10:]),
                                                                                              best_eval_reward))
                plot['final'].append(np.mean(plot['rewards'][-10:]))
                plot['abs'].append(best_eval_reward)
                dict2csv(plot, os.path.join(tr_log['exp_save_dir'], 'train_curve.csv'))
                eval_env.close()
        if done_training.value and test_q.empty():
            th.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents.ckpt'))
            break

def avg_list(l, avg_group_size=2):
    ret_l = []
    n = len(l)
    h_size = avg_group_size / 2
    for i in range(n):
        left = int(max(0, i - h_size))
        right = int(min(n, i + h_size))

        ret_l.append(np.mean(l[left:right]))
    return ret_l


def plot_result(t1, r1, fig_name, x_label, y_label):
    plt.close()
    base = None
    base, = plt.plot(t1, avg_list(r1))


    plt.grid()
    #plt.legend([base, teach1, teach2, teach3], ['CA error < 5%', 'CA error < 10%', 'CA error < 20%', 'MADDPG + global count'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_name)

    try:
        plt.savefig(fig_name + '.pdf')
        plt.savefig(fig_name + '.png')
    except:
        print('ERROR:', sys.exc_info()[0])
        print('Terminate Program')
        sys.exit()
    print('INFO: Wrote plot to ' + fig_name + '.pdf')


def plot_result2(t1, r1, r2, fig_name, x_label, y_label):
    plt.close()
    base = None
    l1, = plt.plot(t1, r1)
    l2, = plt.plot(t1, r2)

    plt.grid()
    plt.legend([l1, l2], ['train', 'val'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_name)

    try:
        plt.savefig(fig_name + '.pdf')
        plt.savefig(fig_name + '.png')
    except:
        print('ERROR:', sys.exc_info()[0])
        print('Terminate Program')
        sys.exit()
    print('INFO: Wrote plot to ' + fig_name + '.pdf')

