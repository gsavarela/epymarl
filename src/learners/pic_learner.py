"""Credits:
    This module implements two variants:
        1. MADDPG based (source: epymarl)
        2. Original implementation (source: PIC)
PIC: Permutation Invariant Critic for Multi-Agent Deep Reinforcement Learning.
Liu, I., Yeh, R.A.  Schwing, A.G. Proceedings of the Conference on Robot Learning, 

* https://proceedings.mlr.press/v100/liu20a.html
* https://github.com/IouJenLiu/PIC
"""
# import cop
# from components.episode_buffer import EpisodeBatch
# import torch as th
# from torch.optim import RMSprop, Adam
# from controllers.maddpg_controller import gumbel_softmax
# from modules.critics import REGISTRY as critic_registry
# from components.standarize_stream import RunningMeanStd


import copy
import sys
import time
# TODO: Remove multiprocessing support
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value

import torch as th
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from IPython.core.debugger import set_trace

from modules.agents import REGISTRY as AGENT_REGISTRY
from modules.critics import REGISTRY as CRITIC_REGISTRY
from components.episode_buffer import EpisodeBatch
from utils.pic import ReplayMemory, Transition, copy_actor_policy, eval_model_q

Actor = AGENT_REGISTRY["pic"]
Critic = CRITIC_REGISTRY["ac_pic"]
M = ReplayMemory(capacity=50_000)
global_total_steps = 0
global_i_episode = 0
global_rewards = []
global_test_q = Queue()
done_training = Value('i', False)
global_best_eval_reward, global_best_good_eval_reward, global_best_adversary_eval_reward = -1000000000, -1000000000, -1000000000
start_time = time.time()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def adjust_lr(optimizer, init_lr, episode_i, num_episode, start_episode):
    if episode_i < start_episode:
        return init_lr
    lr = init_lr * (1 - (episode_i - start_episode) / (num_episode - start_episode))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class DDPG(object):
    """DDPG Implements many kinds of multi-agent deep deterministic policy gradient"""
    def __init__(self, gamma, tau, hidden_size, obs_dim, n_action, n_agent, obs_dims, agent_id, actor_lr, critic_lr,
                 fixed_lr, critic_type, actor_type, train_noise, num_episodes, num_steps,
                 critic_dec_cen, target_update_mode='soft', device='cpu'):
        self.device = device
        self.obs_dim = obs_dim
        self.n_agent = n_agent
        self.n_action = n_action

        # Instanciate actor block
        self.actor = Actor(hidden_size, obs_dim, n_action).to(self.device)
        self.actor_target = Actor(hidden_size, obs_dim, n_action).to(self.device)
        self.actor_perturbed = Actor(hidden_size, obs_dim, n_action)
        self.actor_optim = Adam(self.actor.parameters(),
                                lr=actor_lr, weight_decay=0)

        # Instanciate critic block
        self.critic = Critic(hidden_size, np.sum(obs_dims),
                             n_action * n_agent, n_agent, critic_type, agent_id).to(self.device)
        self.critic_target = Critic(hidden_size, np.sum(
            obs_dims), n_action * n_agent, n_agent, critic_type, agent_id).to(self.device)
        critic_n_params = sum(p.numel() for p in self.critic.parameters())
        print('# of critic params', critic_n_params)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.fixed_lr = fixed_lr
        self.init_act_lr = actor_lr
        self.init_critic_lr = critic_lr
        self.num_episodes = num_episodes
        self.start_episode = 0
        self.num_steps = num_steps
        self.actor_scheduler = LambdaLR(self.actor_optim, lr_lambda=self.lambda1)
        self.critic_scheduler = LambdaLR(self.critic_optim, lr_lambda=self.lambda1)
        self.gamma = gamma
        self.tau = tau
        self.train_noise = train_noise
        self.obs_dims_cumsum = np.cumsum(obs_dims)
        self.critic_dec_cen = critic_dec_cen
        self.agent_id = agent_id
        self.debug = False
        self.target_update_mode = target_update_mode
        # self.actor_params = self.actor.parameters()
        # self.critic_params = self.critic.parameters()
        # Make sure target is with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)


    # ERASEME: Moved to PICLearner
    # def adjust_lr(self, i_episode):
    #     adjust_lr(self.actor_optim, self.init_act_lr, i_episode, self.num_episodes, self.start_episode)
    #     adjust_lr(self.critic_optim, self.init_critic_lr, i_episode, self.num_episodes, self.start_episode)

    # ERASEME: Moved to PICLearner
    # def lambda1(self, step):
    #     start_decrease_step = ((self.num_episodes / 2)
    #                            * self.num_steps) / 100
    #     max_step = (self.num_episodes * self.num_steps) / 100
    #     return 1 - ((step - start_decrease_step) / (
    #             max_step - start_decrease_step)) if step > start_decrease_step else 1

    # ERASEME: Migrated to PICMAC
    # def select_action(self, state, action_noise=None, param_noise=False, grad=False):
    #     self.actor.eval()
    #     if param_noise:
    #         mu = self.actor_perturbed((Variable(state)))
    #     else:
    #         mu = self.actor((Variable(state)))

    #     self.actor.train()
    #     if not grad:
    #         mu = mu.data

    #     if action_noise:
    #         noise = np.log(-np.log(np.random.uniform(0, 1, mu.size())))
    #         try:
    #             mu -= th.Tensor(noise).to(self.device)
    #         except (AttributeError, AssertionError):
    #             mu -= th.Tensor(noise)

    #     action = F.softmax(mu, dim=1)
    #     if not grad:
    #         return action
    #     else:
    #         return action, mu

    # ERAME: Migratef to ACPIC
    # def update_critic_parameters(self, batch, agent_id, shuffle=None,
    #                              eval=False):
    #     state_batch = Variable(th.cat(batch.state)).to(self.device)
    #     action_batch = Variable(th.cat(batch.action)).to(self.device)
    #     reward_batch = Variable(th.cat(batch.reward)).to(self.device)
    #     mask_batch = Variable(th.cat(batch.mask)).to(self.device)
    #     next_state_batch = th.cat(batch.next_state).to(self.device)
    #     if shuffle == 'shuffle':
    #         rand_idx = np.random.permutation(self.n_agent)
    #         new_state_batch = state_batch.view(-1, self.n_agent, self.obs_dim)
    #         state_batch = new_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)
    #         new_next_state_batch = next_state_batch.view(-1, self.n_agent, self.obs_dim)
    #         next_state_batch = new_next_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)
    #         new_action_batch = action_batch.view(-1, self.n_agent, self.n_action)
    #         action_batch = new_action_batch[:, rand_idx, :].view(-1, self.n_action * self.n_agent)


    #     next_action_batch = self.select_action(
    #         next_state_batch.view(-1, self.obs_dim), action_noise=self.train_noise)
    #     next_action_batch = next_action_batch.view(-1, self.n_action * self.n_agent)
    #     next_state_action_values = self.critic_target(
    #             next_state_batch, next_action_batch)

    #     reward_batch = reward_batch[:, agent_id].unsqueeze(1)
    #     mask_batch = mask_batch[:, agent_id].unsqueeze(1)
    #     expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)
    #     self.critic_optim.zero_grad()
    #     state_action_batch = self.critic(state_batch, action_batch)
    #     perturb_out = 0
    #     value_loss = ((state_action_batch - expected_state_action_batch) ** 2).mean()
    #     if eval:
    #         return value_loss.item(), perturb_out
    #     value_loss.backward()
    #     unclipped_norm = clip_grad_norm_(self.critic_params, 0.5)
    #     self.critic_optim.step()
    #     if self.target_update_mode == 'soft':
    #         soft_update(self.critic_target, self.critic, self.tau)
    #     elif self.target_update_mode == 'hard':
    #         hard_update(self.critic_target, self.critic)
    #     return value_loss.item(), perturb_out, unclipped_norm

    # def update_actor_parameters(self, batch, agent_id, shuffle=None):
    #     state_batch = Variable(th.cat(batch.state)).to(self.device)
    #     if shuffle == 'shuffle':
    #         rand_idx = np.random.permutation(self.n_agent)
    #         new_state_batch = state_batch.view(-1, self.n_agent, self.obs_dim)
    #         state_batch = new_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)

    #     self.actor_optim.zero_grad()
    #     set_trace()
    #     action_batch_n, logit = self.select_action(
    #         state_batch.view(-1, self.obs_dim), action_noise=self.train_noise, grad=True)
    #     action_batch_n = action_batch_n.view(-1, self.n_action * self.n_agent)


    #     set_trace()
    #     policy_loss = -self.critic(state_batch, action_batch_n)
    #     policy_loss = policy_loss.mean() + 1e-3 * (logit ** 2).mean()
    #     policy_loss.backward()
    #     #clip_grad_norm_(self.actor.parameters(), 0.00000001)
    #     clip_grad_norm_(self.actor_params, 0.5)
    #     self.actor_optim.step()

    #     soft_update(self.actor_target, self.actor, self.tau)
    #     soft_update(self.critic_target, self.critic, self.tau)

    #     return policy_loss.item()


    # ERASEME: Moved to PICMAC
    # def perturb_actor_parameters(self, param_noise):
    #     """Apply parameter noise to actor model, for exploration"""
    #     hard_update(self.actor_perturbed, self.actor)
    #     params = self.actor_perturbed.state_dict()
    #     for name in params:
    #         if 'ln' in name:
    #             pass
    #         param = params[name]
    #         param += th.randn(param.shape) * param_noise.current_stddev

    # ERASEME: Moved to PICLearner
   #  def save_models(self, path):
   #      th.save(self.actor.state_dict(), "agent.th".format(path))
   #      th.save(self.critic.state_dict(), "critic.th".format(path))

   #      th.save(self.actor_target.state_dict(), "agent_target.th".format(path))
   #      th.save(self.critic_target.state_dict(), "critic_target.th".format(path))

   #      th.save(self.actor_optim.state_dict(), "agent_opt.th".format(path))
   #      th.save(self.critic_optim.state_dict(), "critic_opt.th".format(path))

    # ERASEME: Moved to PICLearner
    # def load_models(self, path):
    #     self.actor.load_state_dict(th.load(path))
    #     self.critic.load_state_dict(th.load(path))

    #     self.actor.load_state_dict(
    #         th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
    #     self.critic.load_state_dict(
    #         th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))

    #     self.actor.load_state_dict(
    #         th.load("{}/agent_target.th".format(path), map_location=lambda storage, loc: storage))
    #     self.critic.load_state_dict(
    #         th.load("{}/critic_target.th".format(path), map_location=lambda storage, loc: storage))

    #     self.actor_optim.load_state_dict(
    #         th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
    #     self.critic_optim.load_state_dict(
    #         th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))

        
    # ERASEME: Moved to PICLearner
    # def _update_targets_hard(self):
    #     hard_update(self.actor_target, self.actor, self.tau)
    #     hard_update(self.critic_target, self.critic, self.tau)


    # ERASEME: Moved to PICLearner
    # def _update_targets_soft(self):
    #     soft_update(self.actor_target, self.actor, self.tau)
    #     soft_update(self.critic_target, self.critic, self.tau)

    # ERASEME: Moved to PICLearner
    # @property
    # def actor_lr(self):
    #     return self.actor_optim.param_groups[0]['lr']

class PICLearner:
    """Thin wrapper over DDPG"""

    def __init__(self, mac, scheme, logger, args):
        # Conventional learner parmeters
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        # PIC paramters 
        self.args.hidden_size = self.args.hidden_dim
        self.args.obs_dim = scheme['obs']['vshape'] + scheme['actions_onehot']['vshape'][0]
        if self.args.obs_agent_id:
            self.args.obs_dim += self.n_agents
        self.args.obs_dims = [self.args.obs_dim] * self.n_agents
        self.args.actor_lr = args.lr
        self.args.critic_lr = args.lr
        self.args.fixed_lr = False
        self.args.critic_type = args.critic_type
        self.args.actor_type = 'mlp'
        self.args.train_noise = False
        self.args.critic_dec_cen = 'cen'
        self.args.target_update_mode = 'soft' if self.args.target_update_interval_or_tau < 1 else 'hard'
        self.args.exp_save_dir = f"results/PIC/{args.env_args['key']}_{args.env_args['seed']}"
        self.device = "cuda" if args.use_cuda else "cpu"


        # Instantiate controller and critic
        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = Critic(self.args.hidden_dim, self.args.obs_dim,
                             1, self.n_agents, self.args.critic_type, 0, use_cuda=self.args.use_cuda).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.agent_optimiser = Adam(params=self.agent_params, lr=self.args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=self.args.lr)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.last_target_update_episode = 0

        # if self.args.standardise_returns:
        #     self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        # if self.args.standardise_rewards:
        #     self.rew_ms = RunningMeanStd(shape=(1,), device=device)
        # TODO: Is it important to adjust the weights between transitions? (online)


        # Imported args
        # self.args.batch_size = 128
        # self.args.updates_per_step = 8
        # self.args.steps_per_actor_update = 100
        # self.args.steps_per_critic_update = 100
        # self.args.critic_updates_per_step = 8
        # self.args.shuffle = None
        # self.args.eval_freq = 1000

        # TODO: Initialize DDPG variables
        # self.agent = DDPG(self.args.gamma, self.args.tau, self.args.hidden_size, self.args.obs_dim, self.n_actions,
        #                   self.n_agents, self.args.obs_dims, 0, self.args.actor_lr, self.args.critic_lr,
        #                   self.args.fixed_lr, self.args.critic_type, self.args.actor_type, self.args.train_noise,
        #                   self.args.num_episodes, self.args.num_steps, self.args.critic_dec_cen,
        #                   target_update_mode=self.args.target_update_mode, device='cpu')

        # self.eval_agent = DDPG(self.args.gamma, self.args.tau, self.args.hidden_size, self.args.obs_dim, self.args.n_actions,
        #                        self.args.n_agents, self.args.obs_dims, 0, self.args.actor_lr, self.args.critic_lr,
        #                        self.args.fixed_lr, self.args.critic_type, self.args.actor_type, self.args.train_noise,
        #                        self.args.num_episodes, self.args.num_steps, self.args.critic_dec_cen,
        #                        target_update_mode=self.args.target_update_mode, device='cpu')
        # self.args = args
        # self.n_agents = args.n_agents
        # self.n_actions = args.n_actions
        # self.logger = logger

    """PIC methods"""
    @property
    def actor_lr(self):
        return self.agent_optimiser.param_groups[0]['lr']

    def lambda1(self, step):
        start_decrease_step = ((self.args.num_episodes / 2)
                               * self.args.num_steps) / 100
        max_step = (self.args.num_episodes * self.args.num_steps) / 100
        return 1 - ((step - start_decrease_step) / (
                max_step - start_decrease_step)) if step > start_decrease_step else 1

    def adjust_lr(self, i_episode):
        adjust_lr(self.agent_optimiser, self.args.lr, i_episode, self.args.num_episodes, self.args.start_episode)
        adjust_lr(self.critic_optimiser, self.args.lr, i_episode, self.args.num_episodes, self.args.start_episode)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Global
        global global_total_steps
        global global_i_episode
        global global_rewards
        global global_test_q
        global global_best_eval_reward
        global global_best_good_eval_reward
        global global_best_adversary_eval_reward
        # Constants
        transitions_batch_size = 1024

        # Batch data
        obs, nobs = batch["obs"][:, :-1], batch["obs"][:, 1:]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # avail_actions = batch["avail_actions"] Unreferenced

        # Auxiliary functions
        ts = th.tensor_split
        def hn(x): return map(lambda y: y.squeeze(0), x)
        def fn(x): return hn(ts(x, batch.batch_size, dim=0))
        def gn(x): return hn(ts(x, batch.max_seq_length-1, dim=0))
        
        for o1, a1, r1, n1, t1, m1 in zip(fn(obs), fn(actions), fn(rewards),
                                      fn(nobs), fn(terminated), fn(mask)):
            episode_step = 0
            episode_reward = 0
            global_i_episode += 1
            agents_rew = [list()] * self.n_agents
            for o2, a2, r2, n2, t2, m2 in zip(gn(o1), gn(a1), gn(r1),
                                              gn(n1), gn(t1), gn(m1)):
                episode_step += 1
                global_total_steps += 1
                # TODO: Fetch this from the buffer
                terminal = (episode_step >= self.args.num_steps)

                # TODO: double loop to iterate per batch and per time_step
                a3 = a2.view(1, -1) # th.Tensor(a2).view(1, -1)
                # TODO: Change shape?
                m3 = ~m2.bool() # th.Tensor([[not done for done in done_n]])
                n3 = n2.view(1, -1) # th.Tensor(np.concatenate(n2, axis=0)).view(1, -1)
                r3 = r2.unsqueeze(0).mean(dim=-1, keepdims=True).tile((1, self.n_agents)) # r2.mean(dim=-1, keepdims=True)# th.Tensor([r2])
                o3 = o2.view(1, -1) # th.Tensor(np.concatenate(o2, axis=0)).view(1, -1)
                M.push(o3, a3, m3, n3, r3)
                for i, r in enumerate(ts(r2, self.n_agents, dim=-1)):
                    agents_rew[i].append(r)
                episode_reward += np.sum(r2.numpy())
                # obs_n = next_obs_n
                # n_update_iter = 5
                if len(M) > transitions_batch_size:
                    if global_total_steps % self.args.steps_per_actor_update == 0:
                        for _ in range(self.args.updates_per_step):
                            transitions = M.sample(self.args.batch_size)
                            experience = Transition(*zip(*transitions))
                            policy_loss = self.mac.update_actor_parameters(
                                # experience, i, self.args.shuffle
                                experience,
                                self.agent_optimiser,
                                self.agent_params,
                                self.critic,
                                shuffle=self.args.shuffle
                            )
                        soft_update(self.target_mac.agent, self.mac.agent, self.args.tau)
                        print('episode {}, p loss {}, p_lr {}'.
                              format(global_i_episode, policy_loss, self.actor_lr))
                    if global_total_steps % self.args.steps_per_critic_update == 0:
                        value_losses = []
                        for i in range(self.args.critic_updates_per_step):
                            transitions = M.sample(self.args.batch_size)
                            experience = Transition(*zip(*transitions))
                            value_losses.append(

                                # def update_critic_parameters(
                                #   batch, agent, agent_id, critic_target, critic_optim, critic_params, shuffle=None,
                                self.critic.update_critic_parameters(
                                    experience,
                                    self.mac,
                                    np.random.randint(self.n_agents),
                                    self.target_critic,
                                    self.critic_optimiser,
                                    self.critic_params,
                                    self.args.shuffle
                                )
                            )

                        if self.args.target_update_mode == 'soft':
                            soft_update(self.target_critic, self.critic, self.args.tau)
                        elif self.args.target_update_mode == 'hard':
                            hard_update(self.target_critic, self.critic)
                        value_loss = np.mean(value_losses)
                        print('episode {}, q loss {},  q_lr {}'.
                              format(global_i_episode, value_loss, self.agent.critic_optim.param_groups[0]['lr']))
                        if self.args.target_update_mode == 'episodic':
                            hard_update(self.target_critic, self.critic)

                # if done_n[0] or terminal:
                if terminal:
                    print('train episode reward', episode_reward)
                    episode_step = 0
                    break
        global_rewards.append(episode_reward)
        if not self.args.fixed_lr:
            self.adjust_lr(global_i_episode)

        if (global_i_episode + 1) % self.args.eval_freq == 0:
            tr_log = {'num_adversary': 0,
                      'best_good_eval_reward': global_best_good_eval_reward,
                      'best_adversary_eval_reward': global_best_adversary_eval_reward,
                      'exp_save_dir': exp_save_dir, 'total_steps': global_total_steps,
                      'value_loss': value_loss, 'policy_loss': policy_loss,
                      'i_episode': global_i_episode, 'start_time': start_time}
            copy_actor_policy(self.agent, self.eval_agent)
            global_test_q.put([self.eval_agent, tr_log])

        if (global_i_episode + 1) % self.args.eval_freq == 0 or t_env >= self.args.t_max:
            done_training.value = True
            eval_model_q(global_test_q, done_training, self.args)


    def save_models(self, path):
        self.mac.save_models(path)
        self.target_mac.save_models(path)
        th.save(self.agent_optimiser.state_dict(), "agent_opt.th".format(path))

        th.save(self.critic.state_dict(), "critic.th".format(path))
        th.save(self.target_critic.state_dict(), "target_critic.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.agent_optimiser.load_state_dict(
            th.load(f"{path}/agent_opt.th")
        )
        self.critic.load_state_dict(th.load(path))
        self.target_critic.load_state_dict(
            th.load(f"{path}/target_critic.th")
        )
        self.critic_optimiser.load_state_dict(
            th.load(f"{path}/critic_opt.th")
        )

    def _update_targets_hard(self):
        hard_update(self.target_mac.agent, self.mac.agent)
        hard_update(self.target_critic, self.critic)


    def _update_targets_soft(self):
        soft_update(self.target_mac.agent, self.mac.agent, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    # TODO: Log somewhere PIC code
    # TODO: Use sacred log from MADDPG (how and when to log?)


# import cop
# from components.episode_buffer import EpisodeBatch
# import torch as th
# from torch.optim import RMSprop, Adam
# from controllers.maddpg_controller import gumbel_softmax
# from modules.critics import REGISTRY as critic_registry
# from components.standarize_stream import RunningMeanStd
# class PICLearner:
#     def __init__(self, mac, scheme, logger, args):
#         self.args = args
#         self.n_agents = args.n_agents
#         self.n_actions = args.n_actions
#         self.logger = logger
# 
#         self.mac = mac
#         self.target_mac = copy.deepcopy(self.mac)
#         self.agent_params = list(mac.parameters())
# 
#         self.critic = critic_registry[args.critic_type](scheme, args)
#         self.target_critic = copy.deepcopy(self.critic)
#         self.critic_params = list(self.critic.parameters())
# 
#         self.agent_optimiser = Adam(params=self.agent_params, lr=self.args.lr)
#         self.critic_optimiser = Adam(params=self.critic_params, lr=self.args.lr)
# 
#         self.log_stats_t = -self.args.learner_log_interval - 1
# 
#         self.last_target_update_episode = 0
# 
#         device = "cuda" if args.use_cuda else "cpu"
#         if self.args.standardise_returns:
#             self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
#         if self.args.standardise_rewards:
#             self.rew_ms = RunningMeanStd(shape=(1,), device=device)
# 
#     def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
#         # Get the relevant quantities
#         rewards = batch["reward"][:, :-1]
#         actions = batch["actions_onehot"]
#         terminated = batch["terminated"][:, :-1].float()
#         rewards = rewards.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
#         terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
#         mask = 1 - terminated
#         batch_size = batch.batch_size
# 
#         if self.args.standardise_rewards:
#             self.rew_ms.update(rewards)
#             rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
# 
#         # Train the critic
#         inputs = self._build_inputs(batch)
#         # actions = actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
#         q_taken = self.critic(inputs[:, :-1], actions[:, :-1].detach())
#         q_taken = q_taken.view(batch_size, -1, 1).tile((1, 1, self.n_agents))
# 
#         # Use the target actor and target critic network to compute the target q
#         self.target_mac.init_hidden(batch.batch_size)
#         target_actions = []
#         for t in range(1, batch.max_seq_length):
#             agent_target_outs = self.target_mac.target_actions(batch, t)
#             target_actions.append(agent_target_outs)
#         target_actions = th.stack(target_actions, dim=1)  # Concat over time
# 
#         # target_actions = target_actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
#         target_vals = self.target_critic(inputs[:, 1:], target_actions.detach())
#         target_vals = target_vals.view(batch_size, -1, 1)
#         target_vals = target_vals.tile((1, 1, self.n_agents)) # [32, 50, 1] -> [32, 50, 3]
# 
#         if self.args.standardise_returns:
#             target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
#         targets = rewards.squeeze(-1) + self.args.gamma * (1 - terminated.squeeze(-1)) * target_vals.detach()
# 
#         if self.args.standardise_returns:
#             self.ret_ms.update(targets)
#             targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)
# 
#         td_error = (q_taken - targets.detach())
#         masked_td_error = td_error * mask.squeeze(-1)
#         loss = (masked_td_error ** 2).mean()
# 
#         self.critic_optimiser.zero_grad()
#         loss.backward()
#         critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
#         self.critic_optimiser.step()
# 
#         # Train the actor
#         self.mac.init_hidden(batch_size)
#         pis = []
#         actions = []
#         for t in range(batch.max_seq_length-1):
#             pi = self.mac.forward(batch, t=t).view(batch_size, 1, self.n_agents, -1)
#             actions.append(gumbel_softmax(pi, hard=True))
#             pis.append(pi)
# 
#         actions = th.cat(actions, dim=1)
#         # ERASEME: maddpg
#         # actions = actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
# 
#         # TODO: Discover what this block does
#         # new_actions = []
#         # for i in range(self.n_agents):
#         #     temp_action = th.split(actions[:, :, i, :], self.n_actions, dim=2)
#         #     actions_i = []
#         #     for j in range(self.n_agents):
#         #         if i == j:
#         #             actions_i.append(temp_action[j])
#         #         else:
#         #             actions_i.append(temp_action[j].detach())
#         #     actions_i = th.cat(actions_i, dim=-1)
#         #     new_actions.append(actions_i.unsqueeze(2))
#         # new_actions = th.cat(new_actions, dim=2)
# 
#         pis = th.cat(pis, dim=1)
#         pis[pis==-1e10] = 0
#         # pis = pis.reshape(-1, 1)
#         # q = self.critic(inputs[:, :-1], new_actions)
#         q = self.critic(inputs[:, :-1], actions).unsqueeze(-1).tile((1, 1, self.n_agents))
#         q = q.unsqueeze(-1).tile((1, 1, 1, self.n_actions))
# 
#         # q = q.reshape(-1, 1)
#         # mask = mask.reshape(-1, 1)
# 
#         # Compute the actor loss
#         pg_loss = -(q * mask.tile(1, 1, 1, self.n_actions)).mean() + self.args.reg * (pis ** 2).mean()
# 
#         # Optimise agents
#         self.agent_optimiser.zero_grad()
#         pg_loss.backward()
#         agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
#         self.agent_optimiser.step()
# 
#         if self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0:
#             self._update_targets_hard()
#             self.last_target_update_episode = episode_num
#         elif self.args.target_update_interval_or_tau <= 1.0:
#             self._update_targets_soft(self.args.target_update_interval_or_tau)
# 
#         if t_env - self.log_stats_t >= self.args.learner_log_interval:
#             self.logger.log_stat("critic_loss", float(loss.item()), t_env)
#             self.logger.log_stat("critic_grad_norm", float(critic_grad_norm.item()), t_env)
#             self.logger.log_stat("agent_grad_norm", float(agent_grad_norm.item()), t_env)
#             mask_elems = mask.sum().item()
#             self.logger.log_stat("td_error_abs", float(masked_td_error.abs().sum().item() / mask_elems), t_env)
#             self.logger.log_stat("q_taken_mean", float((q_taken).sum().item() / mask_elems), t_env)
#             self.logger.log_stat("target_mean", float(targets.sum().item() / mask_elems), t_env)
#             self.logger.log_stat("pg_loss", float(pg_loss.item()), t_env)
#             self.logger.log_stat("agent_grad_norm", float(agent_grad_norm), t_env)
#             self.log_stats_t = t_env
# 
# 
#     def _build_inputs(self, batch, t=None):
#         bs = batch.batch_size
#         max_t = batch.max_seq_length if t is None else 1
#         ts = slice(None) if t is None else slice(t, t + 1)
# 
#         inputs = []
#         # inputs.append(batch["state"][:, ts].unsqueeze(2).expand(-1, -1, self.n_agents, -1))
#         inputs.append(batch["obs"][:, ts])
#         # if self.args.obs_individual_obs:
#         #     inputs.append(batch["obs"][:, ts])
# 
#         # last actions
#         # if self.args.obs_last_action:
#         #     if t == 0:
#         #         inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
#         #     elif isinstance(t, int):
#         #         inputs.append(batch["actions_onehot"][:, slice(t - 1, t)])
#         #     else:
#         #         last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]],
#         #                               dim=1)
#         #         # last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
#         #         inputs.append(last_actions)
#         if self.args.obs_agent_id:
#             inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
# 
#         inputs = th.cat(inputs, dim=-1)
#         return inputs
# 
#     def _update_targets_hard(self):
#         self.target_mac.load_state(self.mac)
#         self.target_critic.load_state_dict(self.critic.state_dict())
# 
#     def _update_targets_soft(self, tau):
#         for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
#             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
# 
#         for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
#             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
# 
#     def cuda(self):
#         self.mac.cuda()
#         self.target_mac.cuda()
#         self.critic.cuda()
#         self.target_critic.cuda()
# 
#     def save_models(self, path):
#         self.mac.save_models(path)
#         th.save(self.critic.state_dict(), "{}/critic.th".format(path))
#         th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
#         th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
# 
#     def load_models(self, path):
#         self.mac.load_models(path)
#         # Not quite right but I don't want to save target networks
#         self.target_mac.load_models(path)
#         self.agent_optimiser.load_state_dict(
#             th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
