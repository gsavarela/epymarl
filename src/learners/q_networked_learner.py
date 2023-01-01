from collections import defaultdict
import copy
from operator import itemgetter

import numpy as np
import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from components.consensus import consensus_matrices


class QNetworkedLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents

        # self.params = list(mac.parameters())
        self.params = [dict(_a.named_parameters()) for _a in mac.agent.agents]
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            raise ValueError("Mixer {} not recognised.".format(args.mixer))

        self.optimisers = [
            Adam(params=list(_params.values()), lr=args.lr) for _params in self.params
        ]

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.joint_rewards = self.args.env_args.get("joint_rewards", True)
            if self.joint_rewards:
                self.rew_ms = RunningMeanStd(shape=(1,), device=device)
            else:
                self.rew_ms = RunningMeanStd(shape=(self.n_agents,), device=device)

        # consensus evaluations
        def fn(x):
            return th.from_numpy(x.astype(np.float32))

        n_edges = self.args.networked_edges
        self.cwms = [*map(fn, consensus_matrices(self.n_agents, n_edges))]

        if not self.args.networked_time_varying:
            # test if consensus graph is fully connected
            if  n_edges < (self.n_agents - 1):
                raise ValueError("For fully_connected graphs n_edges >= (n_agents - 1)")

            idx = np.random.randint(0, high=len(self.cwms))
            self.cwms = [self.cwms[idx]]

        self.consensus_rounds = self.args.networked_rounds
        self.consensus_interval = self.args.networked_interval

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        self.mac.init_hidden(batch.batch_size)

        # Logging metrics
        loss_acum = th.tensor(0.0)
        grad_norm_acum = th.tensor(0.0)
        masked_td_error_acum = th.tensor(0.0)
        masked_elems_acum = th.tensor(0.0)
        chosen_action_qvals_acum = th.tensor(0.0)
        target_mean_acum = th.tensor(0.0)
        for _i, _opt, _params, _actions, _avail_actions in zip(
            range(self.n_agents),
            self.optimisers,
            self.params,
            th.tensor_split(actions, self.n_agents, dim=2),
            th.tensor_split(avail_actions, self.n_agents, dim=2),
        ):
            _actions.squeeze_(-1)
            _avail_actions.squeeze_(2)

            mac_out = []
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t, i=_i)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=-1, index=_actions).squeeze(-1)

            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t, i=_i)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            # Mask out unavailable actions
            target_mac_out[_avail_actions[:, 1:] == 0] = -9999999
            # Max over target Q-Values
            if self.args.double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[_avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=-1, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, -1, cur_max_actions).squeeze(-1)
            else:
                target_max_qvals = target_mac_out.max(dim=-1)[0]

            if self.args.standardise_returns:
                target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

            # Calculate 1-step Q-Learning targets
            if self.joint_rewards:
                targets = rewards[:, :, 0] + self.args.gamma * (1 - terminated.squeeze(-1)) * target_max_qvals.detach()
            else:
                targets = rewards[:, :, _i] + self.args.gamma * (1 - terminated.squeeze(-1)) * target_max_qvals.detach()

            if self.args.standardise_returns:
                self.ret_ms.update(targets)
                targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

            # Td-error
            td_error = (chosen_action_qvals - targets.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask.squeeze(-1)

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask.squeeze(-1).sum()

            # Optimise
            _opt.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(list(_params.values()), self.args.grad_norm_clip)
            _opt.step()

            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                with th.no_grad():
                    loss_acum += loss
                    grad_norm_acum += grad_norm
                    masked_td_error_acum += masked_td_error.abs().sum()
                    masked_elems_acum += mask.sum()
                    chosen_action_qvals_acum += (mask.squeeze(-1) * chosen_action_qvals).sum()
                    target_mean_acum += (targets * mask.squeeze(-1)).sum()
                

        self.training_steps += 1
        consensus_log = defaultdict(list)
        if self.training_steps % self.consensus_interval == 0:
            self.consensus_step(batch, mask, consensus_log, t_env)

        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", float(loss_acum.item()), t_env)
            self.logger.log_stat("grad_norm", float(grad_norm_acum.item()), t_env)
            # mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", float(masked_td_error_acum.item() / masked_elems_acum), t_env)
            self.logger.log_stat("q_taken_mean", float(chosen_action_qvals_acum.item() / (masked_elems_acum * self.args.n_agents)), t_env)
            self.logger.log_stat("target_mean", float(target_mean_acum.item() / (masked_elems_acum * self.args.n_agents)), t_env)
            for k, v in consensus_log.items():
                self.logger.log_stat(k, v, t_env)
            self.log_stats_t = t_env


    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def consensus_step(self, batch, mask, running_log, t_env):

        t_max = batch.max_seq_length - 1

        consensus_parameters = {}
        consensus_parameters_logs = {}
        with th.no_grad():
            # Inititialization: Evaluate each Q-Value individually
            mac_out = []
            for _i in range(self.n_agents):
                mac_i_out = []
                for t in range(batch.max_seq_length):
                    if self._joint_observations(): # full observability
                        agent_outs = self.mac.forward(batch, t=t, i=_i)  # [b, a]
                    else: # Assume state is perceived by agent 0
                        agent_outs = self.mac.forward(batch, t=t, i=_i, j=0)  # [b, a]
                    mac_i_out.append(agent_outs)
                mac_out.append(th.stack(mac_i_out, dim=1))  # [b, t, a]
            mac_out = th.stack(mac_out, dim=2)  # Concat over agents [b, t, n, a]
            # Pick the Q-Values for the actions taken by each agent
            cur_max_actions = mac_out[:, 1:].max(dim=-1, keepdim=True)[1]
            max_qvals = th.gather(mac_out, -1, cur_max_actions).squeeze(-1)

            _mask = mask.expand_as(max_qvals)
            max_qvals[_mask == 0] = 0
            consensus_values = (max_qvals.sum(dim=-1, keepdims=True) / th.clamp(_mask.sum(dim=-1, keepdims=True), min=1))
            consensus_values = consensus_values.repeat(1, 1, self.n_agents)


            # Log: Log step 0 if its logging period.
            if t_env - self.log_stats_t >= self.args.learner_log_interval: # LOG.
                for _i in range(self.n_agents):
                    q_mean_batch_player = ((max_qvals * _mask).sum(dim=(0, 1)) / _mask.sum(dim=(0, 1))).numpy().round(6)
                    key = f"q_mean_batch_player_{0}_{_i}"
                    running_log[key] = float(q_mean_batch_player[_i])


            # Init consensus: Get individual weights
            keys = {_k for _keys in map(lambda x: x.keys(), self.params) for _k in _keys}
            for _key in keys:
                consensus_parameters[_key] = [
                    th.stack([*map(itemgetter(_key), self.params)], dim=0)
                ]
                consensus_parameters_logs[_key + f'_0'] = copy.deepcopy(consensus_parameters[_key])
            consensus_metropolis_logs = {}

            # Consensus Loop
            for k in range(self.consensus_rounds):
                if self.args.networked_time_varying:
                    idx = np.random.randint(0, high=len(self.cwms))
                else:
                    idx = 0
                cwm = self.cwms[idx]
                consensus_metropolis_logs[k] = cwm.clone()

                for _key, _weights in consensus_parameters.items():
                    # [n_agents, features_in, features_out]
                    _w = _weights[0].clone()
                    if 'weight' in _key:
                        _w = th.einsum('nm, mij-> nij', cwm, _w)
                    elif 'bias' in _key:
                        _w = th.einsum('nm, mi-> ni', cwm, _w)
                    else:
                        raise ValueError(f'Unknwon parameter type {_key}')

                    consensus_parameters[_key] = [_w]
                    consensus_parameters_logs[_key + f'_{k + 1}'] = [_w]

                # update parameters after consensus
                for _i, _agent in enumerate(self.mac.agent.agents):
                    for _key, _value in _agent.named_parameters():
                        _value.data = th.nn.parameter.Parameter(consensus_parameters[_key][0][_i, :])

                # Log k-th step
                if t_env - self.log_stats_t >= self.args.learner_log_interval:
                    # consolidates episode segregating by player
                    mac_out = []
                    for _i in range(self.n_agents):
                        mac_i_out = []
                        for t in range(batch.max_seq_length):
                            if self._joint_observations():  # full observability
                                agent_outs = self.mac.forward(batch, t=t, i=_i)  # [b, a]
                            else: # Assume state is perceived by agent 0
                                agent_outs = self.mac.forward(batch, t=t, i=_i, j=0)  # [b, a]
                            mac_i_out.append(agent_outs)
                        mac_out.append(th.stack(mac_i_out, dim=1))  # [b, t, a]
                    mac_out = th.stack(mac_out, dim=2)  # Concat over agents [b, t, n, a]
                    # Pick the Q-Values for the actions taken by each agent
                    cur_max_actions = mac_out[:, 1:].max(dim=-1, keepdim=True)[1]
                    max_qvals = th.gather(mac_out, -1, cur_max_actions).squeeze(-1)
                    _mask = mask.expand_as(max_qvals)
                    max_qvals[_mask == 0] = 0
                    q_mean_batch_player = ((max_qvals * _mask).sum(dim=(0, 1)) / _mask.sum(dim=(0, 1))).numpy().round(6)

                    for _i in range(self.n_agents):
                        key = f"q_mean_batch_player_{k + 1}_{_i}"
                        running_log[key] = float(q_mean_batch_player[_i])

                    # Computes the loss for the current iteration.
                    td_error = max_qvals[:, :t_max] - consensus_values[:, :t_max]
                    loss = (td_error**2).sum() / mask.sum()
                    running_log[f"consensus_loss_{k + 1}"] = float(loss.item())

            # # debugging log j
            # # saving all weights takes way too long.
            # if t_env - self.log_stats_t >= self.args.learner_log_interval:
            #     for _k, _w in self._logfilter(consensus_parameters_logs):
            #         for _i in range(self.n_agents):
            #             _wi = _w[0][_i]
            #             # row tensor -- squeeze.
            #             if _wi.shape[0] == 1 and len(_wi.shape) == 2:
            #                 _wi.squeeze_(0)
            #             n = len(_wi.shape)
            #             if n == 1: # 1D tensors OK
            #                 if _wi.shape[0] > 1:
            #                     # samples weights
            #                     for _n in (7,):
            #                         _key = f'{_k}_{_i}_{_n}'
            #                         running_log[_key].append(float(_wi[_n]))
            #                 else:
            #                     _key = f'{_k}_{_i}_0'
            #                     running_log[_key].append(float(_wi))
            #             else:
            #                 # samples weights
            #                 for _n in (7,):
            #                     _key = f'{_k}_{_i}_{_n}'
            #                     running_log[_key].append(float(_wi[_n, 0]))
            return consensus_parameters_logs

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path, save_mongo=False):
        self.mac.save_models(path, self.logger, save_mongo)
        th.save(
            [_opt.state_dict() for _opt in self.optimisers],
            "{}/opt.th".format(path),
        )

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        optimizers = th.load(
            "{}/opt.th".format(path),
            map_location=lambda storage, loc: storage,
        )
        for _opt, _states in zip(self.optimisers, optimizers):
            _opt.load_state_dict(_states)

    def _logfilter(self, params):
        return filter(self._lftr, params.items())

    def _lftr(self, x):
        return ('fc1.' in x[0])

    def _joint_observations(self):
        return self.args.networked_joint_observations
