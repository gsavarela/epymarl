"""Adversarial Attack on NetworkedActorCritic

"Adversarial attacks in consensus-based multi-agent reinforcement learning,"
M. Figura, K. C. Kosaraju and V. Gupta, 2021 American Control Conference (ACC),
New Orleans, LA, USA, 2021, pp. 3050-3055, doi: 10.23919/ACC50511.2021.9483080.
"""
from collections import defaultdict
import copy
from operator import itemgetter

import numpy as np
import torch as th
from torch.optim import Adam


from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry
from modules.critics.mlp import MLP

from components.consensus import consensus_matrices

class JointRewardPredictor(MLP):
    '''Forecast the joint rewards using the private reward'''
    def __init__(self, scheme, args):
        input_shape = self._get_input_shape(scheme, args)
        hidden_dim = args.hidden_dim
        super(JointRewardPredictor, self).__init__(input_shape, hidden_dim, 1)


    def _get_input_shape(self, scheme, args):
        # observations
        input_shape = scheme["obs"]["vshape"]
        # actions
        input_shape += scheme["avail_actions"]["vshape"][0] * args.n_agents
        return input_shape

class ActorCriticAdversarialNetworkedLearner:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        assert hasattr(self.args, 'networked_adversaries')
        assert self.n_agents > self.n_adversaries
        assert self.n_adversaries == 1 # Not validated

        self.is_adversarial = True
        self.joint_rewards = False
        self.mac = mac
        self.agent_params = [dict(_a.named_parameters()) for _a in mac.agent.agents]
        self.agent_optimisers = [
            Adam(params=list(_params.values()), lr=args.lr) for _params in self.agent_params
        ]

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = [dict(_c.named_parameters()) for _c in self.critic.critics]
        self.critic_optimisers = [
            Adam(params=list(_params.values()), lr=args.lr) for _params in self.critic_params
        ]
        # Distributed V
        jrp = JointRewardPredictor(scheme, args)
        self.joint_reward_predictors = [
            copy.deepcopy(jrp) for _ in range(args.n_agents)
        ]
        self.joint_reward_params = [dict(_c.named_parameters()) for _c in self.joint_reward_predictors]
        self.joint_reward_optimisers = [
            Adam(params=list(_params.values()), lr=args.lr) for _params in self.joint_reward_params
        ]

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(self.n_agents,), device=device)

        # consensus evaluations
        def fn(x):
            return th.from_numpy(x.astype(np.float32))

        n_edges = self.args.networked_edges
        self.cwms = [*map(fn, consensus_matrices(self.n_agents, n_edges))]
        self.consensus_rounds = self.args.networked_rounds
        self.consensus_interval = self.args.networked_interval

    @property
    def n_adversaries(self):
        return getattr(self.args, 'networked_adversaries', 0)

    @property
    def n_teamates(self):
        return self.n_agents - self.n_adversaries

    @property
    def adversarial_pertubation(self):
        return getattr(self.args, 'networked_adversarial_noise', 0)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error(
                "Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env)
            )
            return

        mask = mask.repeat(1, 1, self.n_agents)

        critic_mask = mask.clone()

        # This processes each player sequentially.
        td_error, critic_train_stats = self.train_critic_sequential(
            self.critic, self.target_critic, batch, rewards, critic_mask
        )

        j_rewards, critic_train_stats = self.joint_reward_prediction_step(
            rewards, batch, mask, critic_train_stats
        )
        self.mac.init_hidden(batch.batch_size)

        pg_losses = []
        grad_norms = []
        pis = []
        advantages = j_rewards + td_error.detach()

        # initialize hidden states before new batch arrives.
        for _i, _opt, _params, _actions, _advantages, _mask in zip(
            range(self.n_agents),
            self.agent_optimisers,
            self.agent_params,
            th.tensor_split(actions[:, :-1], self.n_agents, dim=2),
            th.tensor_split(advantages.detach(), self.n_agents, dim=2),
            th.tensor_split(mask, self.n_agents, dim=2),
        ):

            _actions.squeeze_(dim=2)
            _advantages.squeeze_(dim=2)
            mac_out = []
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t, i=_i)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            _pi = mac_out
            # Calculate policy grad with mask
            _mask2 = _mask.tile((1, 1, self.n_actions))

            _pi[_mask2 == 0] = 1.0

            _pi_taken = th.gather(_pi, dim=2, index=_actions).squeeze(2)
            _log_pi_taken = th.log(_pi_taken + 1e-10)

            _entropy = -th.sum(_pi * th.log(_pi + 1e-10), dim=-1)

            # alternative
            _pg_loss = (
                -(
                    (_advantages * _log_pi_taken + self.args.entropy_coef * _entropy)
                    * _mask.squeeze(-1)
                ).sum()
                / _mask.sum()
            )

            # Optimise agents
            _opt.zero_grad()
            _pg_loss.backward()
            _grad_norm = th.nn.utils.clip_grad_norm_(
                list(_params.values()),
                self.args.grad_norm_clip
            )
            _opt.step()
            with th.no_grad():
                pg_losses.append(float(_pg_loss.detach().item()))
                grad_norms.append(float(_grad_norm.detach().item()))
                pis.append(_pi.detach())

        self.critic_training_steps += 1

        # After all updates perform consensus round.
        if self.critic_training_steps % self.consensus_interval == 0:
            # Assert that the adversary's critic network weights are the same after consensus.
            self.consensus_step()
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:

            ts_logged = len(critic_train_stats['critic_loss'])
            for prefix in ('', 'adv_', 'team_'):
                for suffix in [
                    'critic_loss',
                    'critic_grad_norm',
                    'td_error_abs',
                    'q_taken_mean',
                    'target_mean',
                ]:
                    key = f'{prefix}{suffix}'
                    self.logger.log_stat(
                        key, float(sum(critic_train_stats[key]) / ts_logged), t_env
                    )

            # debugging critic
            for key in filter(lambda x: 'joint_reward' in x, critic_train_stats.keys()):
                self.logger.log_stat(
                    key, float(sum(critic_train_stats[key])), t_env
                )

            na, nd = self.n_agents, self.n_adversaries
            for owner, ids in zip(
                ('', 'adv_', 'team_'),
                (slice(0, na), slice(0, nd), slice(nd, na))
                ):
                # debugging consensus
                self.logger.log_stat(
                    f"{owner}advantage_mean",
                    float((advantages[:, :, ids] * mask[:, :, ids]).sum().item() / mask[:, :, ids].sum().item()),
                    t_env,
                )
                self.logger.log_stat(f"{owner}pg_loss", sum(pg_losses[ids]), t_env)
                self.logger.log_stat(f"{owner}agent_grad_norm", sum(grad_norms[ids]), t_env)
                self.logger.log_stat(
                    "pi_max",
                    float((th.stack(pis[ids], dim=2).max(dim=-1)[0] * mask[:, :, ids]).sum().item()
                        / mask[:, :, ids].sum().item()),
                    t_env,
                )
            self.log_stats_t = t_env

    # Optimise critic
    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        with th.no_grad():
            target_vals = th.cat([target_critic(batch, _i) for _i in range(self.n_agents)], dim=2)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(
            rewards, mask, target_vals, self.args.q_nstep
        )
        
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        running_log = defaultdict(list)
        t_max = batch.max_seq_length - 1
        losses = []
        grad_norms = []
        masked_td_errors = []
        vs = []

        for _i, _opt, _params, _target, _mask in zip(
            range(self.n_agents),
            self.critic_optimisers,
            self.critic_params,
            th.tensor_split(target_returns, self.n_agents, dim=2),
            th.tensor_split(mask, self.n_agents, dim=2),
        ):

            _v = critic(batch, _i)
            # FIXME: Remove this t_max
            _td_error = _target.detach() - _v[:, :t_max]
            _masked_td_error = _td_error * _mask
            _loss = (_masked_td_error**2).sum() / mask.sum()

            _opt.zero_grad()
            _loss.backward()
            _grad_norm = th.nn.utils.clip_grad_norm_(
                list(_params.values()), self.args.grad_norm_clip
            )
            _opt.step()

            with th.no_grad():
                losses.append(float(_loss.item()))
                grad_norms.append(float(_grad_norm.item()))
                masked_td_errors.append(_masked_td_error)
                vs.append(_v[:, :t_max])

        with th.no_grad():
            masked_td_error = th.cat(masked_td_errors, dim=2)
            v = th.cat(vs, dim=2)


            na, nd = self.n_agents, self.n_adversaries
            for owner, ids in zip(
                ('', 'adv_', 'team_'),
                (slice(0, na), slice(0, nd), slice(nd, na))
                ):
                running_log[f"{owner}critic_grad_norm"].append(sum(grad_norms[ids]))
                running_log[f"{owner}critic_loss"].append(sum(losses[ids]))
                mask_elems = mask[:, :, ids].sum().item()
                running_log[f"{owner}td_error_abs"].append(
                    float(masked_td_error[:, :, ids].abs().sum().item() / mask_elems)
                )
                running_log[f"{owner}q_taken_mean"].append(float((v[:, :, ids] * mask[:, :, ids]).sum().item() / mask_elems))

                running_log[f"{owner}target_mean"].append(
                    float((target_returns[:, :, ids] * mask[:, :, ids]).sum().item() / mask_elems)
                )

        return masked_td_error, running_log

    def joint_reward_prediction_step(self, rewards, batch, mask, running_log):

        # receives actions from all agents
        with th.no_grad():
            all_actions_onehot = batch["actions_onehot"].reshape(
                (batch.batch_size, batch.max_seq_length, -1)
            ).unsqueeze(dim=2).tile((1, 1, self.n_agents, 1))
            inputs = th.cat((batch["obs"], all_actions_onehot), dim=-1)
        t_max = batch.max_seq_length - 1

        total_loss = th.tensor(0.0)
        total_grad_norm = th.tensor(0.0)
        masked_errors = []
        joint_rewards = []

        for _i, _input, _opt, _params, _target, _mask in zip(
            range(self.n_agents),
            th.tensor_split(inputs, self.n_agents, dim=2),
            self.joint_reward_optimisers,
            self.joint_reward_params,
            th.tensor_split(rewards, self.n_agents, dim=2),
            th.tensor_split(mask, self.n_agents, dim=2),
        ):
            _input.squeeze_(dim=2)

            _output = self.joint_reward_predictors[_i](_input)
            # FIXME: Remove this t_max
            _error = _target - _output[:, :t_max]
            _masked_error = _error * _mask
            _loss = (_error**2).sum() / mask.sum()

            _opt.zero_grad()
            _loss.backward()
            _grad_norm = th.nn.utils.clip_grad_norm_(
                list(_params.values()), self.args.grad_norm_clip
            )
            _opt.step()

            with th.no_grad():
                total_loss += _loss
                total_grad_norm += _grad_norm
                masked_errors.append(_masked_error)
                joint_rewards.append(_output[:, :t_max])

        with th.no_grad():
            masked_error = th.cat(masked_errors, dim=2)
            joint_rewards = th.cat(joint_rewards, dim=2)
            
        running_log["joint_reward_loss"].append(float(total_loss.item()))
        running_log["joint_reward_grad_norm"].append(float(total_grad_norm.item()))
        mask_elems = mask.sum().item()
        running_log["joint_reward_error_abs"].append(
            float(masked_error.abs().sum().item() / mask_elems)
        )
        # TODO: Sum agent axis
        running_log["joint_reward_mean"].append(float((joint_rewards * mask).sum().item() / mask_elems))
        running_log["reward_mean"].append(
            float((rewards * mask).sum().item() / mask_elems)
        )
        return joint_rewards, running_log

    def consensus_step(self):

        consensus_parameters = {}
        consensus_parameters_logs = {}

        # auxiliary function
        def emit(x):
            # 'x' is the parameter id
            def fn(iy):
                # 'i' is an agent order
                # 'y' is a dictionary of parameters
                i, y = iy
                ret = itemgetter(x)(y)
                if i < self.n_adversaries:
                    # According to figura 2021 it suffaces per = 0.0
                    per = self.adversarial_pertubation
                    ret = (ret + per * ret.grad.sign()).clone().detach()
                return ret
            return fn

        def broadcast(x):
            return th.stack([*map(emit(x), enumerate(self.critic_params))], dim=0)

        with th.no_grad():

            # 1) Emit parameters for consensus
            # dict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias'])
            keys = {_k for _keys in map(lambda x: x.keys(), self.critic_params) for _k in _keys}
            for _key in keys:

                # gets each of agents parameters by layer _key
                # 'fc1.weight': [a, input, hidden], 'fc2.weight': [a, hidden, hidden], ..
                consensus_parameters[_key] = broadcast(_key)

                consensus_parameters_logs[_key + f'_0'] = copy.deepcopy(consensus_parameters[_key])

            # 2) Perform consensus rounds
            consensus_metropolis_logs = {}
            for k in range(self.consensus_rounds):
                idx = np.random.randint(0, high=len(self.cwms))
                cwm = self.cwms[idx]
                consensus_metropolis_logs[k] = cwm.clone()

                for _key, _weights in consensus_parameters.items():
                    _w = _weights.clone() # [n_agents, features_in, features_out]
                    if 'weight' in _key:
                        _w = th.einsum('nm, mij-> nij', cwm, _w)
                    elif 'bias' in _key:
                        _w = th.einsum('nm, mi-> ni', cwm, _w)
                    else:
                        raise ValueError(f'Unknwon parameter type {_key}')

                    consensus_parameters[_key] = _w
                    consensus_parameters_logs[_key + f'_{k + 1}'] = _w

                # 3) Update parameters after consensus
                for _i, _critic in enumerate(self.critic.critics):
                    if _i >=  self.n_adversaries: # do not update adversaries
                        for _key, _value in _critic.named_parameters():
                            _value.data = th.nn.parameter.Parameter(consensus_parameters[_key][_i, :])

    def nstep_returns(self, rewards, mask, values, nsteps):
        # nstep is a hyperparameter that regulates the number of look aheads
        # example 1: nsteps = 5, t_start = 0
        # R^5_0 = r_0 + (gamma*r_1) + (gamma**2*r_2) + (gamma**3*r_3) + (gamma**4*r_4) + (gamma**5*v_5)
        # example 2: nsteps = 5, t_start = 1
        # R^5_1 = r_1 + (gamma*r_2) + (gamma**2*r_3) + (gamma**3*r_4) + (gamma**4*r_5) + (gamma**5*v_6)
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += (
                        self.args.gamma**step * values[:, t] * mask[:, t]
                    )
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += (
                        self.args.gamma**step * rewards[:, t] * mask[:, t]
                    )
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += (
                        self.args.gamma**step * rewards[:, t] * mask[:, t]
                    )
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path, save_mongo=False):
        self.mac.save_models(path, self.logger, save_mongo)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(
            [_opt.state_dict() for _opt in self.agent_optimisers],
            "{}/agent_opt.th".format(path),
        )
        th.save(
            [_opt.state_dict() for _opt in self.critic_optimisers],
            "{}/critic_opt.th".format(path))

        if save_mongo:
            self.logger.log_model(filepath="{}/opt.th".format(path), name="opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        actor_optimizers = th.load(
            "{}/agent_opt.th".format(path),
            map_location=lambda storage, loc: storage,
        )
        for _opt, _states in zip(self.agent_optimisers, actor_optimizers):
            _opt.load_state_dict(_states)

        critic_optimizers = th.load(
            "{}/critic_opt.th".format(path),
            map_location=lambda storage, loc: storage,
        )
        for _opt, _states in zip(self.critic_optimisers, critic_optimizers):
            _opt.load_state_dict(_states)
