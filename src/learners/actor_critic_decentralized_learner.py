from collections import defaultdict
import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class ActorCriticDecentralizedLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

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

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            joint_rewards = self.args.env_args.get("joint_rewards", True)
            if joint_rewards:
                self.rew_ms = RunningMeanStd(shape=(1,), device=device)
            else:
                self.rew_ms = RunningMeanStd(shape=(self.n_agents,), device=device)

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
        advantages, critic_train_stats = self.train_critic_sequential(
            self.critic, self.target_critic, batch, rewards, critic_mask
        )

        self.mac.init_hidden(batch.batch_size)
        pg_loss_acum = th.tensor(0.0)
        grad_norm_acum = th.tensor(0.0)
        joint_pi = []

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
                list(_params.values()), self.args.grad_norm_clip
            )
            _opt.step()
            with th.no_grad():
                pg_loss_acum += _pg_loss.detach()
                grad_norm_acum += _grad_norm.detach()
                joint_pi.append(_pi.detach())

        self.critic_training_steps += 1
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
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )
            # debugging critic
            for key in filter(lambda x: 'v_' in x, critic_train_stats.keys()):
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss_acum, t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm_acum, t_env)
            self.logger.log_stat(
                "pi_max",
                (th.stack(joint_pi, dim=2).max(dim=-1)[0] * mask).sum().item()
                / mask.sum().item(),
                t_env,
            )
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        with th.no_grad():
            target_vals = th.cat([target_critic(batch, _i) for _i in range(self.n_agents)], dim=2)
            # target_vals = [target_critic(batch, _i) for _i in range(self.n_agents)]

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
        total_loss = th.tensor(0.0)
        total_grad_norm = th.tensor(0.0)
        masked_td_errors = []
        vs = []

        for _i, _opt, _params, _target, _mask in zip(
            range(self.n_agents),
            self.critic_optimisers,
            self.critic_params,
            th.tensor_split(target_returns, self.n_agents, dim=2),
            th.tensor_split(mask, self.n_agents, dim=2)
        ):

            _v = critic(batch, _i)
            # FIXME: Remove this t_max
            _td_error = _target.detach() - _v[:, :t_max]
            _td_error[_mask ==0]=0
            _loss = (_td_error**2).sum() / _mask.sum()

            _opt.zero_grad()
            _loss.backward()

            _grad_norm = th.nn.utils.clip_grad_norm_(
                list(_params.values()), self.args.grad_norm_clip
            )
            _opt.step()

            with th.no_grad():
                total_loss += _loss
                total_grad_norm += _grad_norm
                masked_td_errors.append(_td_error)
                vs.append(_v[:, :t_max])

        with th.no_grad():
            masked_td_error = th.cat(masked_td_errors, dim=2)
            v = th.cat(vs, dim=2)
        running_log["critic_loss"].append(total_loss.item())
        running_log["critic_grad_norm"].append(total_grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)


        # consolidates episode segregating by player
        v_taken_mean_player = ((v * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1))).numpy().round(6)
        for _i in range(self.n_agents):
            key = f"v_taken_mean_player_{_i}"
            running_log[key].append(float(v_taken_mean_player[_i]))

        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )
        return masked_td_error, running_log

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

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(
            [_opt.state_dict() for _opt in self.agent_optimisers],
            "{}/agent_opt.th".format(path),
        )
        th.save(
            [_opt.state_dict() for _opt in self.critic_optimisers],
            "{}/critic_opt.th".format(path))

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
