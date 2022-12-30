import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd


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

        # self.optimiser = Adam(params=self.params, lr=args.lr)
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
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

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
            # chosen_action_qvals = th.gather(mac_out[:, :-1], dim=-1, index=_actions.squeeze(-1)).squeeze(3)  # Remove the last dim
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
            with th.no_grad():
                loss_acum += loss
                grad_norm_acum += grad_norm
                masked_td_error_acum += masked_td_error.abs().sum()
                masked_elems_acum += mask.sum()
                chosen_action_qvals_acum += (mask.squeeze(-1) * chosen_action_qvals).sum()
                target_mean_acum += (targets * mask.squeeze(-1)).sum()
                

        self.training_steps += 1
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
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

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
