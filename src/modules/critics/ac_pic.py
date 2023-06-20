import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from components.gcn import GraphNet

"""Credits:
    source: PIC/maddpg/ddpg_vec.py
PIC: Permutation Invariant Critic for Multi-Agent Deep Reinforcement Learning.
Liu, I., Yeh, R.A.  Schwing, A.G. Proceedings of the Conference on Robot Learning, 


* https://proceedings.mlr.press/v100/liu20a.html
* https://github.com/IouJenLiu/PIC
"""

import torch
import torch.nn as nn

from components.gcn import GraphNet

POOL_TYPE = {'gcn_max': 'max', 'gcn_mean': 'mean'}

class Critic(GraphNet):
    """Critic from PIC

    Implements gcn_mean and gcn_max
    """
    
    def __init__(self, hidden_size, input_shape, num_outputs, n_agents, critic_type='gcn_mean', agent_id=0, use_cuda=False): # ERASEME: , group=None):
        # super(Critic, self).__init__(input_shape, hidden_size, 1, n_agents)
        super(Critic, self).__init__(
            input_shape, hidden_size, 1, n_agents,
            pool_type=POOL_TYPE[critic_type], use_agent_id=False
        )
        assert critic_type in ('gcn_mean', 'gcn_max')

        self.n_agents = n_agents
        self.critic_type = critic_type
        # sa_dim = int((input_shape + num_outputs) / n_agents)
        self.agent_id = agent_id
        self.input_shape = input_shape

        self.device = "cuda" if use_cuda else "cpu"

        # ERASEME: always PIC.
        # self.net_fn = model_factory.get_model_fn(critic_type)
        # self.net = GraphNet(sa_dim, n_agents, hidden_size, pool_type=POOL_TYPE[critic_type])

    def forward(self, inputs, actions):
        bz = inputs.size()[0]
        s_n = inputs.view(bz, self.n_agents, -1)
        a_n = actions.view(bz, self.n_agents, -1)
        x = torch.cat((s_n, a_n), dim=2)
        # V = self.net(x)
        V = super(Critic, self).forward(x)
        return V

    def update_critic_parameters(self, batch, mac, agent_id, critic_target, critic_optim, critic_params, shuffle=None,
                                 eval=False):
        state_batch = Variable(th.cat(batch.state)).to(self.device)
        action_batch = Variable(th.cat(batch.action)).to(self.device)
        reward_batch = Variable(th.cat(batch.reward)).to(self.device)
        mask_batch = Variable(th.cat(batch.mask)).to(self.device)
        next_state_batch = th.cat(batch.next_state).to(self.device)
        if shuffle == 'shuffle':
            rand_idx = np.random.permutation(self.n_agents)
            new_state_batch = state_batch.view(-1, self.n_agents, self.input_shape)
            state_batch = new_state_batch[:, rand_idx, :].view(-1, self.input_shape * self.n_agents)
            new_next_state_batch = next_state_batch.view(-1, self.n_agents, self.input_shape)
            next_state_batch = new_next_state_batch[:, rand_idx, :].view(-1, self.input_shape * self.n_agents)
            new_action_batch = action_batch.view(-1, self.n_agents, self.n_action)
            action_batch = new_action_batch[:, rand_idx, :].view(-1, self.n_action * self.n_agents)


        next_action_batch = mac.select_action(
            next_state_batch.view(-1, self.input_shape), action_noise=self.args.train_noise)
        next_action_batch = next_action_batch.view(-1, self.n_action * self.n_agents)
        next_state_action_values = critic_target(
                next_state_batch, next_action_batch)

        reward_batch = reward_batch[:, agent_id].unsqueeze(1)
        mask_batch = mask_batch[:, agent_id].unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)
        critic_optim.zero_grad()
        state_action_batch = self.forward(state_batch, action_batch)
        perturb_out = 0
        value_loss = ((state_action_batch - expected_state_action_batch) ** 2).mean()
        if eval:
            return value_loss.item(), perturb_out
        value_loss.backward()
        unclipped_norm = th.nn.utils.clip_grad_norm_(critic_params, 0.5)
        critic_optim.step()
        return value_loss.item(), perturb_out, unclipped_norm

ACPIC = Critic
# class ACPIC(Critic):
#     """Thin wrapper for the critic"""
#     def __init__(self, scheme, args):
#         self.args = args
#         input_shape = self._get_input_shape(scheme)
#         self.output_type = "q"
# 
#         super(ACPIC, self).__init__(args.hidden_size, input_shape, 1, args.n_agents, args.critic_type)
# 
#     
#     def _build_inputs(self, batch, t=None):
#         bs = batch.batch_size
#         max_t = batch.max_seq_length if t is None else 1
#         ts = slice(None) if t is None else slice(t, t+1)
#         inputs = batch["obs"][:, ts]
#         inputs.append(batch["actions"][:, ts])
#         inputs = th.concat(inputs, dim=-1)
#         return inputs, bs, max_t
# 
#     def _get_input_shape(self, scheme):
#         # observations
#         input_shape = scheme["obs"]["vshape"]
#         input_shape += scheme["actions"]["vshape"]
#         return input_shape
# 
#     # def parameters(self):
#     #     params = list(self.critics[0].parameters())
#     #     for i in range(1, self.n_agents):
#     #         params += list(self.critics[i].parameters())
#     #     return params
# 
#     # def state_dict(self):
#     #     return [a.state_dict() for a in self.critics]
# 
#     # def load_state_dict(self, state_dict):
#     #     for i, a in enumerate(self.critics):
#     #         a.load_state_dict(state_dict[i])
# 
#     # def cuda(self):
#     #     for c in self.critics:
#     #         c.cuda()
# 

# class ACPIC(GraphNet):
#     def __init__(self, scheme, args):
#         self.args = args
#         self.n_actions = args.n_actions
#         self.n_agents = args.n_agents
#         self.input_shape = self._get_input_shape(scheme)
#         if self.args.obs_last_action:
#             self.input_shape += self.n_actions
#         self.output_type = "q"
# 
#         super(ACPIC, self).__init__(
#             self.input_shape, args.hidden_dim, 1, self.n_agents,
#             pool_type=args.pool_type, use_agent_id=args.obs_agent_id
#         )
# 
#         
#     def forward(self, inputs, actions):
#         inputs = th.cat((inputs, actions), dim=-1)
#         q = super(ACPIC, self).forward(inputs)
#         return q
# 
#     # TODO: Should be equal to individual learner
#     def _get_input_shape(self, scheme):
#         # obs
#         input_shape = scheme["obs"]["vshape"]
#         # whether to add the individual observation
#         # if self.args.obs_individual_obs:
#         #     input_shape += scheme["obs"]["vshape"]
# 
#         input_shape += self.n_actions
# 
#         # agent id
#         if self.args.obs_agent_id:
#             input_shape += self.n_agents
#         return input_shape
# 
#     # def _build_inputs(self, batch, t=None):
#     #     bs = batch.batch_size
#     #     max_t = batch.max_seq_length if t is None else 1
#     #     ts = slice(None) if t is None else slice(t, t+1)
#     #     inputs = []
#     #     # observation
#     #     inputs.append(batch["obs"][:, ts])
# 
#     #     # observations
#     #     if self.args.obs_individual_obs:
#     #         inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))
# 
#     #     if self.args.obs_last_action:
#     #         # last actions
#     #         if t == 0:
#     #             inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
#     #         elif isinstance(t, int):
#     #             inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
#     #         else:
#     #             last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
#     #             last_actions = last_actions.view(bs, max_t, 1, -1)
#     #             inputs.append(last_actions)
# 
#     #     actions_onehot = batch["actions_onehot"]
#     #     inputs = th.cat((inputs, actions_onehot), dim=-1)
#     #     inputs = th.cat([x.reshape(bs * max_t, -1) for x in inputs], dim=1)
#     #     return inputs, bs, max_t
# 
