from functools import partial
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from components.gcn import GraphNet

class ACGCNCriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(ACGCNCriticNS, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)
        if self.args.obs_last_action:
            self.input_shape += self.n_actions
        self.output_type = "q"

        # Set up network layers
        graph_net = partial(GraphNet, self.input_shape, args.hidden_dim, 1,
                            self.n_agents, pool_type=args.pool_type,
                            use_agent_id=args.obs_agent_id)
        self.critics = [graph_net(use_agent_id=i) for i in range(self.n_agents)]
        
    def forward(self, batch, i=None):
        inputs, bs, max_t, adj = self._build_inputs(batch)

        if i is None:
            q_values = []
            for i in range(self.n_agents):
                q_value = self.critics[i](inputs, adj)
                q_values.append(q_value.view(bs, max_t, 1))
            q_values = th.cat(q_values, dim=2)
        else:
            return self.critics[i](inputs, adj).view(bs, max_t, 1)
        return q_values

    def _get_input_shape(self, scheme):
        # obs
        input_shape = scheme["obs"]["vshape"]
        # whether to add the individual observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]

        input_shape += self.n_actions

        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        inputs.append(batch["obs"][:, ts])

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))

        if self.args.obs_last_action:
            # last actions
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1)
                inputs.append(last_actions)

        inputs.append(batch["actions_onehot"][:, ts])
        # inputs = th.cat(inputs, dim=-1)
        # [b, t, n, m1] + [b, t, n, m2] -> [bt, n, m1 + m2]
        inputs = th.cat([x.reshape(bs * max_t, self.n_agents, -1) for x in inputs], dim=-1)
        adj = batch['A_adj'][:, ts].reshape(bs * max_t, self.n_agents, -1)
        return inputs, bs, max_t, adj
