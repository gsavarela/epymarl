from functools import partial
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.gcn import GraphNet

class ACGCNCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ACGCNCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)
        if self.args.obs_last_action:
            self.input_shape += self.n_actions
        self.output_type = "q"

        # Set up network layers
        self.graph_net = partial(GraphNet, self.input_shape, args.hidden_dim, 1,
                                 self.n_agents, pool_type=args.pool_type,
                                 use_agent_id=args.obs_agent_id)
        
    def forward(self, batch, t=None):
        inputs, _  = self._build_inputs(batch, t=t)
        q = self.graph_net(inputs)
        return q

    # TODO: Shpuld be equal do the infividual learner 
    def _get_input_shape(self, scheme):
        # obs
        input_shape = scheme["obs"]["vshape"]
        # whether to add the individual observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]

        input_shape += self.n_actions * self.n_agents

        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observation
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

        actions_onehot = batch["actions_onehot"]
        inputs = th.cat((inputs, actions_onehot), dim=-1)
        inputs = th.cat([x.reshape(bs * max_t, -1) for x in inputs], dim=1)
        return inputs, bs, max_t
