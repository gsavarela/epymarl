'''Like critic decentralized but all neural networks start on the same place
spot.'''
import numpy as np
import torch as th
import torch.nn as nn
from modules.critics.mlp import MLP

class ACCriticNetworked(nn.Module):
    def __init__(self, scheme, args):

        super(ACCriticNetworked, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        mlp = MLP(input_shape, args.hidden_dim, 1)
        self.critics = []
        for _ in range(self.n_agents):
            self.critics.append(MLP(input_shape, args.hidden_dim, 1))
            self.critics[-1].load_state_dict(mlp.state_dict())

    def forward(self, batch, i, t=None, j=None):
        j = i if j is None else j
        inputs, bs, max_t = self._build_inputs(batch, j, t=t)
        q = self.critics[i](inputs)
        q.view(bs, max_t, 1)
        return q

    def parameters(self):
        params = list(self.critics[0].parameters())
        for i in range(1, self.n_agents):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, a in enumerate(self.critics):
            a.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()

    def _build_inputs(self, batch, i, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        inputs = batch["obs"][:, ts, i].clone()

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        return input_shape
