import torch.nn as nn
import numpy as np


import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        q = self.fc2(x)
        return q

class ACCriticShallow(nn.Module):

    def __init__(self, scheme, args):
        super(ACCriticShallow, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.critics = [MLP(input_shape, args.hidden_dim, 1) for _ in range(self.n_agents)]

        # For consensus to work we standardize the triplets
        # agents observations are different
        # self.standardize_observations = \
        #         ('lbforaging' in self.args.env_args['key'])

        # Enable this for linear consensus
        self.standardize_observations = False

    # For lbforaging all agents see the same state regardless
    # Usually current agent has the view shifted.
    def forward(self, batch, i, t=None, j=None):
        j = i if j is None else j
        inputs, bs, max_t = self._build_inputs(batch, j, t=t)

        q = self.critics[i](inputs)
        q.view(bs, max_t, 1)
        return q

    def _build_inputs(self, batch, i, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        if self._joint_observations():
            # batch, time_max, observation_size
            inputs = batch["state"][:, ts].clone()
        else:
             # batch, time_max, num players, observation_size
            inputs = batch["obs"][:, ts, i].clone()
        if self.standardize_observations:
            # Current agent on the first position
            if i > 0:
                wo = inputs.shape[-1] - 3 * self.n_agents     # write offset
                ro = wo + 3
                for _ in range(i): # perform number of swaps
                    aux = inputs[:, :, wo: wo + 3].clone()

                    inputs[:, :, wo: wo + 3] = inputs[:, :, ro: ro + 3].clone()
                    inputs[:, :, ro: ro + 3] = aux
                    
                    wo += 3
                    ro += 3

        return inputs, bs, max_t

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

    def _get_input_shape(self, scheme):
        if self._joint_observations():
            input_shape = scheme["state"]["vshape"]
        else:
            input_shape = scheme["obs"]["vshape"]
        return input_shape

    def _joint_observations(self):
        return hasattr(self.args, 'networked') and self.args.networked \
            and self.args.networked_joint_observations
