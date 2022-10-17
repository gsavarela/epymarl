import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentralVCriticBaseline(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCriticBaseline, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.layer = nn.Linear(input_shape, 1)

        # For consensus to work we standardize the triplets
        # agents observations are different
        self.standardize_observations = \
                ('lbforaging' in self.args.env_args['key'])

    def forward(self, batch, t=None):
        # originally inputs: [bs, t_max, n_players, input_shape]
        inputs, bs, max_t = self._build_inputs(batch, t=t)

        q = self.layer(inputs)
        # preserve 4 dim tensor:[bs, t_max, n_players, 1]
        q = q.view(bs, max_t, 1, -1)

        q = q.repeat(1, 1, self.n_agents, 1)

        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        # inputs = []
        # inputs = batch["state"][:, ts].clone()     # batch, time_max, num players, observation_size
        inputs = batch["obs"][:, ts, 0].clone()     # batch, time_max, num players, observation_size

        # CentralVCriticBaseline has its observations using the union
        # of observations.
        if self.standardize_observations:
            # Current agent on the first position
            # aligns inputs and complete observations
            # OBS: on partially observable settings
            # this won't work as the fruits also
            # change locations.
            for i in range(1, self.n_agents):
                obs = self.align_inputs(batch["obs"][:, ts, i].clone(), i)
                mask = th.logical_and(inputs <= 0, obs >=0)
                inputs[mask] = obs[mask]

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        # TODO: Use observations
        input_shape = scheme["obs"]["vshape"]
        # input_shape = scheme["state"]["vshape"]
        return input_shape

    # Aligns observations for lbforaging where
    # current agent is the first agent triplet.
    def align_inputs(self, inputs, nswaps):
        wo = inputs.shape[-1] - 3 * self.n_agents     # write offset
        ro = wo + 3
        for _ in range(nswaps): # perform number of swaps
            aux = inputs[:, :, wo: wo + 3].clone()

            inputs[:, :, wo: wo + 3] = inputs[:, :, ro: ro + 3].clone()
            inputs[:, :, ro: ro + 3] = aux
            
            wo += 3
            ro += 3
        return inputs
