import numpy as np
import torch as th
from modules.critics.ac_ns import ACCriticNS


class ACCriticDecentralized(ACCriticNS):
    def __init__(self, scheme, args):
        super().__init__(scheme, args)

        # For consensus to work we standardize the triplets
        # agents observations are different
        self.standardize_observations = \
                ('lbforaging' in self.args.env_args['key'])

    def forward(self, batch, i, t=None):
        inputs, bs, max_t = self._build_inputs(batch, i, t=t)
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

        # TODO: Complete fruits
        if self.standardize_observations and i > 0:
            # Current agent on the first position
            # aligns inputs and complete observations
            # OBS: on partially observable settings
            # this won't work as the fruits also
            # change locations.
            inputs = self._align_inputs(inputs, i)
            # mask = th.logical_and(inputs <= 0, obs >=0)
            # inputs[mask] = obs[mask]
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        return input_shape


    # Aligns observations for lbforaging where
    # current agent is the first agent triplet.
    def _align_inputs(self, inputs, nswaps):
        wo = inputs.shape[-1] - 3 * self.n_agents     # write offset
        ro = wo + 3
        for _ in range(nswaps): # perform number of swaps
            aux = inputs[:, :, wo: wo + 3].clone()

            inputs[:, :, wo: wo + 3] = inputs[:, :, ro: ro + 3].clone()
            inputs[:, :, ro: ro + 3] = aux
            
            wo += 3
            ro += 3
        return inputs
