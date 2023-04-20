import numpy as np
import torch as th
from modules.critics.ac_ns import ACCriticNS


class ACCriticDecentralized(ACCriticNS):
    def __init__(self, scheme, args):
        super().__init__(scheme, args)


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
