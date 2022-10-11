from modules.critics.ac_dec import ACCriticDecentralized
import torch.nn as nn


class ACCriticBaseline(ACCriticDecentralized):

    def __init__(self, scheme, args):
        super(ACCriticDecentralized, self).__init__(scheme=scheme, args=args)

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.critics = [nn.Linear(input_shape, 1) for _ in range(self.n_agents)]
