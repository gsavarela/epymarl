"""Credits:
PIC: Permutation Invariant Critic for Multi-Agent Deep Reinforcement Learning.
Liu, I., Yeh, R.A.  Schwing, A.G. Proceedings of the Conference on Robot Learning, 

* https://proceedings.mlr.press/v100/liu20a.html
* https://github.com/IouJenLiu/PIC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    source: PIC/maddpg/maddpg_vec.py
    """
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu = self.mu(x)
        return mu
PICAgent = Actor
# class PICAgent(Actor):
#    """Thin wrapper for actor"""
#    def __init__(self, input_shape, args):
#        super(PICAgent, self).__init__(args.hidden_size, input_shape, args.n_actions)
#        self.args = args
# 
#    def init_hidden(self):
#        # make hidden states on same device as model
#        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
