
"""
Credits
-------
* https://github.com/IouJenLiu/PIC
"""
import torch
import torch.nn as nn
# from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Implements a GCN layer."""

    def __init__(self, input_shape, output_dim):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(input_shape, output_dim)
        self.input_shape = input_shape
        self.output_dim = output_dim

    def forward(self, inputs, input_adj):
        feat = self.fc(inputs)
        out = torch.matmul(input_adj, feat)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_shape) + ' -> ' \
               + str(self.output_dim) + ')'

class GraphNet(nn.Module):
    """Permutation invariant graph net (L=2)
    A graph net that is used to pre-process actions and states, and 
    solve the order issue.
    """

    def __init__(self, input_shape,  hidden_size, output_dim, n_agents,
                 agent_id=0, pool_type='avg', use_agent_id=False):
        super(GraphNet, self).__init__()

        if pool_type not in ('avg', 'max'):
            raise ValueError(f'{pool_type} not expected')
        self.input_shape = input_shape
        self.n_agents = n_agents
        self.pool_type = pool_type
        # TODO: Deprecate this
        # if use_agent_id:
        #     agent_id_attr_dim = 2
        #     self.gc1 = GraphConvLayer(input_shape + agent_id_attr_dim, hidden_size)
        #     self.nn_gc1 = nn.Linear(input_shape + agent_id_attr_dim, hidden_size)
        # else:
        #     self.gc1 = GraphConvLayer(input_shape, hidden_size)
        #     self.nn_gc1 = nn.Linear(input_shape, hidden_size)
        # self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        # self.nn_gc2 = nn.Linear(hidden_size, hidden_size)

        # self.V = nn.Linear(hidden_size, 1)
        # self.V.weight.data.mul_(0.1)
        # self.V.bias.data.mul_(0.1)

        self.gc1 = GraphConvLayer(input_shape, hidden_size)
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        # self.V.weight.data.mul_(0.1)
        # self.V.bias.data.mul_(0.1)


        # Assumes a fully connected graph.
        self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents)) / self.n_agents)
        self.use_agent_id = use_agent_id
        self.agent_id = agent_id

        # if use_agent_id:
        #     self.curr_agent_attr = nn.Parameter(
        #         torch.randn(agent_id_attr_dim), requires_grad=True)
        #     self.other_agent_attr = nn.Parameter(
        #         torch.randn(agent_id_attr_dim), requires_grad=True)

        #     agent_att = []
        #     for k in range(self.n_agents):
        #         if k == self.agent_id:
        #             agent_att.append(self.curr_agent_attr.unsqueeze(-1))
        #         else:
        #             agent_att.append(self.other_agent_attr.unsqueeze(-1))
        #     agent_att = torch.cat(agent_att, -1)
        #     self.agent_att = agent_att.unsqueeze(0)

    def forward(self, x):
        """
        :param x: [batch_size, self.input_shape, self.n_agent] tensor
        :return: [batch_size, self.output_dim] tensor
        """
        # if self.use_agent_id:
        #     agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
        #     x = torch.cat([x, agent_att], 1)

        feat = F.relu(self.gc1(x, self.adj))
        feat += F.relu(self.fc1(x))
        feat /= (1. * self.n_agents)
        out = F.relu(self.gc2(feat, self.adj))
        out += F.relu(self.fc2(feat))
        out /= (1. * self.n_agents)

        # Pooling
        if self.pool_type == 'avg':
            ret = out.mean(1)  # Pooling over the agent dimension.
        else:
            ret, _ = out.max(1)

        # Compute V
        V = self.fc3(ret)
        return V

