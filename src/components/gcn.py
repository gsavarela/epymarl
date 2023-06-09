
""" Credits
    -------
    * https://github.com/IouJenLiu/PIC
"""
import torch
import torch.nn as nn
# from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Implements a GCN layer. """

    def __init__(self, input_shape, output_dim):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(input_shape, output_dim)
        self.input_shape = input_shape
        self.output_dim = output_dim

    def forward(self, inputs, adj):
        """ 
        :param inputs: observation of hidden state
        :type: torch.tensor (b,a,m)
        :param adj: the adjacency matrix
        :type: torch.tensor (b,a,a)

        :return: hidden representation h_out
        :type: torch.tensor (b,a,out)
        """
        h = self.fc(inputs)
        out = torch.einsum('bij, bim -> bjm', adj.float(), h)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_shape) + ' -> ' \
               + str(self.output_dim) + ')'

class GraphNet(nn.Module):
    """Permutation invariant graph net (L=2)

    Handles the partially observability case.
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
        self.use_agent_id = use_agent_id    # Uses agent_id as a feature.
        self.agent_id = agent_id
        # TODO: include batch
        
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
        # TODO: Include fc in GCL
        self.gc1 = GraphConvLayer(input_shape, hidden_size)
        self.fc1 = nn.Linear(input_shape, hidden_size)  # W_self
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size) # W_self
        self.fc3 = nn.Linear(hidden_size, output_dim)
        # self.V.weight.data.mul_(0.1)
        # self.V.bias.data.mul_(0.1)


        # Assumes a fully connected graph.
        self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents).float()) / self.n_agents)
        # Indicator for this particular agent
        # Zero matrix with one basis array
        # TODO: Provide this as a parameter 
        # Mask that handles the partially observability case.
        self.register_buffer('mask', torch.diag((torch.eye(n_agents)[:, agent_id])))

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

    def forward(self, x, adj=None):
        """
        :param x: [batch_size, self.input_shape, self.n_agent] tensor
        :return: [batch_size, self.output_dim] tensor
        """
        # if self.use_agent_id:
        #     agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
        #     x = torch.cat([x, agent_att], 1)
        adj = self.adj if adj is None else adj.float()
        
        # TODO: Unite this into one single statement
        # TODO: Change GraphConvLayer for multiple inputs 
        y = F.relu(self.gc1(x, adj) +
                   torch.einsum('ij, bim -> bjm', self.mask, self.fc1(x)))
        # feat /= (1. * self.n_agents) # Is this really necessary?
        u = F.relu(self.gc2(y, adj) +
                   torch.einsum('ij, bim -> bjm', self.mask, self.fc2(y)))
        # out += F.relu(self.fc2(feat))
        # out /= (1. * self.n_agents)# Is this really necessary?

        # Pooling over the agent dimension.
        if self.pool_type == 'avg':
            z = u.mean(1)
        else:
            z, _ = u.max(1)

        # Compute V
        v = self.fc3(z).squeeze(1)
        return v
