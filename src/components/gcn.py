
"""Credits
   -------
   * https://github.com/IouJenLiu/PIC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # TODO: Sanity check 

class GraphConvLayer(nn.Module):
    """Implements a GCN layer. """

    def __init__(self, input_shape, output_dim):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(input_shape, output_dim)
        self.input_shape = input_shape
        self.output_dim = output_dim

    def forward(self, inputs, adj):
        """Forward pass on the graph convolutional neural network

        :param inputs: observation or hidden state
        :type: torch.tensor (b,a,m) or (b,t,a,m)
        :param adj: the adjacency matrix
        :type: torch.tensor (b,a,a) or (b,t,a,a)

        :return: hidden representation h_out
        :type: torch.tensor (b,a,out)
        """
        assert len(inputs.shape) in (3, 4)
        assert len(adj.shape) == len(inputs.shape)
        # assert adj.shape.equal(inputs.shape)
        

        # Consolidates accross agent dimension
        oper = 'bij, bim -> bjm' if len(inputs.shape) == 3 else 'btij, btim -> btjm'
        h_out = self.fc(inputs)
        z_out = torch.einsum(oper, adj.float(), h_out)
        return z_out

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
        # self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents).float()) / self.n_agents)
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
        assert len(x.shape) in (3, 4)
        assert len(adj.shape) == len(x.shape)
        # assert adj.shape.equal(inputs.shape)
        

        # Consolidates accross agent dimension
        oper = 'ij, bim -> bjm' if len(x.shape) == 3 else 'ij, btim -> btjm'
        # if self.use_agent_id:
        #     agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
        #     x = torch.cat([x, agent_att], 1)
        # adj = self.adj if adj is None else adj.float()
        
        # TODO: Unite this into one single statement
        # TODO: Change GraphConvLayer for multiple inputs 
        y1 = self.gc1(x, adj)
        y2 = self.fc1(x)
        y3 = torch.einsum(oper, self.mask, y2)

        # ERASEME: Test einsum
        if len(x.shape) == 4:
            with torch.no_grad():
                for ind, ctrl, test in zip(
                        range(self.n_agents),
                        torch.tensor_split(y2, self.n_agents, dim=2),
                        torch.tensor_split(y3, self.n_agents, dim=2)):
                    if ind == self.agent_id:
                        np.testing.assert_almost_equal(test.detach().numpy(), ctrl.detach().numpy())
                    else:
                        np.testing.assert_almost_equal(np.zeros_like(test.detach().numpy()), test.detach().numpy())
        y = F.relu(y1 + y3)
        # feat /= (1. * self.n_agents) # Is this really necessary?
        u = F.relu(self.gc2(y, adj) +
                   torch.einsum(oper, self.mask, self.fc2(y)))
        # out += F.relu(self.fc2(feat))
        # out /= (1. * self.n_agents)# Is this really necessary?
        # Pooling over the agent dimension.
        # [b, t, a, m] -> [b, t, a, m]
        if self.pool_type == 'avg':
            z = u.mean(-2)
        else:
            z, _ = u.max(-2)

        # Compute V squeeze over output
        v = self.fc3(z).squeeze(-1)
        return v
