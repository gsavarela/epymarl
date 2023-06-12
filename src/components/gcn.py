
"""Credits
   -------
   * https://github.com/IouJenLiu/PIC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # TODO: Sanity check 
from IPython.core.debugger import set_trace

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
        # assert len(adj.shape) == len(inputs.shape)
        # assert adj.shape.equal(inputs.shape)
        
        # oper = 'bij, bim -> bjm' if len(inputs.shape) == 3 else 'btij, btim -> btjm'
        oper = 'ij, bim -> bjm' if len(inputs.shape) == 3 else 'ij, btim -> btjm'
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
        self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents).float()) / self.n_agents)
        # Indicator for this particular agent
        # Zero matrix with one basis array
        # TODO: Provide this as a parameter 
        # Mask that handles the partially observability case.
        # self.register_buffer('mask', torch.diag((torch.eye(n_agents)[:, agent_id])))
        self.register_buffer('eye', torch.eye(n_agents))

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
        assert len(x.shape) in (3, 4)
        # assert len(adj.shape) == len(x.shape)
        # assert adj.shape.equal(inputs.shape)
        

        # Consolidates accross agent dimension
        oper = 'ij, bim -> bjm' if len(x.shape) == 3 else 'ij, btim -> btjm'
        # if self.use_agent_id:
        #     agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
        #     x = torch.cat([x, agent_att], 1)
        # adj = self.adj if adj is None else adj.float()
        
        # TODO: Unite this into one single statement
        # TODO: Change GraphConvLayer for multiple inputs 
        y1 = self.gc1(x, self.adj)
        y2 = self.fc1(x)
        y3 = torch.einsum(oper, self.eye, y2)

        # ERASEME: Test einsum
        # if len(x.shape) == 4:
        #     with torch.no_grad():
        #         for ind, ctrl, test in zip(
        #                 range(self.n_agents),
        #                 torch.tensor_split(y2, self.n_agents, dim=2),
        #                 torch.tensor_split(y3, self.n_agents, dim=2)):
        #             if ind == self.agent_id:
        #                 np.testing.assert_almost_equal(test.detach().numpy(), ctrl.detach().numpy())
        #             else:
        #                 np.testing.assert_almost_equal(np.zeros_like(test.detach().numpy()), test.detach().numpy())
        y = F.relu(y1 + y3)
        # feat /= (1. * self.n_agents) # Is this really necessary?
        u = F.relu(self.gc2(y, self.adj) +
                   torch.einsum(oper, self.eye, self.fc2(y)))
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

"""source: PIC"""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# 
# 
# class GraphConvLayer(nn.Module):
#     """Implements a GCN layer."""
# 
#     def __init__(self, input_dim, output_dim):
#         super(GraphConvLayer, self).__init__()
#         self.lin_layer = nn.Linear(input_dim, output_dim)
#         self.input_dim = input_dim
#         self.output_dim = output_dim
# 
#     def forward(self, input_feature, input_adj):
#         feat = self.lin_layer(input_feature)
#         out = torch.matmul(input_adj, feat)
#         return out
# 
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.input_dim) + ' -> ' \
#                + str(self.output_dim) + ')'
# 
# 
# class MessageFunc(nn.Module):
#     """Implements a Message function"""
# 
#     def __init__(self, input_dim, hidden_size):
#         super(MessageFunc, self).__init__()
#         self.fe = nn.Linear(input_dim, hidden_size)
#         self.input_dim = input_dim
#         self.hidden_size = hidden_size
# 
#     def forward(self, input_feature):
#         """
#         :param x: [batch_size, n_agent, self.sa_dim] tensor
#         :return msg: [batch_size, n_agent * n_agent, output_dim] tensor
#         """
#         n_agent = input_feature.size()[1]
#         bz = input_feature.size()[0]
#         x1 = input_feature.unsqueeze(2).repeat(1, 1, n_agent, 1)
#         x1 = x1.view(bz, n_agent * n_agent, -1)
#         x2 = input_feature.repeat(1, n_agent, 1)
#         x = torch.cat((x1, x2), dim=2)
#         msg = self.fe(x)
#         return msg
# 
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.input_dim) + ' -> ' \
#                + str(self.hidden_size) + ')'
# 
# 
# class UpdateFunc(nn.Module):
#     """Implements a Message function"""
# 
#     def __init__(self, sa_dim, n_agents, hidden_size):
#         super(UpdateFunc, self).__init__()
#         self.fv = nn.Linear(hidden_size + sa_dim, hidden_size)
#         self.input_dim = hidden_size + sa_dim
#         self.output_dim = hidden_size
#         self.n_agents = n_agents
# 
#     def forward(self, input_feature, x, extended_adj):
#         """
#           :param input_feature: [batch_size, n_agent ** 2, self.sa_dim] tensor
#           :param x: [batch_size, n_agent, self.sa_dim] tensor
#           :param extended_adj: [n_agent, n_agent ** 2] tensor
#           :return v: [batch_size, n_agent, hidden_size] tensor
#         """
# 
#         agg = torch.matmul(extended_adj, input_feature)
#         x = torch.cat((agg, x), dim=2)
#         v = self.fv(x)
#         return v
# 
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.input_dim) + ' -> ' \
#                + str(self.output_dim) + ')'
# 
# class GraphNetHetro(nn.Module):
# 
#   # A graph net that supports different edge attributes.
# 
#     def __init__(self, sa_dim, n_agents, hidden_size, agent_groups, agent_id=0,
#                  pool_type='avg', use_agent_id=False):
#         """
#         :param sa_dim: integer
#         :param n_agents: integer
#         :param hidden_size: integer
#         :param agent_groups: list, represents number of agents in each group, agents in the same
#         group are homogeneous. Agents in different groups are heterogeneous
#         ex. agent_groups = [4] --> Group three has has agent 0, agent 1, agent 2, agent 3
#             agent_groups =[2, 3] --> Group one has agent 0, agent 1.
#                                      Group two has agent 2, agent 3, agent 4
#             agent_groups =[2,3,4] --> Group one has agent 0, agent 1.
#                                       Group two has agent 2, agent 3, agent 4.
#                                       Group three has has agent 5, agent 6, agent 7
#         """
#         super(GraphNetHetro, self).__init__()
#         assert n_agents == sum(agent_groups)
# 
#         self.sa_dim = sa_dim
#         self.n_agents = n_agents
#         self.pool_type = pool_type
#         self.agent_groups = agent_groups
# 
#         group_emb_dim = 2  # Dimension for the group embedding.
# 
#         if use_agent_id:
#             agent_id_attr_dim = 2
#             self.gc1 = GraphConvLayer(sa_dim + agent_id_attr_dim, hidden_size)
#             self.nn_gc1 = nn.Linear(sa_dim + agent_id_attr_dim, hidden_size)
#         else:
#             self.gc1 = GraphConvLayer(sa_dim + group_emb_dim, hidden_size)
#             self.nn_gc1 = nn.Linear(sa_dim + group_emb_dim, hidden_size)
#         self.gc2 = GraphConvLayer(hidden_size, hidden_size)
#         self.nn_gc2 = nn.Linear(hidden_size, hidden_size)
# 
#         self.V = nn.Linear(hidden_size, 1)
#         self.V.weight.data.mul_(0.1)
#         self.V.bias.data.mul_(0.1)
# 
#         # Create group embeddings.
#         num_groups = len(agent_groups)
# 
#         self.group_emb = nn.ParameterList([nn.Parameter(torch.randn(1, 1, group_emb_dim), requires_grad=True) for k in range(num_groups)])
# 
#         # Assumes a fully connected graph.
#         self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents)) / self.n_agents)
# 
#         self.use_agent_id = use_agent_id
# 
#         self.agent_id = agent_id
# 
#         if use_agent_id:
#             self.curr_agent_attr = nn.Parameter(
#                 torch.randn(agent_id_attr_dim, 1), requires_grad=True)
#             self.other_agent_attr = nn.Parameter(
#                 torch.randn(agent_id_attr_dim, 1), requires_grad=True)
# 
#             agent_att = []
#             for k in range(self.n_agents):
#                 if k == self.agent_id:
#                     agent_att.append(self.curr_agent_attr.unsqueeze(-1))
#                 else:
#                     agent_att.append(self.other_agent_attr.unsqueeze(-1))
#             agent_att = torch.cat(agent_att, -1)
#             self.agent_att = agent_att.unsqueeze(0)
# 
#     def forward(self, x):
#         """
#         :param x: [batch_size, self.sa_dim, self.n_agent] tensor
#         :return: [batch_size, self.output_dim] tensor
#         """
#         if self.use_agent_id:
#             agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
#             x = torch.cat([x, agent_att], 1)
# 
#         # Concat group embeddings, concat to input layer.
#         group_emb_list = []
#         for k_idx, k in enumerate(self.agent_groups):
#           group_emb_list += [self.group_emb[k_idx]]*k
#         group_emb = torch.cat(group_emb_list, 1)
#         group_emb_batch = torch.cat([group_emb]*x.shape[0], 0)
# 
#         x = torch.cat([x, group_emb_batch], -1)
# 
#         feat = F.relu(self.gc1(x, self.adj))
#         feat += F.relu(self.nn_gc1(x))
#         feat /= (1. * self.n_agents)
#         out = F.relu(self.gc2(feat, self.adj))
#         out += F.relu(self.nn_gc2(feat))
#         out /= (1. * self.n_agents)
# 
#         # Pooling
#         if self.pool_type == 'avg':
#             ret = out.mean(1)  # Pooling over the agent dimension.
#         elif self.pool_type == 'max':
#             ret, _ = out.max(1)
# 
#         # Compute V
#         V = self.V(ret)
#         return V
# 
# 
# class GraphNetV(nn.Module):
# 
#     # A graph net that supports different edge attributes and outputs an vector
# 
#     def __init__(self, sa_dim, n_agents, hidden_size, agent_groups, agent_id=0,
#                  pool_type='avg', use_agent_id=False):
#         """
#         :param sa_dim: integer
#         :param n_agents: integer
#         :param hidden_size: integer
#         :param agent_groups: list, represents number of agents in each group, agents in the same
#         group are homogeneous. Agents in different groups are heterogeneous
#         ex. agent_groups = [4] --> Group three has has agent 0, agent 1, agent 2, agent 3
#             agent_groups =[2, 3] --> Group one has agent 0, agent 1.
#                                      Group two has agent 2, agent 3, agent 4
#             agent_groups =[2,3,4] --> Group one has agent 0, agent 1.
#                                       Group two has agent 2, agent 3, agent 4.
#                                       Group three has has agent 5, agent 6, agent 7
#         """
#         super(GraphNetV, self).__init__()
#         assert n_agents == sum(agent_groups)
# 
#         self.sa_dim = sa_dim
#         self.n_agents = n_agents
#         self.pool_type = pool_type
#         self.agent_groups = agent_groups
# 
#         group_emb_dim = 2  # Dimension for the group embedding.
# 
#         if use_agent_id:
#             agent_id_attr_dim = 2
#             self.gc1 = GraphConvLayer(sa_dim + agent_id_attr_dim, hidden_size)
#             self.nn_gc1 = nn.Linear(sa_dim + agent_id_attr_dim, hidden_size)
#         else:
#             self.gc1 = GraphConvLayer(sa_dim + group_emb_dim, hidden_size)
#             self.nn_gc1 = nn.Linear(sa_dim + group_emb_dim, hidden_size)
#         self.gc2 = GraphConvLayer(hidden_size, hidden_size)
#         self.nn_gc2 = nn.Linear(hidden_size, hidden_size)
# 
#         # Create group embeddings.
#         num_groups = len(agent_groups)
# 
#         self.group_emb = nn.ParameterList([nn.Parameter(torch.randn(1, 1, group_emb_dim), requires_grad=True) for k in range(num_groups)])
# 
#         # Assumes a fully connected graph.
#         self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents)) / self.n_agents)
# 
#         self.use_agent_id = use_agent_id
# 
#         self.agent_id = agent_id
# 
#         if use_agent_id:
#             self.curr_agent_attr = nn.Parameter(
#                 torch.randn(agent_id_attr_dim, 1), requires_grad=True)
#             self.other_agent_attr = nn.Parameter(
#                 torch.randn(agent_id_attr_dim, 1), requires_grad=True)
# 
#             agent_att = []
#             for k in range(self.n_agents):
#                 if k == self.agent_id:
#                     agent_att.append(self.curr_agent_attr.unsqueeze(-1))
#                 else:
#                     agent_att.append(self.other_agent_attr.unsqueeze(-1))
#             agent_att = torch.cat(agent_att, -1)
#             self.agent_att = agent_att.unsqueeze(0)
# 
#     def forward(self, x):
#         """
#         :param x: [batch_size, self.sa_dim, self.n_agent] tensor
#         :return: [batch_size, self.output_dim] tensor
#         """
#         if self.use_agent_id:
#             agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
#             x = torch.cat([x, agent_att], 1)
# 
#         # Concat group embeddings, concat to input layer.
#         group_emb_list = []
#         for k_idx, k in enumerate(self.agent_groups):
#           group_emb_list += [self.group_emb[k_idx]]*k
#         group_emb = torch.cat(group_emb_list, 1)
#         group_emb_batch = torch.cat([group_emb]*x.shape[0], 0)
# 
#         x = torch.cat([x, group_emb_batch], -1)
# 
#         feat = F.relu(self.gc1(x, self.adj))
#         feat += F.relu(self.nn_gc1(x))
#         feat /= (1. * self.n_agents)
#         out = F.relu(self.gc2(feat, self.adj))
#         out += F.relu(self.nn_gc2(feat))
#         out /= (1. * self.n_agents)
# 
#         # Pooling
#         if self.pool_type == 'avg':
#             ret = out.mean(1)  # Pooling over the agent dimension.
#         elif self.pool_type == 'max':
#             ret, _ = out.max(1)
#         return ret
# 
# class GraphNet(nn.Module):
#     """
#     A graph net that is used to pre-process actions and states, and solve the order issue.
#     """
# 
#     def __init__(self, sa_dim, n_agents, hidden_size, agent_id=0,
#                  pool_type='avg', use_agent_id=False):
#         super(GraphNet, self).__init__()
#         self.sa_dim = sa_dim
#         self.n_agents = n_agents
#         self.pool_type = pool_type
#         if use_agent_id:
#             agent_id_attr_dim = 2
#             self.gc1 = GraphConvLayer(sa_dim + agent_id_attr_dim, hidden_size)
#             self.nn_gc1 = nn.Linear(sa_dim + agent_id_attr_dim, hidden_size)
#         else:
#             self.gc1 = GraphConvLayer(sa_dim, hidden_size)
#             self.nn_gc1 = nn.Linear(sa_dim, hidden_size)
#         self.gc2 = GraphConvLayer(hidden_size, hidden_size)
#         self.nn_gc2 = nn.Linear(hidden_size, hidden_size)
# 
#         self.V = nn.Linear(hidden_size, 1)
#         self.V.weight.data.mul_(0.1)
#         self.V.bias.data.mul_(0.1)
# 
#         # Assumes a fully connected graph.
#         self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents)) / self.n_agents)
# 
#         self.use_agent_id = use_agent_id
# 
#         self.agent_id = agent_id
# 
#         if use_agent_id:
#             self.curr_agent_attr = nn.Parameter(
#                 torch.randn(agent_id_attr_dim), requires_grad=True)
#             self.other_agent_attr = nn.Parameter(
#                 torch.randn(agent_id_attr_dim), requires_grad=True)
# 
#             agent_att = []
#             for k in range(self.n_agents):
#                 if k == self.agent_id:
#                     agent_att.append(self.curr_agent_attr.unsqueeze(-1))
#                 else:
#                     agent_att.append(self.other_agent_attr.unsqueeze(-1))
#             agent_att = torch.cat(agent_att, -1)
#             self.agent_att = agent_att.unsqueeze(0)
# 
#     def forward(self, x):
#         """
#         :param x: [batch_size, self.sa_dim, self.n_agent] tensor
#         :return: [batch_size, self.output_dim] tensor
#         """
#         if self.use_agent_id:
#             agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
#             x = torch.cat([x, agent_att], 1)
# 
#         feat = F.relu(self.gc1(x, self.adj))
#         feat += F.relu(self.nn_gc1(x))
#         feat /= (1. * self.n_agents)
#         out = F.relu(self.gc2(feat, self.adj))
#         out += F.relu(self.nn_gc2(feat))
#         out /= (1. * self.n_agents)
# 
#         # Pooling
#         if self.pool_type == 'avg':
#             ret = out.mean(1)  # Pooling over the agent dimension.
#         elif self.pool_type == 'max':
#             ret, _ = out.max(1)
# 
#         # Compute V
#         V = self.V(ret)
#         return V
# 
# 
# class MsgGraphNet(nn.Module):
#     """
#     A message-passing GNN
#     """
# 
#     def __init__(self, sa_dim, n_agents, hidden_size):
#         super(MsgGraphNet, self).__init__()
#         self.sa_dim = sa_dim
#         self.n_agents = n_agents
# 
#         self.msg1 = MessageFunc(sa_dim * 2, hidden_size)
#         self.msg2 = MessageFunc(hidden_size * 2, hidden_size)
#         self.update1 = UpdateFunc(sa_dim, n_agents, hidden_size)
#         self.update2 = UpdateFunc(sa_dim, n_agents, hidden_size)
# 
#         self.V = nn.Linear(hidden_size, 1)
#         self.non_linear = F.relu  # tanh, sigmoid
#         self.adj = torch.ones(n_agents, n_agents) - \
#                    torch.eye(n_agents)
#         self.register_buffer('extended_adj', self.extend_adj())
# 
#     def extend_adj(self):
#         ret = torch.zeros(self.n_agents, self.n_agents * self.n_agents)
#         for i in range(self.n_agents):
#             for j in range(self.n_agents):
#                 if self.adj[i, j] == 1:
#                     ret[i, j * self.n_agents + i] = 1
#         return ret
# 
#     def forward(self, x):
#         """
#         :param x: [batch_size, self.n_agent, self.sa_dim, ] tensor
#         :return: [batch_size, self.output_dim] tensor
#         """
# 
#         e1 = self.non_linear(self.msg1(x))
#         v1 = self.non_linear(self.update1(e1, x, self.extended_adj))
# 
#         e2 = self.non_linear(self.msg2(v1))
#         v2 = self.non_linear(self.update2(e2, x, self.extended_adj))
#         out = torch.max(v2, dim=1)[0]
# 
#         # Compute V
#         return self.V(out)
# 
# 
# class MsgGraphNetHard(nn.Module):
#     """
#     A message-passing GNN with 3-clique graph, will extend to general case.
#     """
# 
#     def __init__(self, sa_dim, n_agents, hidden_size):
#         super(MsgGraphNetHard, self).__init__()
#         self.sa_dim = sa_dim
#         self.n_agents = n_agents
# 
#         self.fe1 = nn.Linear(sa_dim * 2, hidden_size)
#         self.fe2 = nn.Linear(hidden_size * 2, hidden_size)
# 
#         self.fv1 = nn.Linear(hidden_size + sa_dim, hidden_size)
#         self.fv2 = nn.Linear(hidden_size + sa_dim, hidden_size)
# 
#         self.msg1 = MessageFunc(sa_dim * 2, hidden_size)
#         self.msg2 = MessageFunc(hidden_size * 2, hidden_size)
#         self.update1 = UpdateFunc(sa_dim, n_agents, hidden_size)
#         self.update2 = UpdateFunc(sa_dim, n_agents, hidden_size)
#         self.V = nn.Linear(hidden_size, 1)
#         self.non_linear = F.relu  # tanh, sigmoid
#         self.adj = torch.ones(n_agents, n_agents) - \
#                    torch.eye(n_agents)
#         self.extended_adj = self.extend_adj()
# 
#     def extend_adj(self):
#         ret = torch.zeros(self.n_agents, self.n_agents * self.n_agents)
#         for i in range(self.n_agents):
#             for j in range(self.n_agents):
#                 if self.adj[i, j] == 1:
#                     ret[i, j * self.n_agents + i] = 1
#         return ret
# 
#     def forward(self, x):
#         """
#           :param x: [batch_size, self.n_agent, self.sa_dim, ] tensor
#           :return: [batch_size, self.output_dim] tensor
#         """
#         x = torch.transpose(x, 1, 2)
#         h1_01 = self.non_linear(self.fe1(torch.cat((x[:, :, 0], x[:, :, 1]), dim=1)))
#         h1_02 = self.non_linear(self.fe1(torch.cat((x[:, :, 0], x[:, :, 2]), dim=1)))
# 
#         h1_10 = self.non_linear(self.fe1(torch.cat((x[:, :, 1], x[:, :, 0]), dim=1)))
#         h1_12 = self.non_linear(self.fe1(torch.cat((x[:, :, 1], x[:, :, 2]), dim=1)))
# 
#         h1_20 = self.non_linear(self.fe1(torch.cat((x[:, :, 2], x[:, :, 0]), dim=1)))
#         h1_21 = self.non_linear(self.fe1(torch.cat((x[:, :, 2], x[:, :, 1]), dim=1)))
# 
#         h2_0 = self.non_linear(self.fv1(torch.cat(((h1_10 + h1_20) / 2, x[:, :, 0]), dim=1)))
#         h2_1 = self.non_linear(self.fv1(torch.cat(((h1_01 + h1_21) / 2, x[:, :, 1]), dim=1)))
#         h2_2 = self.non_linear(self.fv1(torch.cat(((h1_12 + h1_02) / 2, x[:, :, 2]), dim=1)))
# 
#         h2_01 = self.non_linear(self.fe2(torch.cat((h2_0, h2_1), dim=1)))
#         h2_02 = self.non_linear(self.fe2(torch.cat((h2_0, h2_2), dim=1)))
# 
#         h2_10 = self.non_linear(self.fe2(torch.cat((h2_1, h2_0), dim=1)))
#         h2_12 = self.non_linear(self.fe2(torch.cat((h2_1, h2_2), dim=1)))
# 
#         h2_20 = self.non_linear(self.fe2(torch.cat((h2_2, h2_0), dim=1)))
#         h2_21 = self.non_linear(self.fe2(torch.cat((h2_2, h2_1), dim=1)))
# 
#         h3_0 = self.non_linear(self.fv2(torch.cat(((h2_10 + h2_20) / 2, x[:, :, 0]), dim=1)))
#         h3_1 = self.non_linear(self.fv2(torch.cat(((h2_01 + h2_21) / 2, x[:, :, 1]), dim=1)))
#         h3_2 = self.non_linear(self.fv2(torch.cat(((h2_02 + h2_12) / 2, x[:, :, 2]), dim=1)))
# 
#         out = torch.max(torch.max(h3_0, h3_1), h3_2)
#         # Compute V
#         return self.V(out)
# 
# 
# class GraphNetNN(nn.Module):
#     """
#     A graph net that is used to pre-process actions and states, and solve the order issue.
#     """
# 
#     def __init__(self, sa_dim, n_agents, hidden_size, agent_id=0,
#                  pool_type='avg', use_agent_id=False):
#         super(GraphNetNN, self).__init__()
#         self.sa_dim = sa_dim
#         self.n_agents = n_agents
#         self.pool_type = pool_type
#         if use_agent_id:
#             agent_id_attr_dim = 2
#             self.gc1 = GraphConvLayer(sa_dim + agent_id_attr_dim, hidden_size)
#             self.nn_gc1 = nn.Linear(sa_dim + agent_id_attr_dim, hidden_size)
#         else:
#             self.gc1 = GraphConvLayer(sa_dim, hidden_size)
#             self.nn_gc1 = nn.Linear(sa_dim, hidden_size)
#         self.gc2 = GraphConvLayer(hidden_size, hidden_size)
#         self.nn_gc2 = nn.Linear(hidden_size, hidden_size)
# 
#         self.V = nn.Linear(hidden_size, 1)
#         self.V.weight.data.mul_(0.1)
#         self.V.bias.data.mul_(0.1)
# 
#         # Assumes a fully connected graph.
#         self.use_agent_id = use_agent_id
# 
#         self.agent_id = agent_id
# 
#         if use_agent_id:
#             self.curr_agent_attr = nn.Parameter(
#                 torch.randn(agent_id_attr_dim), requires_grad=True)
#             self.other_agent_attr = nn.Parameter(
#                 torch.randn(agent_id_attr_dim), requires_grad=True)
# 
#             agent_att = []
#             for k in range(self.n_agents):
#                 if k == self.agent_id:
#                     agent_att.append(self.curr_agent_attr.unsqueeze(-1))
#                 else:
#                     agent_att.append(self.other_agent_attr.unsqueeze(-1))
#             agent_att = torch.cat(agent_att, -1)
#             self.agent_att = agent_att.unsqueeze(0)
# 
#     def forward(self, x, adj):
#         """
#         :param x: [batch_size, self.sa_dim, self.n_agent] tensor
#         :return: [batch_size, self.output_dim] tensor
#         """
#         if self.use_agent_id:
#             agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
#             x = torch.cat([x, agent_att], 1)
# 
#         feat = F.relu(self.gc1(x, adj))
#         feat += F.relu(self.nn_gc1(x))
#         feat /= (1. * self.n_agents)
#         out = F.relu(self.gc2(feat, adj))
#         out += F.relu(self.nn_gc2(feat))
#         out /= (1. * self.n_agents)
# 
#         # Pooling
#         if self.pool_type == 'avg':
#             ret = out.mean(1)  # Pooling over the agent dimension.
#         elif self.pool_type == 'max':
#             ret, _ = out.max(1)
# 
#         # Compute V
#         V = self.V(ret)
#         return V
