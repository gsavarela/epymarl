from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from IPython.core.debugger import set_trace


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs

def sample_gumbel(shape, eps=1e-20, tens_type=th.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -th.log(-th.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=-1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


# This multi-agent controller shares parameters between agents
class PICMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args

        assert self.args.mac == 'pic_mac'
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type
        self.device = "cuda" if args.use_cuda else "cpu"

        self.action_selector = None

        # self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        """Interactions with the environment"""
        # Only select actions for the selected batch elements in bs
        agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1)
        return chosen_actions

    def select_action(self, state, action_noise=None, param_noise=False, grad=False):
        self.agent.eval()
        if param_noise:
            mu = self.agent_perturbed((Variable(state)))
        else:
            mu = self.agent((Variable(state)))

        self.agent.train()
        if not grad:
            mu = mu.data

        if action_noise:
            noise = np.log(-np.log(np.random.uniform(0, 1, mu.size())))
            try:
                mu -= th.Tensor(noise).to(self.device)
            except (AttributeError, AssertionError):
                mu -= th.Tensor(noise)

        action = F.softmax(mu, dim=1)
        if not grad:
            return action
        else:
            return action, mu

    # ERASEME: MADDPGMAC method
    # def target_actions(self, ep_batch, t_ep):
    #     agent_outputs = self.forward(ep_batch, t_ep)
    #     return onehot_from_logits(agent_outputs)

    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs = self.agent(agent_inputs)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions==0] = -1e10
        return agent_outs

    def init_hidden(self, batch_size):
        # TODO: MLP Agent
        # self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        pass

    def init_hidden_one_agent(self, batch_size):
        # TODO: MLP Agent
        # self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1)  # bav
        pass

    def update_actor_parameters(self, batch, agent_optim, agent_params, critic, shuffle=None):
        state_batch = Variable(th.cat(batch.state)).to(self.device)
        if shuffle == 'shuffle':
            rand_idx = np.random.permutation(self.n_agents)
            new_state_batch = state_batch.view(-1, self.n_agents, self.input_shape)
            state_batch = new_state_batch[:, rand_idx, :].view(-1, self.input_shape * self.n_agents)

        agent_optim.zero_grad()
        action_batch_n, logit = self.select_action(
            state_batch.view(-1, self.input_shape), action_noise=self.args.train_noise, grad=True)

        state_batch = state_batch.view(-1, self.n_agents, self.input_shape)
        action_batch_n = action_batch_n.view(-1, self.n_agents, self.n_actions)
        
        policy_loss = -critic(state_batch, action_batch_n)
        policy_loss = policy_loss.mean() + 1e-3 * (logit ** 2).mean()
        policy_loss.backward()
        th.nn.utils.clip_grad_norm_(agent_params, 0.5)
        agent_optim.step()

        # soft_update(self.actor_target, self.actor, self.tau)
        # soft_update(self.critic_target, self.critic, self.tau)

        return policy_loss.item()


    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.agent_perturbed.state_dict(), "{}/agent_perturbed.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_perturbed.load_state_dict(th.load("{}/agent_perturbed.th".format(path), map_location=lambda storage, loc: storage))

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.agent_perturbed, self.agent)
        params = self.agent_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += th.randn(param.shape) * param_noise.current_stddev

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY['pic'](self.args.hidden_dim, self.input_shape, self.n_actions)
        self.agent_perturbed = agent_REGISTRY['pic'](self.args.hidden_dim, self.input_shape, self.n_actions)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
