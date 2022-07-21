from copy import deepcopy

import numpy as np
import torch as th
import torch.nn.functional as F

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY


# This multi-agent controller shares parameters between agents
class SAC:
    """SingleAgentController (SAC): Models a single RL-based agent

    * Provides a wrapper for the actions emitted from the single agent
    to the environment (embeddings).

    * Delegates actor functions to the agent

    * Requires: RNNAgent CentralVCritic ActorCriticSingleLeaner

    Attributes:
    ----------
    args: Namespace
        The original game settings.

    joint_args: Namespace
        The game settings according to the single agent perspective.

    Methods:
    --------
    select_actions(self, ep_batch, t_ep, t_env, bs, test_mode): torch.LongTensor
        Selects an action for each instance in the ep_batch

    forward(self, ep_batch, t, test_mode): torch.LongTensor
        Performs a forward pass on agent

    init_hidden(self, batch_size: int): None
        Initialize hidden state for the agent

    """ 
    def __init__(self, scheme, groups, args):
        self.n_agents = 1
        self.args = deepcopy(args)
        self.n_players = args.n_agents
        input_shape = self._get_input_shape(scheme)
        self._build_agent(input_shape)
        self._build_embeddings()
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](self.joint_args)
        assert (
            args.action_selector == "soft_policies"
        ), "SAC:action_selectior must be `soft_policies`"
        assert (
            args.mask_before_softmax == False
        ), "SAC:mask_before_softmax not implemented"
        assert (
            args.critic_type == "cv_critic"
        ), "SAC:SingleAgentController requires cv_critic"
        assert (
            args.learner == "actor_critic_single_learner"
        ), "SAC:SingleAgentController requires actor_critic_single_learner"
        self.hidden_states = None

    @property
    def joint_args(self):
        _args = deepcopy(self.args)
        _args.n_actions = _args.n_actions ** _args.n_agents
        _args.n_agents = 1
        return _args

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # This should return a tensor of indexes
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # This should squash logits for unavailable actions.
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        players_actions = F.embedding(chosen_actions, self.embeddings)
        print(players_actions, type(players_actions))
        import ipdb; ipdb.set_trace()
        return players_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = (
            self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        )  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )

    def _build_agent(self, input_shape):
        # Single agent arguments: Leave this to the learner
        # This should be always RNNAgent
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.joint_args)

    def _build_embeddings(self):
        """Embeddings convert joint action into individual actions"""
        base = self.args.n_actions

        def pad(x):
            mp = self.args.n_agents - 1
            if x == 0:
                return mp + 1
            else:
                for i in range(mp, -1, -1):
                    if x >= (self.args.n_actions**i):
                        return mp - i

        # Attention: The most significant agent is the zero-agent
        embs = [
            np.array(list(np.base_repr(na, base=base, padding=pad(na))), dtype=int)
            for na in range(self.joint_args.n_actions)
        ]
        self.embeddings = th.from_numpy(np.vstack(embs))

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["state"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
