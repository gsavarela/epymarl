from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class DAC:
    """Decentralized agent controller

    Guarantees that select_action and forward pass
    Extends non_shared_controller
    happen individually and sequentially.

    init_hidden and parameters
    """
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        # TODO: Make sure avail_actions.shape == agent_outputs.shape
        avail_actions = [
            # x.squeeze()
            x.squeeze(0)
            for x in th.tensor_split(
                ep_batch["avail_actions"][:, t_ep], self.n_agents, dim=1
            )
        ]

        agent_outputs = [
            self.forward(ep_batch, t_ep, i, test_mode=test_mode)
            for i in range(self.n_agents)
        ]
        chosen_actions = [
            self.action_selector.select_action(
                out[bs], avl[bs], t_env, test_mode=test_mode
            )
            for out, avl in zip(agent_outputs, avail_actions)
        ]
        return th.cat([_a.unsqueeze(1) for _a in chosen_actions], dim=1)

    def forward(self, ep_batch, t, i=None, j=None, test_mode=False):
        j = i if j is None else j
        inputs = self._build_inputs(ep_batch, t, j)
        avail = ep_batch["avail_actions"][:, t, i]
        agent_outs, self.hidden_states[i] = self.agent.agents[i](
            inputs, self.hidden_states[i]
        )

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail.reshape(ep_batch.batch_size, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        # May need to squeeze
        return agent_outs

    def init_hidden(self, batch_size):
        # self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav
        # bav
        self.hidden_states = [
            _a.init_hidden().expand(batch_size, -1) for _a in self.agent.agents
        ]  # bav

    def parameters(self):
        return [_a.parameters() for _a in self.agent.agents]

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path, logger=None, save_mongo=False):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

        if logger is not None and save_mongo is True:
            logger.log_model(filepath="{}/agent.th".format(path), name="agent.th")

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t, i):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t, i])  # b1av

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t, i]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1, i])

        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)[i]
                .unsqueeze(0)
                .expand(bs, -1)
            )

        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
