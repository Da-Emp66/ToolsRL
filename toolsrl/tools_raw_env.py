from typing import Literal
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from toolsrl.tools_env import ToolsBaseEnvironment as _env, FPS


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "tools_v1",
        "is_parallelizable": True,
        "render_fps": FPS,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        AECEnv.__init__(self)
        self.env = _env(*args, **kwargs)

        self.agents = ["handyman_" + str(r) for r in range(self.env.num_handymen)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)

        # spaces
        self.action_spaces = dict(zip(self.agents, self.env.per_agent_action_spaces))
        self.observation_spaces = dict(zip(self.agents, self.env.per_agent_observation_spaces))
        self.has_reset = False

        self.render_mode = self.env.render_mode

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env._seed(seed=seed)
        self.has_reset = True
        self.env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

    def close(self):
        if self.has_reset:
            self.env.close()

    def render(self):
        return self.env.render()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        is_last = self._agent_selector.is_last()
        self.env.step(action, self.agent_name_mapping[agent], is_last)

        for r in self.rewards:
            self.rewards[r] = self.env.control_rewards[self.agent_name_mapping[r]]
        if is_last:
            for r in self.rewards:
                self.rewards[r] += self.env.last_rewards[self.agent_name_mapping[r]]

        if self.env.frames >= self.env.max_cycles:
            self.truncations = dict(zip(self.agents, [True for _ in self.agents]))
        else:
            self.terminations = dict(zip(self.agents, self.env.last_dones))

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        return self.env.observe(self.agent_name_mapping[agent])
