"""
This is a custom environment modeled structurally after the WaterWorld environment.
You can find the Waterworld environment here:
https://pettingzoo.farama.org/environments/sisl/waterworld/

Author: Asher B. Gibson
"""

import itertools
from typing import Optional, Literal

import math
import pymunk
import pymunk.pygame_util
import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import pygame
import yaml
from toolsrl.goals import Goal
from toolsrl.utils import convert_coordinates, get_initial_positions, add_bounding_box, WINDOW_SIZE_X, WINDOW_SIZE_Y
from toolsrl.tools import HandyMan, Tool

DEFAULT_CONFIG = "./configurations/pz_ppo_config.yaml"
FPS = 15

class ToolsBaseEnvironment:
    def __init__(
            self,
            policy: Literal["mlp", "cnn"],
            configuration_filename: str = DEFAULT_CONFIG,
            render_mode: Optional[Literal["human", "rgb_array"]] = None,
        ):
        print(f"Using {policy.upper()} policy...")
        self.policy = policy
        self.configuration_file = configuration_filename
        self.environment_name = "ToolsRL_v1"
        self.window_size = (WINDOW_SIZE_X, WINDOW_SIZE_Y)
        self.clock = pygame.time.Clock()
        self.space = pymunk.Space()
        self.fps = FPS
        self.handlers = []
        self.tools = None
        self.goals = None
        self.render_mode = render_mode
        if policy == "mlp":
            self.screen = None
        else:
            if self.render_mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode((self.window_size[0], self.window_size[1]))
                pygame.display.set_caption(self.environment_name)
            else:
                self.screen = pygame.Surface((self.window_size[0], self.window_size[1]))

            self.screen.fill((255, 255, 255))
            draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.space.debug_draw(draw_options)
        self.frames = 0
        self.num_handymen = 1
        self.handymen = [HandyMan() for _ in range(self.num_handymen)]
        self.max_acceleration = 1.0

        with open(self.configuration_file) as cfg:
            self.description = yaml.safe_load(cfg)
            cfg.close()
        for key in self.description:
            self.__setattr__(key, self.description.get(key))
        
        self.tool_names = [key for key in self.tools]
        self.num_available_tools = len(self.tool_names)
        self.goal_names = [key for key in self.goals]
        self.num_available_goals = len(self.goal_names)
        self.goal = None
        self.get_spaces()
        self._seed()

    def get_spaces(self):
        """Define the action and observation spaces for all of the agents.

        ############# OBSERVATION SPACE
        ######## Tool
        ### Positions:
        # - orientation (left [0], right [1])
        # - position x, y
        # - angle, angular velocity
        ###### For current Tool
        #### For every ToolSection (with a hard cap at n possible sections)
        ### Properties:
        #  - weight
        #  - friction
        #  - moment
        ### Shapes:
        #  - initial points x, y of tool section (with a hard cap at n possible points)
        ######## Goal
        #### For every EnvironmentObject (with a hard cap at n possible objects)
        ### Positions:
        #  - position (x, y), velocity (vx, vy)
        #  - angle, angular velocity
        #  - Goal flag
        #  - Goal position x, y to touch
        #  - Directional support vector (x, y)
        ### Properties:
        #  - weight
        #  - friction
        #  - moment
        ### Shapes:
        #  - body type (static [0], dynamic [1])
        #  - section type (ball [0], polygon [1], etc.)
        #  - initial points x, y of environment object (with a hard cap at n possible points) OR radius if section type is ball

        ############# ACTION SPACE
        # - velocity vx, vy of tool
        # - orientation (left [0], right [1])

        """

        obs_for_tool = self.max_tool_sections * (3 + 2 * self.max_vertices_per_tool_section)
        obs_for_environment = self.max_environment_objects * (11 + 3 + 2 + 2 * self.max_vertices_per_environment_object)
        obs_dim = 5 + obs_for_tool + obs_for_environment

        observation_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(obs_dim,),
            dtype=np.float32,
        ) if self.policy == "mlp" else spaces.Box(
            low=0,
            high=255,
            shape=(self.window_size[0], self.window_size[1], 3),
            dtype=np.uint8,
        )

        # action_space = spaces.Box(
        #     low=np.float32(-1.0),
        #     high=np.float32(1.0),
        #     shape=(3,),
        #     dtype=np.float32,
        # )
        action_space = spaces.Box(
            low=np.float32(-100.0),
            high=np.float32(100.0),
            shape=(2,),
            dtype=np.float32,
        )

        # Make a copy of the observation space and action
        # space for each agent to see and use, respectively
        self.per_agent_observation_spaces = [observation_space for _ in range(self.num_handymen)]
        self.per_agent_action_spaces = [action_space for _ in range(self.num_handymen)]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def choose_random_goal(self):
        selected = np.random.choice(list(self.description['goals'].keys()))
        self.initial_tool_position, self.initial_goal_position = get_initial_positions(self.description, selected)
        self.goal = Goal(self.space, {selected: self.description['goals'][selected]}, self.initial_goal_position, convert_coordinates, create_now=False)

    def create(self):
        """Add all moving objects to PyMunk space."""
        for idx, _ in enumerate(self.handymen):
            tool_name = self.tool_names[int(np.random.random() * self.num_available_tools)]
            self.handymen[idx].set_tool(Tool(self.space, {tool_name: self.tools[tool_name]}, self.initial_tool_position))
        self.goal.create()

    def add_handlers(self):
        # Collision handlers for handyman tools v.s. environment_objects
        # NOTE: This looks nasty, but every list has only a few items, and this method
        # does not get called often, so it does not significantly impact runtime
        for handyman in self.handymen:
            for section in handyman.current_tool.sections:
                for tool_section_segment in section.shapes:
                    for environment_object in self.goal.environment_objects:
                        for environment_object_segment in environment_object.shapes:
                            self.handlers.append(
                                self.space.add_collision_handler(
                                    tool_section_segment.collision_type, environment_object_segment.collision_type
                                )
                            )
                            self.handlers[-1].begin = self.tool_section_v_environment_object_begin_callback
                            self.handlers[-1].separate = self.tool_section_v_environment_object_separate_callback
        for environment_object in self.goal.environment_objects:
            if environment_object.name == self.goal.accomplishment_criteria['object']:
                for environment_object_segment in environment_object.shapes:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            environment_object_segment.collision_type, self.goal.goal_touch_shape.collision_type
                        )
                    )
                    self.handlers[-1].begin = self.goal_accomplishment_callback

    def reset(self):
        self.frames = 0

        for idx, _ in enumerate(self.handymen):
            if self.handymen[idx].has_tool():
                self.handymen[idx].remove_tool()

        if self.goal is not None:
            self.goal.destroy()

        # Add objects to space
        add_bounding_box(self.space)
        self.choose_random_goal()
        self.create()
        self.add_handlers()

        # Get observation
        obs_list = self.observe_list()

        self.last_rewards = [np.float64(0) for _ in range(self.num_handymen)]
        self.control_rewards = [0 for _ in range(self.num_handymen)]
        self.behavior_rewards = [0 for _ in range(self.num_handymen)]
        self.last_dones = [False for _ in range(self.num_handymen)]
        self.last_obs = obs_list

        return obs_list[0]

    def step(self, action, agent_idx, is_last):
        action = np.asarray(action) * self.max_acceleration
        action = action.reshape(2)
        thrust = np.linalg.norm(action)
        if thrust > self.max_acceleration:
            action = action * (self.max_acceleration / thrust)

        # handyman = self.handymen[agent_idx]

        # _velocity = np.clip(
        #     handyman.current_tool.body.velocity + np.multiply(action, np.array(self.window_size).reshape(2)),
        #     -self.max_velocity,
        #     self.max_velocity,
        # )

        # print(_velocity)

        # handyman.accelerate(_velocity[0], _velocity[1])
        
        action = np.clip(
            np.multiply(action, np.array(self.window_size).reshape(2)), # np.array([self.max_velocity, self.max_velocity]).reshape(2)),
            -self.max_velocity,
            self.max_velocity,
        )
        
        self.handymen[agent_idx].move(self.handymen[agent_idx].current_tool.grip.position[0] + action[0], self.handymen[agent_idx].current_tool.grip.position[1] + action[1])

        # Penalize large thrusts
        accel_penalty = self.thrust_penalty * math.sqrt((action**2).sum())

        # Average thrust penalty among all agents, and assign each agent global portion designated by (1 - agent_selfishness)
        self.control_rewards = (
            (accel_penalty / self.num_handymen)
            * np.ones(self.num_handymen)
            * (1 - self.agent_selfishness)
        )

        # Assign the current agent the local portion designated by agent_selfishness
        self.control_rewards[agent_idx] += accel_penalty * self.agent_selfishness

        if is_last:
            self.space.step(1 / self.fps)

            obs_list = self.observe_list()
            self.last_obs = obs_list

            for idx in range(self.num_handymen):
                p = self.handymen[idx]

                # Add in support vector following
                self.behavior_rewards[idx] = (
                    self.tool_touching_dynamic_object_reward * sum(list(itertools.chain(*[[shape.begin_dynamic_counter for shape in section.shapes] for section in p.current_tool.shapes]))) +
                    self.tool_separating_from_dynamic_object_reward * sum(list(itertools.chain(*[[shape.separate_dynamic_counter for shape in section.shapes] for section in p.current_tool.shapes]))) +
                    self.tool_touching_static_object_reward * sum(list(itertools.chain(*[[shape.begin_static_counter for shape in section.shapes] for section in p.current_tool.shapes]))) +
                    self.tool_separating_from_static_object_reward * sum(list(itertools.chain(*[[shape.separate_static_counter for shape in section.shapes] for section in p.current_tool.shapes]))) +
                    self.goal_reward * self.goal.goal_touch_shape.goal_accomplishment_counter
                )

                for i1, section in enumerate(p.current_tool.shapes):
                    for i2, _ in enumerate(section.shapes):
                        p.current_tool.shapes[i1].shapes[i2].begin_dynamic_counter = 0
                        p.current_tool.shapes[i1].shapes[i2].separate_dynamic_counter = 0
                        p.current_tool.shapes[i1].shapes[i2].begin_static_counter = 0
                        p.current_tool.shapes[i1].shapes[i2].separate_static_counter = 0

            if self.goal.goal_touch_shape.goal_accomplishment_counter == 1:
                self.last_dones = [True for _ in range(self.num_handymen)]

            rewards = np.array(self.behavior_rewards) + np.array(self.control_rewards)

            local_reward = rewards
            global_reward = local_reward.mean()

            # Distribute local and global rewards according to agent_selfishness
            self.last_rewards = local_reward * self.agent_selfishness + global_reward * (
                1 - self.agent_selfishness
            )

            self.frames += 1

        return self.observe(agent_idx)

    def observe(self, agent_id):
        return np.array(self.last_obs[agent_id], dtype=np.float32)

    def observe_list(self):
        if self.policy == "cnn":
            self.screen.fill((255, 255, 255))
            draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.space.debug_draw(draw_options)
            self.clock.tick(self.fps)
            observation = pygame.surfarray.pixels3d(self.screen)
            observation = np.rot90(observation, k=3)
            observation = np.fliplr(observation)
            # print(observation.tolist())
            return [observation for _ in self.handymen]
        
        accomplishment_criteria = self.description['goals'][self.goal.name]['accomplishment-criteria']
        goal_object_name = accomplishment_criteria['object']
        goal_object_position = (accomplishment_criteria['touches']['x'], accomplishment_criteria['touches']['y'])
        goal_object_support_vector = self.goal.get_support_vector()
        # Grab observations regarding goal and environment
        environment_object_positions = [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0] for _ in range (self.max_environment_objects)]
        environment_object_properties = [[-1.0, -1.0, -1.0, -1.0, -1.0] for _ in range (self.max_environment_objects)]
        environment_object_points = [[[-1.0, -1.0] for _ in range(self.max_vertices_per_environment_object)] for _ in range (self.max_environment_objects)]
        for i, environment_object in enumerate(self.goal.environment_objects):
            environment_object_positions[i] = [
                environment_object.body.position[0] / self.window_size[0],
                environment_object.body.position[1] / self.window_size[1],
                environment_object.body.velocity[0] / self.max_velocity,
                environment_object.body.velocity[1] / self.max_velocity,
                (environment_object.body.angle % (2 * np.pi)) / np.pi,
                environment_object.body.angular_velocity / self.scaling['max_angular_velocity'],
                environment_object.name == goal_object_name,
                goal_object_position[0] / self.window_size[0] if environment_object.name == goal_object_name else -1.0,
                goal_object_position[1] / self.window_size[1] if environment_object.name == goal_object_name else -1.0,
                goal_object_support_vector[0] if environment_object.name == goal_object_name else -1.0, # The support vector is already a unit
                goal_object_support_vector[1] if environment_object.name == goal_object_name else -1.0 # vector, so scaling isn't required here
            ]
            environment_object_properties[i] = [
                environment_object.weight / self.scaling['max_weight'],
                environment_object.friction / self.scaling['max_friction'] if environment_object.friction is not None else 1.0,
                environment_object.body.moment / self.scaling['max_moment'] if environment_object.body.moment != float('inf') else 10000.0,
                ['static', 'dynamic'].index(environment_object.type),
                ['polygon', 'circle'].index(environment_object.shape)
            ]
            for j, point in enumerate(environment_object.points):
                environment_object_points[i][j] = (point[0] / self.window_size[0], point[1] / self.window_size[1])
        
        environment_observations = np.concatenate(
            [
                np.array(environment_object_positions, dtype=np.float64).flatten(),
                np.array(environment_object_properties, dtype=np.float64).flatten(),
                np.array(environment_object_points, dtype=np.float64).flatten()
            ]
        )

        # Grab observations per handyman
        handyman_observations = []
        for _, handyman in enumerate(self.handymen):
            handyman_positions = np.array([
                handyman.current_tool.orientation,
                handyman.current_tool.body.position[0] / self.window_size[0],
                handyman.current_tool.body.position[1] / self.window_size[1],
                (handyman.current_tool.body.angle % (2 * np.pi)) / np.pi,
                handyman.current_tool.body.angular_velocity / self.scaling['max_angular_velocity'],
            ], dtype=np.float64)
        
            # Grab observations regarding tool sections
            section_properties = [[0.0, 0.0, 0.0] for _ in range (self.max_tool_sections)]
            section_points = [[[-1.0, -1.0] for _ in range(self.max_vertices_per_tool_section)] for _ in range (self.max_tool_sections)]
            for i, _ in enumerate(handyman.current_tool.sections):

                section_properties[i] = [
                    handyman.current_tool.sections[i].weight / self.scaling['max_weight'],
                    handyman.current_tool.sections[i].friction / self.scaling['max_friction'],
                    handyman.current_tool.sections[i].body.moment / self.scaling['max_moment']
                ]
                for j, point in enumerate(handyman.current_tool.sections[i].points):
                    section_points[i][j] = (point[0] / self.window_size[0], point[1] / self.window_size[1])
            handyman_tool_observations = np.concatenate(
                [
                    np.array(section_properties, dtype=np.float64).flatten(),
                    np.array(section_points, dtype=np.float64).flatten()
                ]
            )
            # print(handyman_positions)

            # Concatenate all observations
            handyman_observation = np.concatenate(
                [
                    handyman_positions,
                    handyman_tool_observations,
                    environment_observations
                ]
            )

            handyman_observations.append(handyman_observation)

        # print(handyman_observations)
        # print("Position Obs:")
        # print(handyman_positions)
        # print("Tool Obs:")
        # print(handyman_tool_observations)
        # print("Environment Obs:")
        # print(environment_observations)
    
        return handyman_observations

    def tool_section_v_environment_object_begin_callback(self, arbiter, space, data):
        tool_section_segment_shape, environment_object_shape = arbiter.shapes
        # print(tool_section_segment_shape)
        if environment_object_shape.body.body_type == pymunk.Body.DYNAMIC:
            tool_section_segment_shape.begin_dynamic_counter += 1
        elif environment_object_shape.body.body_type == pymunk.Body.STATIC:
            tool_section_segment_shape.begin_static_counter += 1
        return True

    def tool_section_v_environment_object_separate_callback(self, arbiter, space, data):
        tool_section_segment_shape, environment_object_shape = arbiter.shapes
        # print(tool_section_segment_shape)
        if environment_object_shape.body.body_type == pymunk.Body.DYNAMIC:
            tool_section_segment_shape.separate_dynamic_counter += 1
        elif environment_object_shape.body.body_type == pymunk.Body.STATIC:
            tool_section_segment_shape.separate_static_counter += 1
        return True

    def goal_accomplishment_callback(self, arbiter, space, data):
        _, goal_touch_shape = arbiter.shapes
        goal_touch_shape.goal_accomplishment_counter = 1
        return True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode((self.window_size[0], self.window_size[1]))
                pygame.display.set_caption(self.environment_name)
            else:
                self.screen = pygame.Surface((self.window_size[0], self.window_size[1]))

        if self.policy == "mlp":
            self.screen.fill((255, 255, 255))
            draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.space.debug_draw(draw_options)
            # space.step(1/FPS)
            # pygame.display.flip()
            # clock.tick(int(FPS))
            self.clock.tick(self.fps)

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            # if self.policy == "cnn":
            #     self.clock.tick(self.fps)
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
