"""
This is a custom environment modeled structurally after the WaterWorld environment.
You can find the Waterworld environment here:
https://pettingzoo.farama.org/environments/sisl/waterworld/

Author: Asher B. Gibson
"""

from typing import Optional, Literal

import math
import pymunk
import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import pygame
import yaml
from toolsrl.goals import Goal
from toolsrl.utils import convert_coordinates, get_initial_positions, add_bounding_box, WINDOW_SIZE_X, WINDOW_SIZE_Y
from toolsrl.tools import HandyMan, Tool

DEFAULT_CONFIG = "./configurations/tools_env_config.yaml"
FPS = 15

class ToolsBaseEnvironment:
    def __init__(
            self,
            configuration_filename: str = DEFAULT_CONFIG,
            render_mode: Optional[Literal["human"]] = None,
        ):
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
        self.screen = None
        self.frames = 0
        self.num_handymen = 1
        self.handymen = [HandyMan() for _ in range(self.num_handymen)]
        self.max_acceleration = 1.0

    def __post_init__(self):
        with open(self.configuration_file) as cfg:
            self.description = yaml.safe_load(cfg)
            cfg.close()
        for key in self.description:
            if not self.__dict__.get(key, True):
                self.__setattr__(key, self.description.get(key))

        self.tool_names = [key for key in self.tools]
        self.num_available_tools = len(self.tool_names)
        self.goal_names = [key for key in self.goals]
        self.num_available_goals = len(self.goal_names)
        self.choose_random_goal()
        self.initial_tool_position, self.initial_goal_position = get_initial_positions(self.description, self.goal) 
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
        #  - GOAL FLAG
        #  - GOAL POSITION x, y to touch
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
        obs_for_environment = self.max_environment_objects * (2 + 2 * self.max_vertices_per_environment_object + 3 + 9)
        obs_dim = 5 + obs_for_tool + obs_for_environment

        observation_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        action_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(3,),
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
        selected = np.random.choice(self.description['goals'].keys())
        self.goal = Goal(self.space, {selected: self.description['goals'][selected]}, self.initial_goal_pos, convert_coordinates)

    def create(self):
        """Add all moving objects to PyMunk space."""
        for idx, _ in enumerate(self.handymen):
            tool_name = self.tool_names[int(np.random.random() * self.num_available_tools)]
            self.handymen[idx].current_tool = Tool(self.space, {tool_name: self.tools[tool_name]}, self.initial_tool_position)
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
                                    tool_section_segment.shape.collision_type, environment_object_segment.shape.collision_type
                                )
                            )
                            self.handlers[-1].begin = self.pursuer_evader_begin_callback
                            self.handlers[-1].separate = self.pursuer_evader_separate_callback

    def reset(self):
        self.frames = 0

        # Add objects to space
        add_bounding_box(self.space)
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
            # Limit added thrust to self.pursuer_max_accel
            action = action * (self.max_acceleration / thrust)

        handyman = self.handymen[agent_idx]

        # Clip pursuer speed
        _velocity = np.clip(
            handyman.current_tool.body.velocity + action * self.pixel_scale,
            -self.pursuer_speed,
            self.pursuer_speed,
        )

        # Set pursuer speed
        handyman.accelerate(_velocity[0], _velocity[1])

        # Penalize large thrusts
        accel_penalty = self.thrust_penalty * math.sqrt((action**2).sum())

        # Average thrust penalty among all agents, and assign each agent global portion designated by (1 - local_ratio)
        self.control_rewards = (
            (accel_penalty / self.num_handymen)
            * np.ones(self.num_handymen)
            * (1 - self.local_ratio)
        )

        # Assign the current agent the local portion designated by local_ratio
        self.control_rewards[agent_idx] += accel_penalty * self.local_ratio

        if is_last:
            self.space.step(1 / self.fps)

            obs_list = self.observe_list()
            self.last_obs = obs_list

            for idx in range(self.num_handymen):
                p = self.handymen[idx]

                # reward for food caught, encountered and poison
                self.behavior_rewards[idx] = (
                    self.food_reward * p.shape.food_indicator
                    + self.encounter_reward * p.shape.food_touched_indicator
                    + self.poison_reward * p.shape.poison_indicator
                )

                p.shape.food_indicator = 0
                p.shape.poison_indicator = 0

            rewards = np.array(self.behavior_rewards) + np.array(self.control_rewards)

            local_reward = rewards
            global_reward = local_reward.mean()

            # Distribute local and global rewards according to local_ratio
            self.last_rewards = local_reward * self.local_ratio + global_reward * (
                1 - self.local_ratio
            )

            self.frames += 1

        return self.observe(agent_idx)

    def observe(self, agent_id):
        return np.array(self.last_obs[agent_id], dtype=np.float32)

    def observe_list(self):
        accomplishment_criteria = self.description['goals'][self.goal]['accomplishment-criteria']
        goal_object_name = accomplishment_criteria['object']
        goal_object_position = (accomplishment_criteria['touches']['x'], accomplishment_criteria['touches']['y'])
        # Grab observations regarding goal and environment
        environment_object_positions = [[(-1.0, -1.0), (-1.0, -1.0), -1.0, -1.0, -1.0, (-1.0, -1.0)] for _ in range (self.max_environment_objects)]
        environment_object_properties = [[-1.0, -1.0, -1.0, -1.0, -1.0] for _ in range (self.max_environment_objects)]
        environment_object_points = [[(-1.0, -1.0) for _ in range(self.max_vertices_per_environment_object)] for _ in range (self.max_environment_objects)]
        for i, environment_object in enumerate(self.goal.environment_objects):
            environment_object_positions[i] = [
                environment_object.body.position,
                environment_object.body.velocity,
                environment_object.body.angle % (2 * np.pi),
                environment_object.body.angular_velocity,
                environment_object.name == goal_object_name,
                goal_object_position
            ]
            environment_object_properties[i] = [
                environment_object.weight,
                environment_object.friction,
                environment_object.moment,
                environment_object.type,
                environment_object.shape
            ]
            for j, point in enumerate(environment_object.points):
                environment_object_points[i][j] = (point[0] / self.window_size[0], point[1] / self.window_size[1])
        environment_observations = np.array(list(zip(environment_object_positions, environment_object_properties, environment_object_points))).flatten()

        # Grab observations per handyman
        handyman_observations = []
        for _, handyman in enumerate(self.handymen):
            handyman_positions = [
                handyman.current_tool.orientation,
                handyman.current_tool.body.position,
                handyman.current_tool.body.angle % (2 * np.pi),
                handyman.current_tool.body.angular_velocity,
            ]
        
            # Grab observations regarding tool sections
            section_properties = [[0.0, 0.0, 0.0] for _ in range (self.max_tool_sections)]
            section_points = [[(-1.0, -1.0) for _ in range(self.max_vertices_per_tool_section)] for _ in range (self.max_tool_sections)]
            for i, _ in enumerate(handyman.current_tool.sections):
                section_properties[i] = [
                    handyman.current_tool.sections[i].weight,
                    handyman.current_tool.sections[i].friction,
                    handyman.current_tool.sections[i].moment
                ]
                for j, point in enumerate(handyman.current_tool.sections[i].points):
                    section_points[i][j] = (point[0] / self.window_size[0], point[1] / self.window_size[1])
            handyman_tool_observations = np.array(list(zip(section_properties, section_points))).flatten()

            # Concatenate all observations
            handyman_observation = np.concatenate(
                [
                    handyman_positions,
                    handyman_tool_observations,
                    environment_observations
                ]
            )

            handyman_observations.append(handyman_observation)

        return handyman_observations

    # def pursuer_evader_begin_callback(self, arbiter, space, data):
    #     """Called when a collision between a pursuer and an evader occurs.

    #     The counter of the evader increases by 1, if the counter reaches
    #     n_coop, then, the pursuer catches the evader and gets a reward.
    #     """
    #     pursuer_shape, evader_shape = arbiter.shapes

    #     # Add one collision to evader
    #     evader_shape.counter += 1

    #     # Indicate that food is touched by pursuer
    #     pursuer_shape.food_touched_indicator += 1

    #     if evader_shape.counter >= self.n_coop:
    #         # For giving reward to pursuer
    #         pursuer_shape.food_indicator = 1

    #     return False

    # def pursuer_evader_separate_callback(self, arbiter, space, data):
    #     """Called when a collision between a pursuer and a poison ends.

    #     If at this moment there are greater or equal than n_coop pursuers
    #     that collides with this evader, the evader's position gets reset
    #     and the pursuers involved will be rewarded.
    #     """
    #     pursuer_shape, evader_shape = arbiter.shapes

    #     if evader_shape.counter < self.n_coop:
    #         # Remove one collision from evader
    #         evader_shape.counter -= 1
    #     else:
    #         evader_shape.counter = 0

    #         # For giving reward to pursuer
    #         pursuer_shape.food_indicator = 1

    #         # Reset evader position & velocity
    #         x, y = self._generate_coord(evader_shape.radius)
    #         vx, vy = self._generate_speed(evader_shape.max_speed)

    #         evader_shape.reset_position(x, y)
    #         evader_shape.reset_velocity(vx, vy)

    #     pursuer_shape.food_touched_indicator -= 1

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

        self.screen.fill((255, 255, 255))
        self.draw()
        self.clock.tick(self.fps)

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
