"""
This is a custom environment modeled structurally after the WaterWorld environment.
You can find the Waterworld environment here:
https://pettingzoo.farama.org/environments/sisl/waterworld/

Author: Asher B. Gibson
"""

from typing import Optional, Literal

import pymunk
import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import pygame
import yaml

DEFAULT_CONFIG = "./configurations/tools_env_config.yaml"
FPS = 15

class ToolsBaseEnvironment:
    def __init__(
            self,
            configuration_filename: str = DEFAULT_CONFIG,
            render_mode: Optional[Literal["human"]] = None,
        ):
        self.configuration_file = configuration_filename
        self.pixel_scale = 30 * 25
        self.clock = pygame.time.Clock()
        self.FPS = FPS

        self.handlers = []

        self.tools = None
        self.render_mode = render_mode
        self.screen = None
        self.frames = 0
        self.num_agents = 1

    def __post_init__(self):
        with open(self.configuration_file) as cfg:
            overrides = yaml.safe_load(cfg)
            cfg.close()
        for key in overrides:
            if not self.__dict__.get(key, True):
                self.__setattr__(key, overrides.get(key))

        self.get_spaces()
        self._seed()

    def get_spaces(self):
        """Define the action and observation spaces for all of the agents."""

        ############# OBSERVATION SPACE
        ###### Tool
        # - position x, y
        # - angle, angular velocity
        #### For every ToolSection
        #  - section type (ball [0], polygon [1], etc.)
        ### Properties:
        #  - weight
        #  - friction
        #  - moment
        ### Positions:
        #  - initial points x, y of tool section OR radius if section type is ball
        ###### Goal
        #### For every EnvironmentObject
        #  - body type (static [0], dynamic [1])
        #  - section type (ball [0], polygon [1], etc.)
        #  - initial points x, y of environment object OR radius if section type is ball
        ### Properties:
        #  - weight
        #  - friction
        #  - moment
        ### Positions:
        #  - angle, angular velocity
        #  - position (x, y), velocity
        #  - GOAL FLAG
        #  - GOAL POSITION x, y to touch

        ############# ACTION SPACE
        # - position x, y of tool
        # - orientation (left [0], right [1])

        obs_dim = 0

        obs_space = spaces.Box(
            low=np.float32(-np.sqrt(2)),
            high=np.float32(np.sqrt(2)),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        act_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = [obs_space for i in range(self.n_pursuers)]
        self.action_space = [act_space for i in range(self.n_pursuers)]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_obj(self):
        """Create all moving object instances."""
        self.pursuers = []
        self.evaders = []
        self.poisons = []
        self.obstacles = []

        for i in range(self.n_pursuers):
            x, y = self._generate_coord(self.base_radius)
            self.pursuers.append(
                Pursuers(
                    x,
                    y,
                    self.pursuer_max_accel,
                    self.pursuer_speed,
                    radius=self.base_radius,
                    collision_type=i + 1,
                    n_sensors=self.n_sensors,
                    sensor_range=self.sensor_range,
                    speed_features=self.speed_features,
                )
            )

        for i in range(self.n_evaders):
            x, y = self._generate_coord(2 * self.base_radius)
            vx, vy = (
                (2 * self.np_random.random(1) - 1) * self.evader_speed,
                (2 * self.np_random.random(1) - 1) * self.evader_speed,
            )
            self.evaders.append(
                Evaders(
                    x,
                    y,
                    vx,
                    vy,
                    radius=2 * self.base_radius,
                    collision_type=i + 1000,
                    max_speed=self.evader_speed,
                )
            )

        for i in range(self.n_poisons):
            x, y = self._generate_coord(0.75 * self.base_radius)
            vx, vy = (
                (2 * self.np_random.random(1) - 1) * self.poison_speed,
                (2 * self.np_random.random(1) - 1) * self.poison_speed,
            )
            self.poisons.append(
                Poisons(
                    x,
                    y,
                    vx,
                    vy,
                    radius=0.75 * self.base_radius,
                    collision_type=i + 2000,
                    max_speed=self.poison_speed,
                )
            )

        for _ in range(self.n_obstacles):
            self.obstacles.append(
                Obstacle(
                    self.pixel_scale / 2,
                    self.pixel_scale / 2,
                    radius=self.obstacle_radius,
                )
            )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def convert_coordinates(self, value, option="position"):
        """This function converts coordinates in pymunk into pygame coordinates.

        The coordinate system in pygame is:
                 (0, 0) +-------+ (WINDOWSIZE, 0)           + ──── → x
                        |       |                           │
                        |       |                           │
        (0, WINDOWSIZE) +-------+ (WINDOWSIZE, WINDOWSIZE)  ↓ y
        The coordinate system in pymunk is:
        (0, WINDOWSIZE) +-------+ (WINDOWSIZE, WINDOWSIZE)  ↑ y
                        |       |                           │
                        |       |                           │
                 (0, 0) +-------+ (WINDOWSIZE, 0)           + ──── → x
        """
        if option == "position":
            return int(value[0]), self.pixel_scale - int(value[1])

        if option == "velocity":
            return value[0], -value[1]

    def _generate_coord(self, radius):
        """Generates a random coordinate for an object with given radius such that it does not collide with an obstacle.

        radius: radius of the object
        """
        # Sample random coordinate (x, y) with x, y ∈ [0, pixel_scale]
        coord = self.np_random.random(2) * self.pixel_scale

        # If too close to obstacles, resample
        for obstacle in self.obstacles:
            x, y = obstacle.body.position
            while (
                ssd.cdist(coord[None, :], np.array([[x, y]]))
                <= radius * 2 + obstacle.radius
            ):
                coord = self.np_random.random(2) * self.pixel_scale

        return coord

    def add(self):
        """Add all moving objects to PyMunk space."""
        self.space = pymunk.Space()

        for obj_list in [self.pursuers, self.evaders, self.poisons, self.obstacles]:
            for obj in obj_list:
                obj.add(self.space)

    def add_bounding_box(self):
        """Create bounding boxes around the window so that the moving object will not escape the view window.

        The four bounding boxes are aligned in the following way:
        (-100, WINDOWSIZE + 100) ┌────┬────────────────────────────┬────┐ (WINDOWSIZE + 100, WINDOWSIZE + 100)
                                 │xxxx│////////////////////////////│xxxx│
                                 ├────┼────────────────────────────┼────┤
                                 │////│    (WINDOWSIZE, WINDOWSIZE)│////│
                                 │////│                            │////│
                                 │////│(0, 0)                      │////│
                                 ├────┼────────────────────────────┼────┤
                                 │xxxx│////////////////////////////│xxxx│
                    (-100, -100) └────┴────────────────────────────┴────┘ (WINDOWSIZE + 100, -100)
        where "x" represents overlapped regions.
        """
        # Bounding dox edges
        pts = [
            (-100, -100),
            (self.pixel_scale + 100, -100),
            (self.pixel_scale + 100, self.pixel_scale + 100),
            (-100, self.pixel_scale + 100),
        ]

        self.barriers = []

        for i in range(4):
            self.barriers.append(
                pymunk.Segment(self.space.static_body, pts[i], pts[(i + 1) % 4], 100)
            )
            self.barriers[-1].elasticity = 0.999
            self.space.add(self.barriers[-1])

    def add_handlers(self):
        # Collision handlers for pursuers v.s. evaders & poisons
        for pursuer in self.pursuers:
            for obj in self.evaders:
                self.handlers.append(
                    self.space.add_collision_handler(
                        pursuer.shape.collision_type, obj.shape.collision_type
                    )
                )
                self.handlers[-1].begin = self.pursuer_evader_begin_callback
                self.handlers[-1].separate = self.pursuer_evader_separate_callback

            for obj in self.poisons:
                self.handlers.append(
                    self.space.add_collision_handler(
                        pursuer.shape.collision_type, obj.shape.collision_type
                    )
                )
                self.handlers[-1].begin = self.pursuer_poison_begin_callback

        # Collision handlers for poisons v.s. evaders
        for poison in self.poisons:
            for evader in self.evaders:
                self.handlers.append(
                    self.space.add_collision_handler(
                        poison.shape.collision_type, evader.shape.collision_type
                    )
                )
                self.handlers[-1].begin = self.return_false_begin_callback

        # Collision handlers for evaders v.s. evaders
        for i in range(self.n_evaders):
            for j in range(i, self.n_evaders):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.evaders[i].shape.collision_type,
                            self.evaders[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

        # Collision handlers for poisons v.s. poisons
        for i in range(self.n_poisons):
            for j in range(i, self.n_poisons):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.poisons[i].shape.collision_type,
                            self.poisons[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

        # Collision handlers for pursuers v.s. pursuers
        for i in range(self.n_pursuers):
            for j in range(i, self.n_pursuers):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.pursuers[i].shape.collision_type,
                            self.pursuers[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

    def reset(self):
        self.add_obj()
        self.frames = 0

        # Initialize obstacles positions
        if self.initial_obstacle_coord is None:
            for i, obstacle in enumerate(self.obstacles):
                obstacle_position = (
                    self.np_random.random((self.n_obstacles, 2)) * self.pixel_scale
                )
                obstacle.body.position = (
                    obstacle_position[0, 0],
                    obstacle_position[0, 1],
                )
        else:
            for i, obstacle in enumerate(self.obstacles):
                obstacle.body.position = (
                    self.initial_obstacle_coord[i][0] * self.pixel_scale,
                    self.initial_obstacle_coord[i][1] * self.pixel_scale,
                )

        # Add objects to space
        self.add()
        self.add_handlers()
        self.add_bounding_box()

        # Get observation
        obs_list = self.observe_list()

        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.behavior_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = obs_list

        return obs_list[0]

    def step(self, action, agent_id, is_last):
        action = np.asarray(action) * self.pursuer_max_accel
        action = action.reshape(2)
        thrust = np.linalg.norm(action)
        if thrust > self.pursuer_max_accel:
            # Limit added thrust to self.pursuer_max_accel
            action = action * (self.pursuer_max_accel / thrust)

        p = self.pursuers[agent_id]

        # Clip pursuer speed
        _velocity = np.clip(
            p.body.velocity + action * self.pixel_scale,
            -self.pursuer_speed,
            self.pursuer_speed,
        )

        # Set pursuer speed
        p.reset_velocity(_velocity[0], _velocity[1])

        # Penalize large thrusts
        accel_penalty = self.thrust_penalty * math.sqrt((action**2).sum())

        # Average thrust penalty among all agents, and assign each agent global portion designated by (1 - local_ratio)
        self.control_rewards = (
            (accel_penalty / self.n_pursuers)
            * np.ones(self.n_pursuers)
            * (1 - self.local_ratio)
        )

        # Assign the current agent the local portion designated by local_ratio
        self.control_rewards[agent_id] += accel_penalty * self.local_ratio

        if is_last:
            self.space.step(1 / self.FPS)

            obs_list = self.observe_list()
            self.last_obs = obs_list

            for id in range(self.n_pursuers):
                p = self.pursuers[id]

                # reward for food caught, encountered and poison
                self.behavior_rewards[id] = (
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

        return self.observe(agent_id)

    def observe(self, agent_id):
        return np.array(self.last_obs[agent_id], dtype=np.float32)

    def observe_list(self):
        observe_list = []

        for i, pursuer in enumerate(self.pursuers):
            obstacle_distances = []

            evader_distances = []
            evader_velocities = []

            poison_distances = []
            poison_velocities = []

            _pursuer_distances = []
            _pursuer_velocities = []

            for obstacle in self.obstacles:
                obstacle_distance, _ = pursuer.get_sensor_reading(
                    obstacle.body.position, obstacle.radius, obstacle.body.velocity, 0.0
                )
                obstacle_distances.append(obstacle_distance)

            obstacle_sensor_vals = self.get_sensor_readings(
                obstacle_distances, pursuer.sensor_range
            )

            barrier_distances = pursuer.get_sensor_barrier_readings()

            for evader in self.evaders:
                evader_distance, evader_velocity = pursuer.get_sensor_reading(
                    evader.body.position,
                    evader.radius,
                    evader.body.velocity,
                    self.evader_speed,
                )
                evader_distances.append(evader_distance)
                evader_velocities.append(evader_velocity)

            (
                evader_sensor_distance_vals,
                evader_sensor_velocity_vals,
            ) = self.get_sensor_readings(
                evader_distances,
                pursuer.sensor_range,
                velocites=evader_velocities,
            )

            for poison in self.poisons:
                poison_distance, poison_velocity = pursuer.get_sensor_reading(
                    poison.body.position,
                    poison.radius,
                    poison.body.velocity,
                    self.poison_speed,
                )
                poison_distances.append(poison_distance)
                poison_velocities.append(poison_velocity)

            (
                poison_sensor_distance_vals,
                poison_sensor_velocity_vals,
            ) = self.get_sensor_readings(
                poison_distances,
                pursuer.sensor_range,
                velocites=poison_velocities,
            )

            # When there is only one pursuer the sensors will not sense
            # another pursuer
            if self.n_pursuers > 1:
                for j, _pursuer in enumerate(self.pursuers):
                    # Get sensor readings only for other pursuers
                    if i == j:
                        continue

                    _pursuer_distance, _pursuer_velocity = pursuer.get_sensor_reading(
                        _pursuer.body.position,
                        _pursuer.radius,
                        _pursuer.body.velocity,
                        self.pursuer_speed,
                    )
                    _pursuer_distances.append(_pursuer_distance)
                    _pursuer_velocities.append(_pursuer_velocity)

                (
                    _pursuer_sensor_distance_vals,
                    _pursuer_sensor_velocity_vals,
                ) = self.get_sensor_readings(
                    _pursuer_distances,
                    pursuer.sensor_range,
                    velocites=_pursuer_velocities,
                )
            else:
                _pursuer_sensor_distance_vals = np.zeros(self.n_sensors)
                _pursuer_sensor_velocity_vals = np.zeros(self.n_sensors)

            if pursuer.shape.food_touched_indicator >= 1:
                food_obs = 1
            else:
                food_obs = 0

            if pursuer.shape.poison_indicator >= 1:
                poison_obs = 1
            else:
                poison_obs = 0

            # concatenate all observations
            if self.speed_features:
                pursuer_observation = np.concatenate(
                    [
                        obstacle_sensor_vals,
                        barrier_distances,
                        evader_sensor_distance_vals,
                        evader_sensor_velocity_vals,
                        poison_sensor_distance_vals,
                        poison_sensor_velocity_vals,
                        _pursuer_sensor_distance_vals,
                        _pursuer_sensor_velocity_vals,
                        np.array([food_obs]),
                        np.array([poison_obs]),
                    ]
                )
            else:
                pursuer_observation = np.concatenate(
                    [
                        obstacle_sensor_vals,
                        barrier_distances,
                        evader_sensor_distance_vals,
                        poison_sensor_distance_vals,
                        _pursuer_sensor_distance_vals,
                        np.array([food_obs]),
                        np.array([poison_obs]),
                    ]
                )

            observe_list.append(pursuer_observation)

        return observe_list

    def pursuer_poison_begin_callback(self, arbiter, space, data):
        """Called when a collision between a pursuer and a poison occurs.

        The poison indicator of the pursuer becomes 1, the pursuer gets
        a penalty for this step.
        """
        pursuer_shape, poison_shape = arbiter.shapes

        # For giving reward to pursuer
        pursuer_shape.poison_indicator += 1

        # Reset poision position & velocity
        x, y = self._generate_coord(poison_shape.radius)
        vx, vy = self._generate_speed(poison_shape.max_speed)

        poison_shape.reset_position(x, y)
        poison_shape.reset_velocity(vx, vy)

        return False

    def pursuer_evader_begin_callback(self, arbiter, space, data):
        """Called when a collision between a pursuer and an evader occurs.

        The counter of the evader increases by 1, if the counter reaches
        n_coop, then, the pursuer catches the evader and gets a reward.
        """
        pursuer_shape, evader_shape = arbiter.shapes

        # Add one collision to evader
        evader_shape.counter += 1

        # Indicate that food is touched by pursuer
        pursuer_shape.food_touched_indicator += 1

        if evader_shape.counter >= self.n_coop:
            # For giving reward to pursuer
            pursuer_shape.food_indicator = 1

        return False

    def pursuer_evader_separate_callback(self, arbiter, space, data):
        """Called when a collision between a pursuer and a poison ends.

        If at this moment there are greater or equal than n_coop pursuers
        that collides with this evader, the evader's position gets reset
        and the pursuers involved will be rewarded.
        """
        pursuer_shape, evader_shape = arbiter.shapes

        if evader_shape.counter < self.n_coop:
            # Remove one collision from evader
            evader_shape.counter -= 1
        else:
            evader_shape.counter = 0

            # For giving reward to pursuer
            pursuer_shape.food_indicator = 1

            # Reset evader position & velocity
            x, y = self._generate_coord(evader_shape.radius)
            vx, vy = self._generate_speed(evader_shape.max_speed)

            evader_shape.reset_position(x, y)
            evader_shape.reset_velocity(vx, vy)

        pursuer_shape.food_touched_indicator -= 1

    def return_false_begin_callback(self, arbiter, space, data):
        """Callback function that simply returns False."""
        return False

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale, self.pixel_scale)
                )
                pygame.display.set_caption("Waterworld")
            else:
                self.screen = pygame.Surface((self.pixel_scale, self.pixel_scale))

        self.screen.fill((255, 255, 255))
        self.draw()
        self.clock.tick(self.FPS)

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
