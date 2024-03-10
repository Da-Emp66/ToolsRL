import itertools
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import yaml
import pymunk
from pymunk.constraints import PinJoint

EXPECTED_GOAL_PROPERTIES = {
    "accomplishment-criteria": None,
    "environment-objects": [],
    "hinges": None,
    "scale": 1.0,
}

EXPECTED_ENVIRONMENT_OBJECT_PROPERTIES = {
    "color": {"red":0,"green":0,"blue":0},
    "friction": None,
    "moment": 0,
    "points": [],
    "radius": 10.0,
    "scale": 1.0,
    "shape": "polygon",
    "type": "static",
    "weight": 10,
}

class EnvironmentObject:
    def create(self):
        body_type = None
        if self.type == "static":
            body_type = pymunk.Body.STATIC
        elif self.type == "dynamic":
            body_type = pymunk.Body.DYNAMIC

        self.body = pymunk.Body(self.weight, self.moment, body_type)
        self.space.add(self.body)

        if self.shape == "polygon":
            self.shapes = [pymunk.Segment(self.body, edge[0], edge[1], self.line_width) for edge in self.edges]
        elif self.shape == "circle":
            self.shapes = [pymunk.Circle(self.body, self.radius)]

        for idx, _ in enumerate(self.shapes):
            self.shapes[idx].custom_value = 1
            self.shapes[idx].collision_type = 2
            
        if self.friction is not None:
            for idx, _ in enumerate(self.shapes):
                self.shapes[idx].friction = self.friction

        self.space.add(*self.shapes)

        return self

    def load_from_dict(self, description: Dict[str, Any]):
        if len(description.keys()) > 1:
            raise ValueError("EnvironmentObject has more than one name.")
        self.name = next(iter(description))

        properties = EXPECTED_ENVIRONMENT_OBJECT_PROPERTIES.copy()
        properties.update(description[self.name])

        for property in properties:
            self.__setattr__(property.replace("-", "_"), properties[property])
        
        self.points = list(map(lambda point: (self.coordinate_conversion_fn((point[next(iter(point))]['x'] * self.scale, point[next(iter(point))]['y'] * self.scale))), self.points))
        self._update_edges()

    def __init__(self, space: pymunk.Space, description: Dict[str, Any], initial_point: Tuple[int, int] = (0, 0), coordinate_conversion_fn: Optional[Any] = lambda x: x):
        self.space = space
        self.initial_point = initial_point
        self.coordinate_conversion_fn = coordinate_conversion_fn
        self.line_width = 7.0 # TODO: Make this configurable
        self.load_from_dict(description)

    def _update_edges(self):
        num_edges = len(self.points)
        self.edges = [[self.points[idx]] + [self.points[(idx + 1) % num_edges]] for  idx, _ in enumerate(self.points)]

    def _scale(self, factor: int | float) -> 'EnvironmentObject':
        self.points = [(int(point[0] * factor), int(point[1] * factor)) for point in self.points]
        self._update_edges()
        return self
    
    def __mul__(self, other: int | float) -> 'EnvironmentObject':
        return self._scale(other)

    def __rmul__(self, other: int | float) -> 'EnvironmentObject':
        return self._scale(other)
    
    def __add__(self, other: Tuple[int, int]):
        self.points = [(int(point[0] + other[0]), int(point[1] + other[1])) for point in self.points]
        self._update_edges()
        return self

    def __radd__(self, other: Tuple[int, int]):
        self.points = [(int(point[0] + other[0]), int(point[1] + other[1])) for point in self.points]
        self._update_edges()
        return self


class Goal:
    def create(self):
        self.shapes = [environment_object.create() for environment_object in self.environment_objects]
        
        for environment_object in self.environment_objects:
            if environment_object.type == "dynamic":
                environment_object.body.position = self.initial_point
                environment_object.body.velocity = (0, 0)

        self.hinge_shapes = []
        if self.hinges is not None:
            all_hinges, hinge_objects, hinge_points = [], [], []
            for hinge in self.hinges:
                hinge_name = next(iter(hinge))
                hinge_details = hinge[hinge_name]
                hinge_objects = [environment_object for environment_object in hinge_details]
                hinge_object_names = [next(iter(environment_object)) for environment_object in hinge_objects]
                hinge_points = [hinge_object[hinge_object_name]['point'] for hinge_object, hinge_object_name in zip(hinge_objects, hinge_object_names)]
                hinge_object_types = []
                for hinge_object_name in hinge_object_names:
                    for environment_object in self.environment_objects:
                        if environment_object.name == hinge_object_name:
                            hinge_object_types.append(environment_object.type)
                assert len(hinge_object_names) == len(hinge_points) == len(hinge_object_types) == 2, "Each hinge can only have 2 objects and 2 points (1 per object). No more, no less <(*_,*)> Both objects must also already exist in the goal and have either `static` or `dynamic` type."
                all_hinges.append((hinge_object_names, hinge_object_types, hinge_points))

            def get_body(self, environment_object_name: str):
                for environment_object in self.environment_objects:
                    if environment_object.name == environment_object_name:
                        return environment_object.body
                return None
            
            self.hinge_shapes = [
                PinJoint(
                    *[get_body(self, hinge_object_name) for hinge_object_name in hinge_object_names],
                    *list(map(lambda hinge_object_type, point: (self.initial_point[0] - point['x'], self.initial_point[1] - point['y']) if hinge_object_type == "static" else (-point['x'], -point['y']), hinge_object_types, hinge_points))
                ) for hinge_object_names, hinge_object_types, hinge_points in all_hinges
            ]

            self.space.add(*self.hinge_shapes)

        self.goal_touch_body = pymunk.Body(100000, 1000000, body_type=pymunk.Body.STATIC)
        self.goal_touch_shape = pymunk.Circle(self.goal_touch_body, 5)
        self.goal_touch_shape.goal_accomplishment_counter = 0
        self.goal_touch_shape.color = (255, 43, 203, 255)
        self.goal_touch_shape.custom_value = 1
        self.goal_touch_shape.collision_type = 3
        self.goal_touch_body.position = self.initial_point[0] - self.accomplishment_criteria['touches']['x'], self.initial_point[1] - self.accomplishment_criteria['touches']['y']
        self.space.add(self.goal_touch_body, self.goal_touch_shape)
        
        return self

    def load_from_dict(self, description: Dict[str, Any]):
        if len(description.keys()) > 1:
            raise ValueError("Goal has more than one name.")
        self.name = next(iter(description))

        properties = EXPECTED_GOAL_PROPERTIES.copy()
        properties.update(description[self.name])

        for property in properties:
            self.__setattr__(property.replace("-", "_"), properties[property])
        
        self.environment_objects = [EnvironmentObject(self.space, environment_object, self.initial_point) * -self.scale for environment_object in self.environment_objects]
        self.environment_objects = [environment_object + self.initial_point if environment_object.type == "static" else environment_object for environment_object in self.environment_objects]
        
    def __init__(self, space: pymunk.Space, description: Dict[str, Any], initial_point: Tuple[int, int] = (0, 0), coordinate_conversion_fn: Optional[Any] = lambda x: x):
        self.space = space
        self.initial_point = initial_point
        self.coordinate_conversion_fn = coordinate_conversion_fn
        self.hinge_point_on_accomplishment_object = None
        self.load_from_dict(description)
        self.create()

    def _scale(self, factor: int | float) -> 'Goal':
        self.shapes = [shape * factor for shape in self.shapes]
        return self
    
    def __mul__(self, other: int | float) -> 'Goal':
        return self._scale(other)

    def __rmul__(self, other: int | float) -> 'Goal':
        return self._scale(other)

    def goal_accomplished(self) -> bool:
        index = None
        for idx, environment_object in enumerate(self.environment_objects):
            if environment_object.name == self.accomplishment_criteria['object']:
                index = idx

        if self.environment_objects[index]._is_touching((self.accomplishment_criteria['touches']['x'], self.accomplishment_criteria['touches']['y'])):
            return True
        
        return False
    
    def _get_hinge_point_on_accomplishment_object(self):
            if self.hinge_point_on_accomplishment_object is not None:
                return self.hinge_point_on_accomplishment_object
            if self.hinges is not None:
                hinge_objects = list(itertools.chain(*[hinge[next(iter(hinge))] for hinge in self.hinges]))
                hinge_on_accomplishment_object_idx = [next(iter(hinge_object)) for hinge_object in hinge_objects].index(self.accomplishment_criteria['object'])
                if hinge_on_accomplishment_object_idx == -1:
                    return None
                else:
                    hinge_on_accomplishment_object_point_definitions = hinge_objects[hinge_on_accomplishment_object_idx][next(iter(hinge_objects[hinge_on_accomplishment_object_idx]))]
                    self.hinge_point_on_accomplishment_object = (hinge_on_accomplishment_object_point_definitions['point']['x'], hinge_on_accomplishment_object_point_definitions['point']['y'])
                    return self.hinge_point_on_accomplishment_object
            else:
                return None
            
    def get_support_vector(self):
        index = None
        for idx, environment_object in enumerate(self.environment_objects):
            if environment_object.name == self.accomplishment_criteria['object']:
                index = idx
        
        # TODO: Dynamically adjust this vector
        # vector_start_loc = self.environment_objects[index].body.position
        centers_of_all_edges = list(map(lambda edge: ((edge[0][0] + edge[1][0]) * -0.5, (edge[0][1] + edge[1][1]) * -0.5), self.environment_objects[index].edges))
        x_accumulator = 0
        y_accumulator = 0
        for center in centers_of_all_edges:
            x_accumulator += center[0]
            y_accumulator += center[1]
        num_edges_in_accomplishment_object = len(self.environment_objects[index].edges)
        if num_edges_in_accomplishment_object == 0:
            x_accumulator, y_accumulator = (15, 15)
            num_edges_in_accomplishment_object = 1
        vector_start_loc = tuple([x_accumulator / num_edges_in_accomplishment_object, y_accumulator / num_edges_in_accomplishment_object])
        vector_destination = (self.accomplishment_criteria['touches']['x'], self.accomplishment_criteria['touches']['y'])
        vector_aim = [vector_destination[i] - vector_start_loc[i] for i in range(2)]
        length = np.linalg.norm(vector_aim)
        unit_vector_aim = [direction / length for direction in vector_aim]

        # TODO: During dynamic adjustment of the vector, account for hinges
        # hinge_point_on_accomplishment_object = self._get_hinge_point_on_accomplishment_object()
        # if self.hinges is not None and hinge_point_on_accomplishment_object is not None:
        #     hinge_radius = np.linalg.norm([vector_start_loc[i] - self.hinge_point_on_accomplishment_object[i] for i in range(2)])
        #     slope = ((self.hinge_point_on_accomplishment_object[1] - vector_start_loc[1]) /
        #                 (self.hinge_point_on_accomplishment_object[0] - vector_start_loc[0]))
        #     tangent_slope = -1 / slope
        #     # Find the tangent and account for obstacles
        #     tangent_vector = [tangent_slope, 1]
        #     unit_vector_aim = tangent_vector / np.linalg.norm(tangent_vector)

        return unit_vector_aim

    def destroy(self):
        self.space.remove(*list(itertools.chain(*[shapes.shapes for shapes in self.shapes])))
        self.space.remove(*self.hinge_shapes)
        self.space.remove(self.goal_touch_body)
        return self
    