from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import yaml
import pygame
import pymunk
from pymunk.constraints import PivotJoint

EXPECTED_GOAL_PROPERTIES = {
    "accomplishment-criteria": None,
    "environment-objects": [],
    "scale": 1.0,
}

EXPECTED_ENVIRONMENT_OBJECT_PROPERTIES = {
    "color": {"red":0,"green":0,"blue":0},
    "friction": None,
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

        self.body = pymunk.Body(self.weight, 98, body_type)
        self.space.add(self.body)

        if self.shape == "polygon":
            self.shapes = [pymunk.Segment(self.body, edge[0], edge[1], self.line_width) for edge in self.edges]
        elif self.shape == "circle":
            self.shapes = [pymunk.Circle(self.body, self.radius)]

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

    def __init__(self, space: pymunk.Space, description: Dict[str, Any], coordinate_conversion_fn: Optional[Any] = lambda x: x):
        self.space = space
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
        return self

    def load_from_dict(self, description: Dict[str, Any]):
        if len(description.keys()) > 1:
            raise ValueError("Goal has more than one name.")
        self.name = next(iter(description))

        properties = EXPECTED_GOAL_PROPERTIES.copy()
        properties.update(description[self.name])

        for property in properties:
            self.__setattr__(property.replace("-", "_"), properties[property])
        
        self.environment_objects = [EnvironmentObject(self.space, environment_object) * -self.scale for environment_object in self.environment_objects]
        self.environment_objects = [env_obj + self.initial_point if env_obj.type == "static" else env_obj for env_obj in self.environment_objects]
        
    def __init__(self, space: pymunk.Space, description: Dict[str, Any], initial_point: Tuple[int, int] = (0, 0), coordinate_conversion_fn: Optional[Any] = lambda x: x):
        self.space = space
        self.initial_point = initial_point
        self.coordinate_conversion_fn = coordinate_conversion_fn
        self.load_from_dict(description)
        self.create()

    def _update_edges(self):
        num_edges = len(self.points)
        self.edges = [[self.points[idx]] + [self.points[(idx + 1) % num_edges]] for  idx, _ in enumerate(self.points)]

    def _scale(self, factor: int | float) -> 'Goal':
        self.shapes = [shape * factor for shape in self.shapes]
        return self
    
    def __mul__(self, other: int | float) -> 'Goal':
        return self._scale(other)

    def __rmul__(self, other: int | float) -> 'Goal':
        return self._scale(other)

    def _goal_accomplished(self) -> bool:
        index = None
        for idx, environment_object in enumerate(self.environment_objects):
            if environment_object.name == self.accomplishment_criteria['object']:
                index = idx

        if self.environment_objects[index].is_touching((self.accomplishment_criteria['touches']['x'], self.accomplishment_criteria['touches']['y'])):
            return True
        
        return False
    
    def get_reward(self) -> float:
        pass
