from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import yaml
import pygame
import pymunk
from pymunk.constraints import PivotJoint, PinJoint

EXPECTED_GOAL_PROPERTIES = {
    "accomplishment-criteria": None,
    "environment-objects": [],
    "hinges": None,
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

        self.body = pymunk.Body(self.weight, 10000000000, body_type)
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

        if self.hinges is not None:
            all_hinges, hinge_objects, hinge_points = [], [], []
            for hinge in self.hinges:
                hinge_name = next(iter(hinge))
                hinge_details = hinge[hinge_name]
                hinge_objects = [environment_object for environment_object in hinge_details]
                hinge_object_names = [next(iter(environment_object)) for environment_object in hinge_objects]
                hinge_points = [hinge_object[hinge_object_name]['point'] for hinge_object, hinge_object_name in zip(hinge_objects, hinge_object_names)]
                assert len(hinge_object_names) == len(hinge_points) == 2, "Each hinge can only have 2 objects and 2 points (1 per object). No more, no less <(*_,*)>"
                all_hinges.append((hinge_object_names, hinge_points))

            def get_body(self, environment_object_name: str):
                for environment_object in self.environment_objects:
                    if environment_object.name == environment_object_name:
                        return environment_object.body
                return None
            
            # TODO: Variably adjust coordinates if the body is dynamic
            self.hinge_shapes = [
                PinJoint(
                    *[get_body(self, hinge_object_name) for hinge_object_name in hinge_object_names],
                    *list(map(lambda point: (self.initial_point[0] - point['x'], self.initial_point[1] - point['y']), hinge_points))
                ) for hinge_object_names, hinge_points in all_hinges
            ]

            self.space.add(*self.hinge_shapes)
        
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
