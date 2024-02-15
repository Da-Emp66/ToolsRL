from typing import Dict, Any, Optional, List

import yaml
import pygame
import pymunk
from pymunk.constraints import PivotJoint

EXPECTED_TOOL_PROPERTIES = {
    "color": (0, 0, 0),
    "grip-point": (0, 0),
    "scale": 1.0,
    "sections": [],
}
EXPECTED_TOOL_SECTION_PROPERTIES = {
    "color": (0, 0, 0),
    "edges": [],
    "friction": 0.5,
    "layer": 1,
    "scale": 1.0,
    "weight": 10,
}

class ToolSection:
    def create(self) -> List[pymunk.Segment]:
        self.shape = [pymunk.Segment(self.body, edge[0], edge[1], self.line_width) for edge in self.edges]
        return self.shape

    def load_from_dict(self, description: Dict[str, Any]):
        if len(description.keys()) > 1:
            raise ValueError("Tool section has more than one name.")
        self.name = next(iter(description))

        properties = EXPECTED_TOOL_SECTION_PROPERTIES.copy()
        properties.update(description[self.name])

        for property in properties:
            self.__setattr__(property.replace("-", "_"), properties[property])

        self.points = list(map(lambda point: (point[next(iter(point))]['x'], point[next(iter(point))]['y']), self.points))
        num_edges = len(self.points)
        self.edges = [[self.points[idx]] + [self.points[(idx + 1) % num_edges]] for  idx, _ in enumerate(self.points)]
        
    def __init__(self, body: pymunk.Body, description: Dict[str, Any]):
        self.body = body
        self.line_width = 3.0
        self.load_from_dict(description)
        self.create()
    
    def draw(self, display: pygame.Surface, convert_coordinates: Any):
        pygame.draw.polygon(display, self.color, list(map(lambda point: convert_coordinates(point), self.points)), width=int(self.line_width))

class Tool:
    def create(self):
        self.shape = [section.create() for section in self.sections]

    def load_from_dict(self, description: Dict[str, Any]):
        if len(description.keys()) > 1:
            raise ValueError("Tool has more than one name.")
        self.name = next(iter(description))

        properties = EXPECTED_TOOL_PROPERTIES.copy()
        properties.update(description[self.name])

        for property in properties:
            self.__setattr__(property.replace("-", "_"), properties[property])

        self.sections = [ToolSection(self.body, section) for section in self.sections]

    def load_from_config(self, configuration_file: str, name: Optional[str] = None):
        with open(configuration_file) as config:
            description = yaml.safe_load(config)
            config.close()

        if description.get("tools", False):
            description = description['tools']

        if len(description.keys()) > 1:
            if description.get(name, False):
                description = description[name]
            else:
                raise ValueError("`name` is None and description contains more than one tool. \
                                 Ambiguous tool cannot be loaded from configuration file.")
        
        self.load_from_dict({name: description})

    def __init__(self, description: Dict[str, Any]):
        self.body = pymunk.Body()
        self.load_from_dict(description)
        self.create()

        self.rotation_joint = PivotJoint(pymunk.Body(), self.body, (0, 0), (self.grip_point['x'], self.grip_point['y']))

    def add(self, space: pymunk.Space):
        space.add(self.body, self.shape)

    def draw(self, display: pygame.Surface, convert_coordinates: Any):
        for section in self.sections:
            section.draw(display, convert_coordinates)
    
    def reset_position(self, x, y):
        self.body.position = x, y

    def reset_velocity(self, vx, vy):
        self.body.velocity = vx, vy

