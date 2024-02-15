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
        self.shape = [pymunk.Segment(self.body, edge[0] * self.scale, edge[1] * self.scale, self.line_width) for edge in self.edges]
        self.space.add(*self.shape)
        return self.shape

    def load_from_dict(self, description: Dict[str, Any]):
        if len(description.keys()) > 1:
            raise ValueError("Tool section has more than one name.")
        self.name = next(iter(description))

        properties = EXPECTED_TOOL_SECTION_PROPERTIES.copy()
        properties.update(description[self.name])

        for property in properties:
            self.__setattr__(property.replace("-", "_"), properties[property])

        self.points = list(map(lambda point: (point[next(iter(point))]['x'] * self.scale, point[next(iter(point))]['y'] * self.scale), self.points))
        self._update_edges()
        
    def __init__(self, space: pymunk.Space, body: pymunk.Body, description: Dict[str, Any]):
        self.space = space
        self.body = body
        self.line_width = 3.0
        self.load_from_dict(description)
    
    def draw(self, display: pygame.Surface, convert_coordinates: Any) -> pygame.Rect:
        return pygame.draw.polygon(display, pygame.Color(self.color['red'], self.color['green'], self.color['blue']), list(map(lambda point: convert_coordinates(point), self.points)), width=int(self.line_width))
    
    def _scale(self, factor: int | float) -> 'ToolSection':
        self.points = [(int(point[0] * factor), int(point[1] * factor)) for point in self.points]
        self._update_edges()
        return self
    
    def _update_edges(self):
        num_edges = len(self.points)
        self.edges = [[self.points[idx]] + [self.points[(idx + 1) % num_edges]] for  idx, _ in enumerate(self.points)]

    def __lt__(self, other: 'ToolSection') -> bool:
        return self.layer < other.layer
    
    def __mul__(self, other: int | float) -> 'ToolSection':
        return self._scale(other)

    def __rmul__(self, other: int | float) -> 'ToolSection':
        return self._scale(other)

class Tool:
    def create(self):
        self.space.add(self.body)
        self.shape = [section.create() for section in self.sections]

    def load_from_dict(self, description: Dict[str, Any]):
        if len(description.keys()) > 1:
            raise ValueError("Tool has more than one name.")
        self.name = next(iter(description))

        properties = EXPECTED_TOOL_PROPERTIES.copy()
        properties.update(description[self.name])

        for property in properties:
            self.__setattr__(property.replace("-", "_"), properties[property])

        self.sections = [ToolSection(self.space, self.body, section) * self.scale for section in self.sections]

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

    def __init__(self, space: pymunk.Space, description: Dict[str, Any]):
        self.space = space
        self.body = pymunk.Body(10, 98)
        self.load_from_dict(description)
        self.create()

        self.grip = pymunk.Body(10, 98, pymunk.Body.STATIC)
        self.rotation_joint = PivotJoint(self.grip, self.body, (0, 0), (self.grip_point['x'] * self.scale, self.grip_point['y'] * self.scale))
        self.space.add(self.grip, self.rotation_joint)

    def draw(self, display: pygame.Surface, convert_coordinates: Any) -> List[pygame.Rect]:
        self.sections.sort()
        return [section.draw(display, convert_coordinates) for section in self.sections[::-1]]
    
    def reset_position(self, x, y):
        self.grip.position = x, y

    def reset_velocity(self, vx, vy):
        self.body.velocity = vx, vy

