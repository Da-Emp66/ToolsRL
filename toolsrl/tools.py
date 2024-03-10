from typing import Dict, Any, Optional, List, Tuple, Literal

import itertools
import yaml
import pymunk
from pymunk.constraints import PivotJoint


EXPECTED_TOOL_PROPERTIES = {
    "color": {"red":0,"green":0,"blue":0},
    "grip-point": (0, 0),
    "scale": 1.0,
    "sections": [],
}
EXPECTED_TOOL_SECTION_PROPERTIES = {
    "color": {"red":0,"green":0,"blue":0},
    "edges": [],
    "friction": None,
    "layer": 1,
    "scale": 1.0,
    "weight": 10,
}

class ToolSection:
    def create(self) -> List[pymunk.Segment]:
        self.shapes = [pymunk.Segment(self.body, edge[0], edge[1], self.line_width) for edge in self.edges]
        for idx, _ in enumerate(self.shapes):
            self.shapes[idx].custom_value = 1
            self.shapes[idx].collision_type = 1
            self.shapes[idx].begin_dynamic_counter = 0
            self.shapes[idx].separate_dynamic_counter = 0
            self.shapes[idx].begin_static_counter = 0
            self.shapes[idx].separate_static_counter = 0

        for segment in self.shapes:
            segment.color = (self.color['red'], self.color['green'], self.color['blue'], 255)

        if self.friction is not None:
            for idx, _ in enumerate(self.shapes):
                self.shapes[idx].friction = self.friction

        self.space.add(*self.shapes)
        return self

    def load_from_dict(self, description: Dict[str, Any]):
        if len(description.keys()) > 1:
            raise ValueError("Tool section has more than one name.")
        self.name = next(iter(description))

        properties = EXPECTED_TOOL_SECTION_PROPERTIES.copy()
        properties.update(description[self.name])

        for property in properties:
            self.__setattr__(property.replace("-", "_"), properties[property])

        self.points = list(map(lambda point: (self.coordinate_conversion_fn((point[next(iter(point))]['x'] * self.scale, point[next(iter(point))]['y'] * self.scale))), self.points))
        self._update_edges()
        
    def __init__(self, space: pymunk.Space, body: pymunk.Body, description: Dict[str, Any], coordinate_conversion_fn: Optional[Any] = lambda x: x):
        self.space = space
        self.body = body
        self.coordinate_conversion_fn = coordinate_conversion_fn
        self.line_width = 7.0 # TODO: Make this configurable
        self.load_from_dict(description)
    
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
    
    def __add__(self, other: Tuple[int, int]):
        self.points = [(int(point[0] + other[0]), int(point[1] + other[1])) for point in self.points]
        self._update_edges()
        return self

    def __radd__(self, other: Tuple[int, int]):
        self.points = [(int(point[0] + other[0]), int(point[1] + other[1])) for point in self.points]
        self._update_edges()
        return self

class Tool:
    def create(self):
        if not self.created:
            self.space.add(self.body)
            self.created = True
        self.sections.sort()
        self.shapes = [section.create() for section in self.sections[::-1]]
        return self
    
    def destroy(self):
        self.space.remove(*list(itertools.chain(*[shapes.shapes for shapes in self.shapes])))
        return self

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

    def __init__(self, space: pymunk.Space, description: Dict[str, Any], initial_point: Tuple[int, int] = (0, 0), coordinate_conversion_fn: Optional[Any] = lambda x: x):
        self.space = space
        self.body = pymunk.Body(1, 10000)
        self.coordinate_conversion_fn = coordinate_conversion_fn
        self.created = False
        self.load_from_dict(description)
        self.create()

        self.orientation = 0
        self.grip = pymunk.Body(1, 10000, pymunk.Body.STATIC)
        self.rotation_joint = PivotJoint(self.grip, self.body, initial_point, (self.grip_point['x'] * self.scale, self.grip_point['y'] * self.scale))
        self.space.add(self.grip, self.rotation_joint)

    def reset_position(self, x, y):
        self.grip.position = x, y

    def reset_velocity(self, vx, vy):
        self.body.velocity = vx, vy

    def flip_orientation(self, axis: Literal['x', 'y']):
        self.orientation ^= 1
        self.destroy()
        axis = {'x': 0, 'y': 1}[axis]
        for idx, _ in enumerate(self.sections):
            for idx2, _ in enumerate(self.sections[idx].points):
                if axis == 0:
                    self.sections[idx].points[idx2] = (-1 * self.sections[idx].points[idx2][0], self.sections[idx].points[idx2][1])
                elif axis == 1:
                    self.sections[idx].points[idx2] = (self.sections[idx].points[idx2][0], -1 * self.sections[idx].points[idx2][1])
                self.sections[idx]._update_edges()
        # TODO: Remove the self.grip body and self.rotation_joint, multiply self.grip by -1, and recreate both
        self.create()

    def reset(self, position: Optional[Tuple[int, int]] = None):
        self.destroy()
        if position is not None:
            self.reset_position(*position)
        self.create()


class HandyMan:
    def __init__(self, tool: Optional[Tool] = None):
        self.observation_space = None
        self.action_space = None
        self.current_tool = tool

    def set_tool(self, tool: Tool):
        self.current_tool = tool

    def move(self, x, y):
        self.current_tool.reset_position(x, y)
        
    def accelerate(self, vx, vy):
        self.current_tool.reset_velocity(vx, vy)

    def remove_tool(self):
        self.current_tool.destroy()
        self.current_tool = None

    def has_tool(self) -> bool:
        return self.current_tool is not None