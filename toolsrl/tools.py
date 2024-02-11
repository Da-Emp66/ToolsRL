from typing import Dict, Any

import pygame
import pymunk
from pymunk.constraints import PivotJoint

class Tool:
    def load_from_dict(self, description: Dict[str, Any]):
        pass

    def __init__(self, description: Dict[str, Any]):
        self.body = pymunk.Body()
        self.load_from_dict(description)
        PivotJoint(self.body, self.shape.body, (0,0), (self.grip_point.x, self.grip_point.y))

    def add(self, space: pymunk.Space):
        space.add(self.body, self.shape)

    def draw(self, display: pygame.Surface, convert_coordinates: Any):
        # for polygon in 
        # pygame.draw.polygon(
        #     display, self.color, convert_coordinates(self.body.position), self.radius
        # )
        pass
    
    def reset_position(self, x, y):
        self.body.position = x, y

    def reset_velocity(self, vx, vy):
        self.body.velocity = vx, vy

