from typing import List
import pymunk

WINDOW_SIZE_X = 1200
WINDOW_SIZE_Y = 900
BARRIER_WIDTH = 1000

"""Helper functions from WaterWorld"""

def convert_coordinates(value, option="position"):
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
        return int(value[0]), WINDOW_SIZE_Y - int(value[1])

    if option == "velocity":
        return value[0], -value[1]


def add_bounding_box(
        space: pymunk.Space,
        barrier_width: int = BARRIER_WIDTH,
        box_size_x: int = WINDOW_SIZE_X,
        box_size_y: int = WINDOW_SIZE_Y
    ):
    pts = [
        (-barrier_width, -barrier_width),
        (box_size_x + barrier_width, -barrier_width),
        (box_size_x + barrier_width, box_size_y + barrier_width),
        (-barrier_width, box_size_y + barrier_width),
    ]
    barriers: List[pymunk.Segment] = []
    for i in range(4):
        barriers.append(
            pymunk.Segment(space.static_body, pts[i], pts[(i + 1) % 4], barrier_width)
        )
        barriers[-1].elasticity = 0.999
        space.add(barriers[-1])