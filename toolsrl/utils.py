from typing import List, Dict, Any
import re
import pymunk

WINDOW_SIZE_X = 1200
WINDOW_SIZE_Y = 900
BARRIER_WIDTH = 1000

def get_initial_positions(description: Dict[str, Any], goal: str):
    offsets = description['offsets'][goal]
    positions = [offsets['initial_tool_position'], offsets['initial_goal_position']]
    for idx, initial_pos in enumerate(positions):
        potential_position = str(initial_pos).replace("WINDOW_SIZE_X", str(WINDOW_SIZE_X)).replace("WINDOW_SIZE_Y", str(WINDOW_SIZE_Y))
        if re.match("^([\d\%\*\/\+\-\(\),]*)*$", potential_position) and 0 < len(potential_position) < 1000:
            positions[idx] = eval(potential_position)
        else:
            raise ValueError("""Configuration file contains potentially dangerous initial positions.
                               Remove all characters other than parentheses, digits, and basic mathematical operators.
                               The type should be a tuple.""")
    initial_tool_pos, initial_goal_pos = positions
    return (initial_tool_pos, initial_goal_pos)

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