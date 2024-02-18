from typing import List

import argparse
import sys
import yaml
import pygame
import pymunk
import pymunk.pygame_util
from toolsrl.tools import Tool
from toolsrl.goals import Goal

FPS = 50.0
WINDOW_SIZE_X = 1200
WINDOW_SIZE_Y = 900
BARRIER_WIDTH = 1000

def main(args):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
    pygame.display.set_caption("Tools Sandbox")
    clock = pygame.time.Clock()
    space = pymunk.Space()
    space.gravity = (0.0, 980.0)
    initial_tool_pos = (600, 450)
    initial_goal_pos = (2*WINDOW_SIZE_X/3, WINDOW_SIZE_Y) # In PyGame Coordinates

    draw_options = pymunk.pygame_util.DrawOptions(screen)

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

    tool = None
    with open(args.configuration) as toolfile:
        description = yaml.safe_load(toolfile)
        tool = Tool(space, {args.tool: description['tools'][args.tool]}, initial_tool_pos, convert_coordinates)
        goal = Goal(space, {args.goal: description['goals'][args.goal]}, initial_goal_pos, convert_coordinates)
        toolfile.close()
    
    def add_bounding_box(space: pymunk.Space):
        pts = [
            (-BARRIER_WIDTH, -BARRIER_WIDTH),
            (WINDOW_SIZE_X + BARRIER_WIDTH, -BARRIER_WIDTH),
            (WINDOW_SIZE_X + BARRIER_WIDTH, WINDOW_SIZE_Y + BARRIER_WIDTH),
            (-BARRIER_WIDTH, WINDOW_SIZE_Y + BARRIER_WIDTH),
        ]
        barriers: List[pymunk.Segment] = []
        for i in range(4):
            barriers.append(
                pymunk.Segment(space.static_body, pts[i], pts[(i + 1) % 4], BARRIER_WIDTH)
            )
            barriers[-1].elasticity = 0.999
            space.add(barriers[-1])

    add_bounding_box(space)

    dragging = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False

        if dragging:
            pos = pygame.mouse.get_pos()
            print(pos)
            tool.reset_position(pos[0] - initial_tool_pos[0], pos[1] - initial_tool_pos[1])

        screen.fill((255,255,255))
    
        space.debug_draw(draw_options)

        space.step(1/FPS)

        pygame.display.flip()
        clock.tick(int(FPS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configuration", default="./configurations/pz_ppo_config.yaml")
    ap.add_argument("--tool", default="hammer")
    ap.add_argument("--goal", default="open-the-chest")
    main(ap.parse_args())
