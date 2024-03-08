from typing import List

import argparse
import sys
import yaml
import pygame
import pymunk
import pymunk.pygame_util
from toolsrl.tools import Tool
from toolsrl.goals import Goal
from toolsrl.utils import (
    get_initial_positions, convert_coordinates, add_bounding_box,
    WINDOW_SIZE_X, WINDOW_SIZE_Y
)

FPS = 100.0

def main(args):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
    pygame.display.set_caption("Tools Sandbox")
    clock = pygame.time.Clock()
    space = pymunk.Space()
    space.gravity = (0.0, 980.0)
    initial_tool_pos = None
    initial_goal_pos = None # In PyGame Coordinates

    draw_options = pymunk.pygame_util.DrawOptions(screen)

    tool = None
    with open(args.configuration) as toolfile:
        description = yaml.safe_load(toolfile)
        initial_tool_pos, initial_goal_pos = get_initial_positions(description, args.goal)
        tool = Tool(space, {args.tool: description['tools'][args.tool]}, initial_tool_pos, convert_coordinates)
        goal = Goal(space, {args.goal: description['goals'][args.goal]}, initial_goal_pos, convert_coordinates)
        toolfile.close()
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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    tool.flip_orientation('x')

        if dragging:
            pos = pygame.mouse.get_pos()
            # print(tool.body.angular_velocity)
            # print(tool.body.angle % (2 * np.pi))
            # print([[shape.a for shape in created_section.shapes] for created_section in tool.shapes])
            print(goal.get_support_vector())
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
