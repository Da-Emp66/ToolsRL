from typing import List

import argparse
import sys
import yaml
import pygame
import pymunk
import pymunk.pygame_util
from toolsrl.tools import Tool

FPS = 50.0
PIXEL_SCALE = 30 * 25

def main(args):
    pygame.init()
    screen = pygame.display.set_mode((1200, 900))
    pygame.display.set_caption("Tools Sandbox")
    clock = pygame.time.Clock()
    space = pymunk.Space()
    space.gravity = (0.0, 980.0)

    draw_options = pymunk.pygame_util.DrawOptions(screen)

    tool = None
    with open(args.configuration) as toolfile:
        tool = Tool(space, {args.tool: yaml.safe_load(toolfile)['tools'][args.tool]})
        toolfile.close()

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
            return int(value[0]), PIXEL_SCALE - int(value[1])

        if option == "velocity":
            return value[0], -value[1]
    
    def add_bounding_box(space: pymunk.Space):
        pts = [
            (-100, -100),
            (PIXEL_SCALE + 100, -100),
            (PIXEL_SCALE + 100, PIXEL_SCALE + 100),
            (-100, PIXEL_SCALE + 100),
        ]
        barriers: List[pymunk.Segment] = []
        for i in range(4):
            barriers.append(
                pymunk.Segment(space.static_body, pts[i], pts[(i + 1) % 4], 100)
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
            tool.reset_position(pos[0], pos[1])

        screen.fill((255,255,255))

        # tool.draw(screen, convert_coordinates)
    
        space.debug_draw(draw_options)

        space.step(1/FPS)

        pygame.display.flip()
        clock.tick(int(FPS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configuration", default="./configurations/pz_ppo_config.yaml")
    ap.add_argument("--tool", default="hammer")
    main(ap.parse_args())