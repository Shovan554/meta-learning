# maze_env_rl.py

import numpy as np
import pygame
import sys
import math
from maze_loader import load_maze

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PUSH"]  # 0–3 move, 4 = push/climb
CELL_SIZE = 30  # size of each cell in pixels
FPS = 10000  # high fps so training runs fast


class MazeEnv:
    def __init__(self, map_path, render_mode=True):
        self.map_path = map_path  # path to maze file
        self.grid, self.start, self.goal = load_maze(map_path)  # load maze layout
        self.rows, self.cols = self.grid.shape  # maze dimensions
        self.rat_pos = list(self.start)  # current rat position
        self.render_mode = render_mode  # whether to show pygame window
        self.energy = 100  # unused for now but kept
        self.done = False  # episode done flag
        self.reward = 0  # reward for current step
        self.last_action_dir = (0, 0)  # last movement direction
        self.just_climbed = False  # flag when rat climbs
        self.prev_dist = None  # previous normalized distance to goal

        if render_mode:
            pygame.init()  # init pygame
            self.screen = pygame.display.set_mode(
                (self.cols * CELL_SIZE, self.rows * CELL_SIZE)
            )  # game window
            pygame.display.set_caption("🐀 MetaRat RL Simulation")  # window title
            self.clock = pygame.time.Clock()  # clock for fps

    def get_state(self):
        state = np.copy(self.grid)  # copy grid
        r, c = self.rat_pos  # current rat row/col
        state[r, c] = 9  # mark rat position
        return state.flatten()  # flatten to 1D for RL agent

    def step(self, action):
        self.reward = -0.3  # base step penalty so fewer steps are better
        self.just_climbed = False  # reset climb flag
        dr, dc = 0, 0  # movement delta

        if action == 0:
            dr, dc = -1, 0  # up
        elif action == 1:
            dr, dc = 1, 0  # down
        elif action == 2:
            dr, dc = 0, -1  # left
        elif action == 3:
            dr, dc = 0, 1  # right

        if action in [0, 1, 2, 3]:
            self.last_action_dir = (dr, dc)  # remember last move direction
        if action == 4:
            dr, dc = self.last_action_dir  # push/climb in last move direction

        self.move(dr, dc, pushing=(action == 4))  # apply movement logic

        goal_r, goal_c = self.goal  # goal cell
        r, c = self.rat_pos  # current rat pos
        curr_dist = math.dist((r, c), (goal_r, goal_c))  # euclidean distance
        max_dist = math.dist((0, 0), (self.rows - 1, self.cols - 1))  # max possible
        norm_dist = curr_dist / max_dist  # normalized distance [0, 1]

        if self.prev_dist is not None:
            delta = self.prev_dist - norm_dist  # positive if closer to goal
            self.reward += delta * 2.0  # distance-based shaping
        self.prev_dist = norm_dist  # update prev_dist

        if self.render_mode:
            self.render()  # draw frame

        return self.get_state(), self.reward, self.done, self.just_climbed  # standard gym-style return

    def move(self, dr, dc, pushing=False):
        if dr == 0 and dc == 0:
            return  # no movement

        r, c = self.rat_pos  # current position
        nr, nc = r + dr, c + dc  # next position

        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return  # out of bounds

        cell = self.grid[nr, nc]  # cell type

        if cell == 1:
            self.reward -= 0.2  # hit wall
            return

        if cell == 2:  # pushable block
            if pushing:
                next_r, next_c = nr + dr, nc + dc  # cell after block
                if (
                    0 <= next_r < self.rows
                    and 0 <= next_c < self.cols
                    and self.grid[next_r, next_c] == 0
                ):
                    self.grid[next_r, next_c] = 2  # move block forward
                    self.grid[nr, nc] = 0  # clear old block position
                    self.rat_pos = [nr, nc]  # rat moves into block cell
                    self.reward += 1  # for a sucesscful push
                else:
                    self.reward -= 0.3  # bad push into non-empty or wall
            else:
                self.reward -= 0.1  # bumped block without pushing
            return

        elif cell == 3:  # climbable block
            if pushing:
                self.rat_pos = [nr, nc]  # rat climbs onto block
                self.energy -= 1  # small energy cost
                self.reward += 1  # small positive reward for climbing
                self.just_climbed = True  # flag climb
            else:
                self.reward -= 0.05  # touched climbable but didn't climb
            return

        elif cell in (0, 4):  # empty floor or start
            self.rat_pos = [nr, nc]  # regular move

        elif cell == 5:  # goal (cheese)
            self.rat_pos = [nr, nc]  # move onto goal
            self.reward += 10  # big reward for reaching goal
            self.done = True  # end episode

        elif cell == 6:  # trap
            self.rat_pos = [nr, nc]  # move onto trap
            self.reward -= 10  # big penalty
            self.done = True  # end episode

    def reset(self):
        self.grid, self.start, self.goal = load_maze(self.map_path)  # reload maze
        self.rat_pos = list(self.start)  # reset rat pos
        self.energy = 100  # reset energy
        self.done = False  # reset done flag
        self.reward = 0  # reset reward
        self.last_action_dir = (0, 0)  # reset last dir
        self.just_climbed = False  # reset climb flag
        self.prev_dist = None  # reset distance history
        return self.get_state()  # initial state

    def render(self):
        COLORS = {
            0: (230, 230, 230),  # empty
            1: (50, 50, 50),  # wall
            2: (170, 110, 40),  # pushable
            3: (100, 180, 250),  # climbable
            4: (0, 200, 0),  # start
            5: (255, 215, 0),  # cheese
            6: (200, 0, 0),  # trap
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()  # quit pygame
                sys.exit()  # exit program

        self.clock.tick(FPS)  # control speed
        self.screen.fill((0, 0, 0))  # black background

        for r in range(self.rows):
            for c in range(self.cols):
                color = COLORS[self.grid[r, c]]  # pick color for cell
                pygame.draw.rect(
                    self.screen,
                    color,
                    (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1),
                )  # draw cell

        rx, ry = self.rat_pos[1] * CELL_SIZE, self.rat_pos[0] * CELL_SIZE  # rat pixel pos
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),
            (rx + CELL_SIZE // 2, ry + CELL_SIZE // 2),
            CELL_SIZE // 3,
        )  # draw rat

        pygame.display.flip()  # update screen

    def close(self):
        if self.render_mode:
            pygame.quit()  # close pygame
