import pygame
import sys
from maze_loader import load_maze

# --- Colors ---
COLORS = {
    0: (230, 230, 230),  # empty
    1: (50, 50, 50),     # wall
    2: (170, 110, 40),   # pushable block (B)
    3: (100, 180, 250),  # climbable block (C)
    4: (0, 200, 0),      # start
    5: (255, 215, 0),    # goal
    6: (200, 0, 0),      # trap
}

CELL_SIZE = 30
FPS = 60

class MazeGame:
    def __init__(self, map_path):
        self.grid, self.start, self.goal = load_maze(map_path)
        self.rows, self.cols = self.grid.shape
        self.rat_pos = list(self.start)  # [row, col]
        self.holding_push = False
        self.energy = 100  # optional stamina mechanic
        pygame.init()
        self.screen = pygame.display.set_mode((self.cols * CELL_SIZE, self.rows * CELL_SIZE))
        pygame.display.set_caption("🐀 MetaRat — Push & Climb Maze")
        self.clock = pygame.time.Clock()

    def draw(self):
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r, c]
                color = COLORS[val]
                pygame.draw.rect(
                    self.screen,
                    color,
                    (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
                )

        # Draw the rat
        rx, ry = self.rat_pos[1] * CELL_SIZE, self.rat_pos[0] * CELL_SIZE
        pygame.draw.circle(self.screen, (0, 0, 255), (rx + CELL_SIZE//2, ry + CELL_SIZE//2), CELL_SIZE//3)

        # Draw energy bar
        pygame.draw.rect(self.screen, (0, 255, 0), (10, 10, self.energy * 2, 10))

    def move(self, dr, dc):
        r, c = self.rat_pos
        nr, nc = r + dr, c + dc

        # Bounds check
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return
        
        cell = self.grid[nr, nc]

        # Walls block movement
        if cell == 1:
            return

        # Pushable block (B)
        if cell == 2:
            if self.holding_push:  # Only push if 'B' key held
                next_r, next_c = nr + dr, nc + dc
                if (
                    0 <= next_r < self.rows and 0 <= next_c < self.cols and
                    self.grid[next_r, next_c] == 0
                ):
                    # push block
                    self.grid[next_r, next_c] = 2
                    self.grid[nr, nc] = 0
                    self.rat_pos = [nr, nc]
            else:
                # can't push without holding B
                return

        # Climbable block (C)
        elif cell == 3:
            # Can climb only if this block was pushed
            # simulate "using energy"
            if self.energy > 0:
                self.energy -= 5
                self.rat_pos = [nr, nc]
            else:
                print("Rat is too tired to climb.")
                return

        # Empty floor, start, etc.
        elif cell in (0, 4):
            self.rat_pos = [nr, nc]

        # Goal
        elif cell == 5:
            print("🎉 The rat found the cheese! You win!")
            pygame.quit()
            sys.exit()

        # Trap
        elif cell == 6:
            print("💀 The rat fell into a trap! Game over!")
            pygame.quit()
            sys.exit()

    def run(self):
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.move(-1, 0)
                    elif event.key == pygame.K_DOWN:
                        self.move(1, 0)
                    elif event.key == pygame.K_LEFT:
                        self.move(0, -1)
                    elif event.key == pygame.K_RIGHT:
                        self.move(0, 1)
                    elif event.key == pygame.K_b:
                        self.holding_push = True  # start pushing

                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_b:
                        self.holding_push = False  # stop pushing

            self.screen.fill((0, 0, 0))
            self.draw()
            pygame.display.flip()
