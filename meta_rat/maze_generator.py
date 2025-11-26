import os
import random
import numpy as np

MAP_DIR = "maps"
os.makedirs(MAP_DIR, exist_ok=True)

# Symbols:
# S = Start, G = Goal, # = Wall, . = Empty, B = Pushable, C = Climbable, T = Trap
SYMBOLS = {
    "empty": ".",
    "wall": "#",
    "start": "S",
    "goal": "G",
    "push": "B",
    "climb": "C",
    "trap": "T"
}

def generate_maze(rows=20, cols=20, density=0.25, seed=None):
    """Generate a random maze with start, goal, obstacles, and traps."""
    if seed:
        random.seed(seed)
    
    # Initialize empty maze
    maze = np.full((rows, cols), SYMBOLS["empty"], dtype=str)
    
    # Add outer walls
    maze[0, :] = SYMBOLS["wall"]
    maze[-1, :] = SYMBOLS["wall"]
    maze[:, 0] = SYMBOLS["wall"]
    maze[:, -1] = SYMBOLS["wall"]

    # Place start and goal far apart
    start = (1, 1)
    goal = (rows - 2, cols - 2)
    maze[start] = SYMBOLS["start"]
    maze[goal] = SYMBOLS["goal"]

    # Randomly add internal walls (ensure not blocking path fully)
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if (r, c) not in [start, goal] and random.random() < density:
                maze[r, c] = SYMBOLS["wall"]

    # Ensure rough path connectivity using DFS-based carving
    carve_path(maze, start, goal)

    # Add pushable, climbable, traps
    add_special_blocks(maze)

    return maze

def carve_path(maze, start, goal):
    """Carve a guaranteed path between start and goal using DFS."""
    stack = [start]
    visited = set()
    while stack:
        r, c = stack.pop()
        visited.add((r, c))
        if (r, c) == goal:
            return

        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        random.shuffle(neighbors)
        for nr, nc in neighbors:
            if 1 <= nr < maze.shape[0] - 1 and 1 <= nc < maze.shape[1] - 1:
                if (nr, nc) not in visited and maze[nr, nc] != SYMBOLS["goal"]:
                    maze[nr, nc] = SYMBOLS["empty"]
                    stack.append((nr, nc))

def add_special_blocks(maze):
    """Randomly scatter pushables (B), climbables (C), and traps (T)."""
    rows, cols = maze.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if maze[r, c] == SYMBOLS["empty"]:
                p = random.random()
                if p < 0.03:
                    maze[r, c] = SYMBOLS["push"]
                elif p < 0.05:
                    maze[r, c] = SYMBOLS["climb"]
                elif p < 0.07:
                    maze[r, c] = SYMBOLS["trap"]

def save_maze(maze, filename):
    """Save maze to file."""
    with open(os.path.join(MAP_DIR, filename), "w") as f:
        for row in maze:
            f.write("".join(row) + "\n")

def generate_multiple_mazes(count=30):
    """Generate multiple random mazes and save them."""
    for i in range(count):
        maze = generate_maze(rows=20, cols=20, density=random.uniform(0.15, 0.3), seed=i)
        filename = f"meta_maze_{i+1}.txt"
        save_maze(maze, filename)
        print(f"✅ Generated {filename}")

if __name__ == "__main__":
    generate_multiple_mazes()
