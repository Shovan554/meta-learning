import numpy as np

SYMBOLS = {
    '.': 0,   # empty
    '#': 1,   # wall
    'B': 2,   # pushable
    'C': 3,   # climbable
    'S': 4,   # start
    'G': 5,   # goal (cheese)
    'T': 6    # trap
}

def load_maze(path):
    with open(path, 'r') as f:
        lines = [list(line.strip()) for line in f.readlines()]
    height, width = len(lines), len(lines[0])
    grid = np.zeros((height, width), dtype=int)
    start = None
    goal = None

    for i in range(height):
        for j in range(width):
            char = lines[i][j]
            grid[i, j] = SYMBOLS.get(char, 0)
            if char == 'S':
                start = (i, j)
            elif char == 'G':
                goal = (i, j)
    return grid, start, goal
