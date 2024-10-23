import sys
import heapq
from math import sqrt

# import matplotlib.pyplot as plt


def read_world_from_file(filename):
    """
    Reads the world from a file and returns the grid and the start and end positions.

    Args:
    filename (str): The name of the file containing the world.

    Returns:
    grid (list of lists): A 2D list representing the grid.
    start (tuple): The position of the start cell.
    end (tuple): The position of the end cell.
    """
    with open(filename, 'r') as f:
        M = int(f.readline())
        N = int(f.readline())
        grid = []
        start = None
        end = None
        for i in range(M):
            row = f.readline().split(",")
            grid_row = []
            for j, cell in enumerate(row):
                if cell == '-2':
                    start = (i, j)
                    grid_row.append(0)
                elif cell == '-3':
                    end = (i, j)
                    grid_row.append(0)
                elif cell == '-1':
                    grid_row.append(sys.maxsize)
                else:
                    grid_row.append(int(cell))
            grid.append(grid_row)
    return grid, start, end


def a_star(grid, start, end):
    """
    Finds the cheapest path from the start to the end in the grid using Dijkstra's algorithm.

    Args:
    grid (list of lists): A 2D list representing the grid.
    start (tuple): The position of the start cell.
    end (tuple): The position of the end cell.

    Returns:
    int: The cost of the cheapest path. Returns -1 if there is no path.
    """
    m = len(grid)
    n = len(grid[0])
    distances = [[sys.maxsize for _ in range(n)] for _ in range(m)]
    distances[start[0]][start[1]] = 0
    parent = {start: None}
    heap = [(0, start)]
    while heap:
        d, (i, j) = heapq.heappop(heap)
        if (i, j) == end:
            path = []
            ij = (i, j)
            while ij is not None:
                i, j = ij
                path.append((i, j))
                ij = parent[(i, j)]
            path.reverse()
            return d, path

        for i2, j2 in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1), (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]:
            if 0 <= i2 < m and 0 <= j2 < n:
                if grid[i2][j2] == sys.maxsize:
                    continue
                new_d = d + grid[i2][j2]
                if new_d < distances[i2][j2]:
                    parent[(i2, j2)] = (i, j)
                    distances[i2][j2] = new_d
                    heuristic = sqrt((i2 - end[0]) ** 2 + (j2 - end[1]) ** 2)
                    heapq.heappush(heap, (new_d + heuristic, (i2, j2)))

    return -1, []

def path_cost(grid, path):
    """
    Calculates the cost of a path in the grid.

    Args:
    grid (list of lists): A 2D list representing the grid.
    path (list of tuples): A list of tuples representing the path.

    Returns:
    int: The cost of the path.
    """
    cost = 0
    for i, j in path:
        cost += grid[i][j]
    return cost


if __name__ == "__main__":
    world, start_i, end_i = read_world_from_file("world/world_easy.txt")

    print("World:", world)
    print("Start:", start_i)
    print("End:", end_i)
    min_path_cost, min_path  = a_star(world, start_i, end_i)
    print("Minimum path a*:", min_path)
    print("Path:", min_path_cost)
    print("Path cost:", path_cost(world, min_path))

    # plt.imshow(world, cmap='hot', interpolation='nearest')
    # plt.imshow(world, cmap='hot', interpolation='nearest')
    # path_y, path_x = zip(*min_path)
    # plt.scatter(path_x, path_y, c='blue', marker='o')
    # plt.show()
