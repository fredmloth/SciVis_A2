import numpy as np

# update all neighbours (at the same time, vectorize)
def neighbour_update():

    return

# initialize system
def initialise_system(n, inside, t_inner, t_top=0, t_tank=0):
    # start grid
    grid = np.zeros((n, n))

    # set inner temp
    start = (n - inside) // 2
    end = start + inside
    grid[start:end, start:end] = t_inner

    # set top temp
    grid[0, 0:] = t_top

    # set tank temp
    # bottom
    grid[n-1, 0:] = t_tank

    return grid


grid = initialise_system(4, 2, 2, 1, 3)
print(grid)