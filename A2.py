import numpy as np

# update all neighbours (at the same time, vectorize)
def neighbour_update():

    return

# initialize system
def initialise_system(n=4, inside=2, h_tank=1, t_inner=3, t_top=2, t_tank=1):
    # start grid
    grid = np.zeros((n, n))

    # set inner temp
    start = (n - inside) // 2
    end = start + inside
    grid[start:end, start:end] = t_inner

    # set top temp
    grid[0, 0:] = t_top

    # set tank temp
    bottom = n -1
    grid[bottom, 0:] = t_tank
    grid[bottom-h_tank:, 0] = t_tank
    grid[bottom-h_tank:, n-1] = t_tank

    return grid


grid = initialise_system()
print(grid)