import numpy as np

class Grid():
    def __init__(
            self,
            center=3,
            h_tank=2,
            t_inner=3,
            t_top=2,
            t_tank=1):
        self.center=center
        self.n=center*3
        self.h_tank=h_tank
        self.t_inner=t_inner
        self.t_top=t_top
        self.t_tank=t_tank

        self.grid = self._init_grid()

    def _init_grid(self):
        g = np.zeros((self.n, self.n))

        # set inner temp
        start = (self.n - self.center) // 2
        end = start + self.center
        g[start:end, start:end] = self.t_inner

        # set top temp
        g[0, :] = self.t_top

        # set tank temp
        bottom = self.n - 1
        g[bottom, 0:] = self.t_tank
        g[bottom-self.h_tank + 1:, 0] = self.t_tank
        g[bottom-self.h_tank + 1:, self.n-1] = self.t_tank

        return g

g = Grid()
print(g.grid)



# update all neighbours (at the same time, vectorize)
def neighbour_update():

    return

# initialize system
def initialise_system(n=6, center=2, h_tank=1, t_inner=3, t_top=2, t_tank=1):
    # start grid
    grid = np.zeros((n, n))

    # set inner temp
    start = (n - center) // 2
    end = start + center
    grid[start:end, start:end] = t_inner

    # set top temp
    grid[0, 0:] = t_top

    # set tank temp
    bottom = n -1
    grid[bottom, 0:] = t_tank
    grid[bottom-h_tank:, 0] = t_tank
    grid[bottom-h_tank:, n-1] = t_tank

    return grid

def reset_temps(grid):

    return grid


#grid = initialise_system()
#print(grid)