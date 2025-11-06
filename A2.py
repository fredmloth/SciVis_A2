import numpy as np

class Grid():
    # Define proportions
    _BIG = 9
    _CENTER = 3
    _TANK = 4

    def __init__(
            self,
            scale=1,
            t_inner=212,
            t_top=100,
            t_tank=32):
        
        self.scale = scale
        self.center = scale * self._CENTER
        self.n = scale * self._BIG
        self.h_tank = scale * self._TANK
        self.t_inner = t_inner
        self.t_top = t_top
        self.t_tank = t_tank

        self.grid = self._init_grid()
        self.mask = self.mask()

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
    
    def mask(self):
        m = np.zeros((self.n, self.n), dtype=bool)

        # center
        start = (self.n - self.center) // 2
        end = start + self.center
        m[start:end, start:end] = True

        m[0, :] = True

        bottom = self.n - 1
        m[bottom, :] = True
        m[bottom - self.h_tank + 1:, 0] = True
        m[bottom - self.h_tank + 1:, -1] = True

        return m


g = Grid()
print(g.grid)
print(g.mask)



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