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

        # set linear increase
        start_increase = self.n - self.h_tank + 1
        increase = np.linspace(self.t_top, self.t_tank, start_increase)
        g[:start_increase, 0] = increase
        g[:start_increase, -1] = increase
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
        m[:, 0] = True
        m[:, -1] = True

        return m
    
    def neighbour_update(self):
        
        # update according to neighboring squares
        # vectorize
        # only update for grid spaces where mask is False
        g = self.grid
        for i in range(0, self.n):
            for j in range(0, self.n):
                g[i,j] = 1/4 * (g[i-1, j] + g[i+1,j] + g[i,j-1] + g[i, j+1])

        return g


g = Grid()
#g.neighbour_update()




#grid = initialise_system()
#print(grid)