import numpy as np

class Grid():
    """This class initiates a grid. Proportions are listed according to 
    the given experiment and ensure the grid sclaes appropriately."""
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
        """Creates a boolean mask of grid spaces that should remain the 
        same temperature for True."""
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
    
    def neighbour_update(grid, self):
        """Does not directly update the grid, set 
        self.grid = self.neighbour_update(self.grid) in code"""
        g = grid.copy()

        # pad grid to enable vectorized updating (faster)
        padded = np.pad(g, 1, mode='edge')

        new = 0.25 * (
            padded[:-2, 1:-1] +
            padded[2:, 1:-1] +
            padded[1:-1, :-2] +
            padded[1:-1, 2:])
        
        grid = np.where(self.mask, grid, new)
        return grid


g = Grid()
#g.neighbour_update()




#grid = initialise_system()
#print(grid)