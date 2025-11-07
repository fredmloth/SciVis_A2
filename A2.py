# This module runs the code for the second assignment of 
# Scientific Visualisation and Virtual Reality
# Written by Frederieke Loth, 12016926

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Grid():
    """This class initiates a grid. Proportions are listed according to 
    the given experiment and ensure the grid sclaes appropriately."""
    # Define proportions
    _BIG = 20
    _CENTER = 4
    _TANK = 9

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
        m[-1, :] = True
        m[:, 0] = True
        m[:, -1] = True

        return m
    
    def neighbour_update(self, grid):
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
        
        return np.where(self.mask, grid, new)
    
    def solve(self, tolerance=0.5, max_iters=10000, save_steps=1):
        """Iterate neighbour updates until maximum t change < 0.5 OR the 
        max iteration is reached. Saves the grid for every save_steps."""
        # store values and history for later use
        current = self.grid.copy()
        self.history = [current.copy()]
        self.iters = 0
        self.last_max_t = None

        # iterate over update for tolerance or max iterations
        for i in range(1, max_iters+1):
            next_grid = self.neighbour_update(current)
            self.history.append(next_grid)
            max_t = np.max(np.abs(next_grid - current))
            self.last_max_t = max_t
            self.iters = i

            if max_t < tolerance:
                self.grid = next_grid
                return self.grid
            
            current = next_grid

        # update the grid from history
        self.grid = self.history[-1]

        return self.grid


def animate_from_history(g, interval=10):
    """Animate the grid heatmap from history"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Heat diffusion (iters: {g.iters}, final Δ={g.last_max_t:.3f})")
    ax.set_xticks([]); ax.set_yticks([])

    # Fixed color scale
    vmin = min(g.t_tank, g.t_top, g.t_inner)
    vmax = max(g.t_tank, g.t_top, g.t_inner)

    im = ax.imshow(g.history[0], origin='upper', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Temperature")

    # set the correct updating function from history
    def update(i):
        im.set_data(g.history[i])
        return (im,)

    ani = FuncAnimation(fig, update, frames=len(g.history), interval=interval, blit=True)
    plt.tight_layout()
    plt.show()

    return ani


if __name__ == "__main__":
    g = Grid(scale=1)
    g.solve(tolerance=0.5)
    animate_from_history(g, interval=10)
    print("iters:", g.iters, "final max Δ:", g.last_max_t, "saved:", len(g.history))
