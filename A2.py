# This module runs the code for the second assignment of 
# Scientific Visualisation and Virtual Reality
# Written by Frederieke Loth, 12016926

import numpy as np
import os
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
            t_tank=32,
            mode=1, # 1 = constant, 2 = uniform random
            t_value=32.0,
            t_low=32.0,
            t_high=100.0):
        
        self.scale = scale
        self.center = scale * self._CENTER
        self.n = scale * self._BIG
        self.h_tank = scale * self._TANK
        self.t_inner = t_inner
        self.t_top = t_top
        self.t_tank = t_tank

        self.grid = self._init_grid()
        self.mask = self.mask()
        self._init_free_cells(mode, t_value, t_low, t_high)

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
    
    def _init_free_cells(self, mode, t_value, t_low, t_high):
        """Fill non-masked cells based on a simple mode: 1=constant, 
        2=uniform random."""
        free = ~self.mask
        if mode == 1:
            self.grid[free] = float(t_value)
        elif mode == 2:
            self.grid[free] = np.random.uniform(float(t_low), float(t_high), size=free.sum())
        else:
            raise ValueError("mode must be 1 (constant) or 2 (uniform random).")
    
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


# Plotting code
def plot_heatmap(g, steps=(0, 25, 50, "final"), fname="snapshots.png"):
    """
    Save a 2x2 figure with snapshots from g.history.
    steps: tuple of indices; use "final" for the last grid.
    """
    # Resolve requested steps -> history indices + labels
    idxs, labels = [], []
    last = len(g.history) - 1
    for s in steps:
        if s == "final":
            idxs.append(last)
            labels.append("final")
        else:
            i = min(int(s), last)
            idxs.append(i)
            labels.append(f"t = {s}" if i == s else f"t = {i} (clamped)")

    # Consistent color scale
    vmin = min(g.t_tank, g.t_top, g.t_inner)
    vmax = max(g.t_tank, g.t_top, g.t_inner)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    axes = axes.ravel()

    im = None
    for ax, i, lab in zip(axes, idxs, labels):
        im = ax.imshow(g.history[i], origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(lab)
        ax.set_xticks([]); ax.set_yticks([])

    # One shared colorbar for all subplots
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, fraction=0.05, pad=0.02)
    cbar.set_label("Temperature")

    fig.suptitle(f"Heat diffusion snapshots (iters={g.iters}, final Δ={g.last_max_t:.3f})")
    fig.savefig(os.path.join("figures", fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_three_inits_panel(t_values=[0, 32.0], t_low=32.0, t_high=100.0, steps=(0, 10, 25, "final")):
    """
    Build 3 grids (mode 1, 1, 2), solve each and plot a 3x4 panel of given timesteps.
    """
    # build and solve the 3 grids
    g1 = Grid(scale=1, mode=1, t_value=t_values[0])
    g1.solve(tolerance=0.5)

    g2 = Grid(scale=1, mode=1, t_value=t_values[1])
    g2.solve(tolerance=0.5)

    g3 = Grid(scale=1, mode=2, t_low=t_low, t_high=t_high)
    g3.solve(tolerance=0.5)

    grids = [
        ("(1)", g1),
        ("(2)", g2),
        ("(3)", g3),
    ]

    # Choose frames and labels
    def resolve_idxs(history, steps=steps):
        last = len(history) - 1
        idxs, labs = [], []
        for s in steps:
            if s == "final":
                idxs.append(last); labs.append("final")
            
            # If converge early
            else:
                i = min(int(s), last)
                idxs.append(i)
                labs.append(f"t = {s}" if i == s else f"t = {i} (clamped)")

        return idxs, labs

    # consistent color scale across subplots with boundary temps
    vmin = min(g1.t_tank, g1.t_top, g1.t_inner)
    vmax = max(g1.t_tank, g1.t_top, g1.t_inner)

    # figure layout: 3 rows x 4 cols
    fig, axes = plt.subplots(
        nrows=3, ncols=4, figsize=(16, 10),
        constrained_layout=True
    )

    all_images = []
    row_labels = []

    for r, (row_label, g) in enumerate(grids):
        idxs, labs = resolve_idxs(g.history)
        for c, (i, lab) in enumerate(zip(idxs, labs)):
            ax = axes[r, c]
            im = ax.imshow(g.history[i], origin="upper", vmin=vmin, vmax=vmax)
            all_images.append(im)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(lab, fontsize=12)  # readable titles
        row_labels.append(row_label)

    # Row labels along the left
    for r, text in enumerate(row_labels):
        # place text label just outside the first column
        axes[r, 0].text(
            -0.15, 0.5, text, transform=axes[r, 0].transAxes,
            va="center", ha="right", fontsize=14, fontweight="bold"
        )

    cbar = fig.colorbar(all_images[-1], ax=axes.ravel().tolist(), location="right", shrink=0.9, pad=0.02)
    cbar.set_label("Temperature", fontsize=12)

    fig.suptitle("Heat diffusion across 3 initializations.", fontsize=16)
    fig.savefig(os.path.join("figures", "three_inits_panel.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved three_inits_panel.png")
    print(f"(1) mode=1, t_value=0 > iters: {g1.iters}, final Δ={g1.last_max_t:.3f}")
    print(f"(2) mode=1, t_value=32 -> iters: {g2.iters}, final Δ={g2.last_max_t:.3f}")
    print(f"(3) mode=2, 32–100 uniform -> iters: {g3.iters}, final Δ={g3.last_max_t:.3f}")


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


def export_to_csv(g, out_dir="out", prefix="grid"):
    """Exports the information from history to a csv file per grid"""
    os.makedirs(out_dir, exist_ok=True)
    n = g.n

    # xx is row index, yy is col index
    ii, jj = np.indices((n,n))

    for t, arr in enumerate(g.history):
        table = np.column_stack((ii.ravel(), jj.ravel(), arr.ravel()))
        fname = os.path.join(out_dir, f"{prefix}_{t:04d}.csv")
        np.savetxt(fname, table, delimiter=",", fmt=["%d", "%d", "%.6f"],
                   header="i, j, T", comments="")
        
    print(f"Wrote {len(g.history)} files to {out_dir} with prefix '{prefix}_####.csv'")


if __name__ == "__main__":
    # plot three differen t initial conditions
    plot_three_inits_panel(t_values=[0, 32.0], t_low=32.0, t_high=100.0, steps=(0, 10, 25, "final"))

    # animate and export data from one grid
    g = Grid(scale=1)
    g.solve(tolerance=0.5)
    export_to_csv(g, out_dir="out", prefix="grid")
    plot_heatmap(g, steps=(0, 25, 50, "final"), fname="snapshots.png")
    animate_from_history(g, interval=10)
