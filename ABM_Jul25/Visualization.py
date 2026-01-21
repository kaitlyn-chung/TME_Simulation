# Visualization.py


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import random

from agents import (
    CancerCell, CancerSubtype,
    CD8TCell,
    CD4TCell, CD4TSubtype,
    MDSC,
    Macrophage, MacSubtype
)
from model import ABM_Model
import params as P


# Visualization functions

def plot_cell_counts(model, steps):
    """
    Plot counts of each agent subtype over time using the model's DataCollector.
    """
    df = model.datacollector.get_model_vars_dataframe()
    plt.figure(figsize=(12, 8))

    for col in df.columns:
        if col.lower() != "total cells":
            plt.plot(df.index, df[col], label=col)

    plt.xlabel("Step")
    plt.ylabel("Cell Count")
    plt.title("Cell Population Dynamics")
    plt.legend()
    plt.grid(True)
    return plt


def plot_grid(model):
    """
    Build a 2D array of integer codes representing each agent (by subtype) on the grid,
    then display it with a ListedColormap and colorbar.
    Codes:
      0 = empty
      1 = CancerCell (Stem)
      2 = CancerCell (Progenitor)
      3 = CancerCell (Senescent)
      4 = CD8 T cell
      5 = CD4 Thelper
      6 = CD4 Treg
      7 = MDSC
      8 = Macrophage (M1)
      9 = Macrophage (M2)
    """
    width, height = model.width, model.height
    grid_map = np.zeros((width, height), dtype=int)

    for x in range(width):
        for y in range(height):
            occupant = model.grid[x][y]
            if occupant is None:
                grid_map[x, y] = 0
            elif isinstance(occupant, CancerCell):
                if occupant.subtype == CancerSubtype.STEM:
                    grid_map[x, y] = 1
                elif occupant.subtype == CancerSubtype.PROGENITOR:
                    grid_map[x, y] = 2
                else:  # SENESCENT
                    grid_map[x, y] = 3

            elif isinstance(occupant, CD8TCell):
                grid_map[x, y] = 4

            elif isinstance(occupant, CD4TCell):
                if occupant.subtype == CD4TSubtype.CD4THELPER:
                    grid_map[x, y] = 5
                else:  # CD4TREG
                    grid_map[x, y] = 6

            elif isinstance(occupant, MDSC):
                grid_map[x, y] = 7

            elif isinstance(occupant, Macrophage):
                if occupant.subtype == MacSubtype.M1:
                    grid_map[x, y] = 8
                else:  # M2
                    grid_map[x, y] = 9

            else:
                grid_map[x, y] = 0

    # Define colormap and normalization
    cmap = mcolors.ListedColormap([
        "white",       # 0 empty
        "lightgreen",  # 1 Stem
        "orange",      # 2 Progenitor
        "red",         # 3 Senescent
        "blue",        # 4 CD8
        "purple",      # 5 CD4 Thelper
        "pink",        # 6 CD4 Treg
        "gray",        # 7 MDSC
        "cyan",        # 8 M1
        "magenta"      # 9 M2
    ])
    bounds = np.arange(-0.5, 10.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_map.T, cmap=cmap, norm=norm, origin="lower")
    cbar = plt.colorbar(ticks=list(range(10)))
    cbar.set_ticklabels([
        "Empty",
        "CancerStem",
        "CancerProg",
        "CancerSenescent",
        "CD8",
        "CD4_Helper",
        "CD4_Treg",
        "MDSC",
        "M1",
        "M2"
    ])
    plt.title("Model Grid State")
    return plt


def plot_signal_grid(model, signal_name):
    """
    Plot the 2D concentration field for a given signal. Assumes model has attributes
    like model.TGFb_field, model.CCL2_field, etc.
    """
    if not hasattr(model, f"{signal_name}_field"):
        print(f"Signal '{signal_name}' not found in model.")
        return None

    signal_grid = getattr(model, f"{signal_name}_field")
    plt.figure(figsize=(8, 8))
    plt.imshow(signal_grid.T, origin="lower", cmap="viridis")
    plt.colorbar(label=f"{signal_name} concentration")
    plt.title(f"{signal_name} Distribution")
    return plt


def run_simulation(
    steps=100,
    width=30,
    height=30,
    initial_tumor_cells=100,
    initial_CD8Tcells=20,
    initial_CD4Tcells=30,
    initial_macrophages=30,
    initial_MDSC=10
):
    """
    Instantiate ABM_Model with initial seeding of various agent types, run for a number of steps,
    and then return the model instance.
    """
    model = ABM_Model(width=width, 
                      height=height, 
                      initial_tumor_cells=initial_tumor_cells,
                      initial_CD8Tcells=initial_CD8Tcells,
                      initial_CD4Tcells=initial_CD4Tcells,
                      initial_macrophages=initial_macrophages,
                      initial_MDSC=initial_MDSC)

    # Run the model
    for i in range(steps):
        model.step()
        if not model.running:
            print(f"Simulation stopped at step {i}: no cancer cells remaining")
            break

    return model


# Main execution & plotting
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    steps = 100
    width = 30
    height = 30

    model = run_simulation(
        steps=100,
        width=30,
        height=30,
        initial_tumor_cells=100,
        initial_CD8Tcells=20,
        initial_CD4Tcells=30,
        initial_macrophages=30,
        initial_MDSC=10
    )

    sns.set_style("whitegrid")

    # Plot 1: Cell counts over time
    plot_cell_counts(model, steps)
    plt.savefig("cell_counts.png")
    plt.close()

    # Plot 2: Final grid state
    plot_grid(model)
    plt.savefig("final_grid.png")
    plt.close()

    # Plot 3: Signal distributions (all fields in model)
    signal_names = [
        "Arg1", "CCL2", "IFNg", "IL2", "IL10", "IL12", "NO", "TGFb", "VEGFA"
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    for idx, sig in enumerate(signal_names):
        ax = axes[idx]
        if hasattr(model, f"{sig}_field"):
            grid = getattr(model, f"{sig}_field")
            im = ax.imshow(grid.T, origin="lower", cmap="viridis")
            ax.set_title(f"{sig} Distribution")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, f"{sig} not found", ha="center")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("signal_distributions.png")
    plt.close()

    # Print final statistics
    print("\nFinal Cell Counts:")
    print(f"Stem Cancer: {model.count_cell_type(CancerCell, subtype=CancerSubtype.STEM)}")
    print(f"Prog Cancer: {model.count_cell_type(CancerCell, subtype=CancerSubtype.PROGENITOR)}")
    print(f"Senescent Cancer: {model.count_cell_type(CancerCell, subtype=CancerSubtype.SENESCENT)}")
    print(f"CD8 T Cells: {model.count_cell_type(CD8TCell)}")
    print(f"CD4 Thelpers: {model.count_cell_type(CD4TCell, subtype=CD4TSubtype.CD4THELPER)}")
    print(f"CD4 Tregs: {model.count_cell_type(CD4TCell, subtype=CD4TSubtype.CD4TREG)}")
    print(f"MDSCs: {model.count_cell_type(MDSC)}")
    print(f"M1 Macs: {model.count_cell_type(Macrophage, subtype=MacSubtype.M1)}")
    print(f"M2 Macs: {model.count_cell_type(Macrophage, subtype=MacSubtype.M2)}")

    plt.show()