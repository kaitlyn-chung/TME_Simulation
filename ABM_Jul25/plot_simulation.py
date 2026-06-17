#!/usr/bin/env python3
"""
Enhanced ABM runner with detailed step-by-step output and monitoring.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import os
import pandas as pd

from ABM_Jul25.agents import (
    CancerCell, CancerSubtype,
    CD8TCell, CD8DiffState, CD8ExhaustionState,
    CD4TCell, CD4DiffState,
    MDSC,
    Macrophage, MacSubtype
)
from ABM_Jul25.model import ABM_Model
from ABM_Jul25.scenarios import DEFAULT_SCENARIOS

### Utility functions for printing summaries and saving data

def print_step_summary(counts, step_num, step_time=None):
    """Print detailed summary of current model state."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}")
    if step_time:
        print(f"Step execution time: {step_time:.3f}s")
    print(f"{'='*60}")
    
    # Count all cell types and subtypes
    cancer_stem = counts['Cancer_Stem']
    cancer_prog = counts['Cancer_Prog']
    cancer_senes = counts['Cancer_Senes']
    total_cancer = cancer_stem + cancer_prog + cancer_senes
    
    cd8_naive = counts['CD8_Naive']
    cd8_activated = counts['CD8_Activated']
    cd8_memory = counts['CD8_Memory']
    cd8_count = cd8_naive + cd8_activated + cd8_memory

    cd8_exhausted = counts['CD8_Exhausted']
    cd8_effector = counts['CD8_Effector']
    cd8_notexhausted = counts['CD8_NotExhausted']
    ratio_exhausted = cd8_exhausted / cd8_count if cd8_count > 0 else 0

    cd4_naive = counts['CD4_Naive']
    cd4_activated = counts['CD4_Activated']
    cd4_helper = counts['CD4_Helper']
    cd4_treg = counts['CD4_Treg']
    total_cd4 = cd4_naive + cd4_activated + cd4_helper + cd4_treg
    
    m1_count = counts['M1']
    m2_count = counts['M2']
    total_mac = m1_count + m2_count
    
    mdsc_count = counts['MDSC']
    
    total_immune = cd8_count + total_cd4 + total_mac + mdsc_count
    total_cells = total_cancer + total_immune
    
    print(f"CELL POPULATIONS:")
    print(f"  Cancer Cells (Total: {total_cancer:3d})")
    print(f"    ├─ Stem:       {cancer_stem:3d}")
    print(f"    ├─ Progenitor: {cancer_prog:3d}")
    print(f"    └─ Senescent:  {cancer_senes:3d}")
    print(f"")
    print(f"  Immune Cells (Total: {total_immune:3d})")
    print(f"    ├─ CD8+ T:         {cd8_count:3d}")
    print(f"    │   ├─ Naive:      {cd8_naive:3d}")
    print(f"    │   ├─ Activated:  {cd8_activated:3d}")
    print(f"    │   ├─ Memory:     {cd8_memory:3d}")
    print(f"    │   └─ Exhaustion States:")
    print(f"    │      ├─ Not Exhausted: {cd8_notexhausted:3d}")
    print(f"    │      ├─ Effector:   {cd8_effector:3d}")
    print(f"    │      └─ Exhausted:  {cd8_exhausted:3d}")
    print(f"    ├─ CD4+ T:         {total_cd4:3d}")
    print(f"    │   ├─ Naive:      {cd4_naive:3d}")
    print(f"    │   ├─ Activated:  {cd4_activated:3d}")
    print(f"    │   ├─ Helper:     {cd4_helper:3d}")
    print(f"    │   ├─ Regulatory: {cd4_treg:3d}")
    print(f"    ├─ Macrophage:     {total_mac:3d}")
    print(f"    │   ├─ M1:         {m1_count:3d}")
    print(f"    │   └─ M2:         {m2_count:3d}")
    print(f"    └─ MDSC:           {mdsc_count:3d}")
    print(f"")
    print(f"  TOTAL CELLS: {total_cells:3d}")
    
    # Calculate ratios
    if total_cancer > 0:
        immune_cancer_ratio = total_immune / total_cancer
        cd8_cancer_ratio = cd8_count / total_cancer
    else:
        immune_cancer_ratio = float('inf') if total_immune > 0 else 0
        cd8_cancer_ratio = float('inf') if cd8_count > 0 else 0
    
    if total_mac > 0:
        m1_m2_ratio = m1_count / m2_count if m2_count > 0 else float('inf')
    else:
        m1_m2_ratio = 0
    
    if total_cd4 > 0:
        helper_treg_ratio = cd4_helper / cd4_treg if cd4_treg > 0 else float('inf')
    else:
        helper_treg_ratio = 0
    
    print(f"\nKEY RATIOS:")
    print(f"  Immune:Cancer           = {immune_cancer_ratio:.2f}")
    print(f"  CD8:Cancer              = {cd8_cancer_ratio:.2f}")
    print(f"  Exhausted CD8:Total CD8 = {ratio_exhausted:.2f}")
    print(f"  M1:M2                   = {m1_m2_ratio:.2f}")
    print(f"  Helper:Treg             = {helper_treg_ratio:.2f}")

def print_molecular_summary(model):
    """Print summary of molecular concentrations."""
    print(f"\nMOLECULAR CONCENTRATIONS (mean ± std):")
    
    molecules = ['Arg1', 'CCL2', 'IFNg', 'IL2', 'IL10', 'IL12', 'NO', 'TGFb', 'VEGFA']
    
    for mol in molecules:
        field = getattr(model, f"{mol}_field")
        mean_conc = np.mean(field)
        std_conc = np.std(field)
        max_conc = np.max(field)
        print(f"  {mol:5s}: {mean_conc:.2e} ± {std_conc:.2e} (max: {max_conc:.2e})")

def get_cell_counts(model):
    counts = {}

    counts['Cancer_Stem'] = model.count_cell_type(CancerCell, subtype=CancerSubtype.STEM)
    counts['Cancer_Prog'] = model.count_cell_type(CancerCell, subtype=CancerSubtype.PROGENITOR)
    counts['Cancer_Senes'] = model.count_cell_type(CancerCell, subtype=CancerSubtype.SENESCENT)

    counts['CD8_Naive'] = model.count_cell_type(CD8TCell, diff_state=CD8DiffState.CD8TNAIVE)
    counts['CD8_Activated'] = model.count_cell_type(CD8TCell, diff_state=CD8DiffState.CD8TACTIVATED)
    counts['CD8_Memory'] = model.count_cell_type(CD8TCell, diff_state=CD8DiffState.CD8TMEMORY)
    counts['CD8_Exhausted'] = model.count_cell_type(CD8TCell,
        exhaustion_state=CD8ExhaustionState.CD8TTERMINAL)
    counts['CD8_Effector'] = model.count_cell_type(
        CD8TCell, exhaustion_state=CD8ExhaustionState.CD8TEFFECTOR
    )
    counts['CD8_NotExhausted'] = model.count_cell_type(
        CD8TCell, exhaustion_state=CD8ExhaustionState.CD8TNOTEXHAUSTED
    )


    counts['CD4_Naive'] = model.count_cell_type(CD4TCell, diff_state=CD4DiffState.CD4NAIVE)
    counts['CD4_Activated'] = model.count_cell_type(CD4TCell, diff_state=CD4DiffState.CD4ACTIVATED)
    counts['CD4_Helper'] = model.count_cell_type(CD4TCell, diff_state=CD4DiffState.CD4THELPER)
    counts['CD4_Treg'] = model.count_cell_type(CD4TCell, diff_state=CD4DiffState.CD4TREG)

    counts['M1'] = model.count_cell_type(Macrophage, subtype=MacSubtype.M1)
    counts['M2'] = model.count_cell_type(Macrophage, subtype=MacSubtype.M2)
    counts['MDSC'] = model.count_cell_type(MDSC)

    return counts

def save_step_data(counts, step_num, output_dir='simulation_output'):
    """Save step data to files."""
    
    # Save to CSV for easy plotting later
    csv_file = os.path.join(output_dir, "cell_counts.csv")
    if step_num == 0:
        with open(csv_file, 'w') as f:
            f.write("Step,Cancer_Stem,Cancer_Prog,Cancer_Senes,"
                    "CD8_Naive,CD8_Activated,CD8_Memory,CD8_Exhausted,"
                    "CD4_Naive,CD4_Activated,CD4_Helper,CD4_Treg,M1,M2,MDSC\n")
    with open(csv_file, 'a') as f:
        f.write(f"{step_num},{counts['Cancer_Stem']},{counts['Cancer_Prog']},{counts['Cancer_Senes']},"
                f"{counts['CD8_Naive']},{counts['CD8_Activated']},{counts['CD8_Memory']},{counts['CD8_Exhausted']},"
                f"{counts['CD4_Naive']},{counts['CD4_Activated']},{counts['CD4_Helper']},{counts['CD4_Treg']},{counts['M1']},{counts['M2']},{counts['MDSC']}\n")

def run_simulation_verbose(
    scenario='standard',
    custom_params=None,
    print_every=1,
    save_data=True,
    show_molecules=False,
    pdl1_pd1_axis=True         
):
    """
    Run simulation with detailed step-by-step output.
    
    Parameters:
    -----------
    scenario : str
        Predefined scenario name ('small_test', 'standard', 'large_tumor', 'immune_rich')
    custom_params : dict
        Custom parameters to override scenario defaults
    print_every : int
        Print summary every N steps (1 = every step)
    save_data : bool
        Whether to save step data to CSV
    show_molecules : bool
        Whether to print molecular concentrations (can be verbose)
    """
    
    params = DEFAULT_SCENARIOS[scenario].copy()
    
    # Override with custom parameters if provided
    if custom_params is not None:
        params.update(custom_params)
    
    # Extract parameters
    steps = params['steps']
    width = params['width']
    height = params['height']
    initial_tumor_cells = params['initial_tumor_cells']
    initial_CD8Tcells = params['initial_CD8Tcells']
    initial_CD4Tcells = params['initial_CD4Tcells']
    initial_macrophages = params['initial_macrophages']
    initial_MDSC = params['initial_MDSC']
    
    print("="*80)
    print("STARTING ABM SIMULATION")
    print("="*80)
    print(f"Grid size: {width}x{height}")
    print(f"Simulation steps: {steps}")
    print(f"Initial populations:")
    print(f"  Cancer cells: {initial_tumor_cells}")
    print(f"  CD8+ T cells: {initial_CD8Tcells}")
    print(f"  CD4+ T cells: {initial_CD4Tcells}")
    print(f"  Macrophages:  {initial_macrophages}")
    print(f"  MDSCs:        {initial_MDSC}")

    # Initialize model
    start_time = time.time()
    model = ABM_Model(
        width=width, 
        height=height, 
        initial_tumor_cells=initial_tumor_cells,
        initial_CD8Tcells=initial_CD8Tcells,
        initial_CD4Tcells=initial_CD4Tcells,
        initial_macrophages=initial_macrophages,
        initial_MDSC=initial_MDSC,
        pdl1_pd1_axis=pdl1_pd1_axis
    )
    
    init_time = time.time() - start_time
    print(f"\nModel initialized in {init_time:.3f}s")

    counts = get_cell_counts(model)
    
    # Print initial state
    if print_every <= 1:
        print_step_summary(counts, 0)
        if show_molecules:
            print_molecular_summary(model)

    if save_data:
        save_step_data(counts, 0)

    # Run simulation
    total_step_time = 0
    for step in range(1, steps + 1):
        step_start = time.time()
        
        model.step()
        step_time = time.time() - step_start
        total_step_time += step_time

        counts = get_cell_counts(model)

        if step % print_every == 0:
            print_step_summary(counts, step, step_time)
            if show_molecules:
                print_molecular_summary(model)

        if save_data:
            save_step_data(counts, step)

        # Check if simulation should stop
        if not model.running:
            print(f"\n{'!'*60}")
            print(f"SIMULATION TERMINATED AT STEP {step}")
            print(f"Reason: No cancer cells remaining")
            print(f"{'!'*60}")
            break
    
    # Final summary
    final_time = time.time() - start_time
    avg_step_time = total_step_time / step if step > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"SIMULATION COMPLETED")
    print(f"{'='*80}")
    print(f"Total simulation time: {final_time:.3f}s")
    print(f"Average time per step: {avg_step_time:.3f}s")
    print(f"Steps completed: {step}/{steps}")
    print(f"Final model state:")
    counts = get_cell_counts(model)
    print_step_summary(counts, step)

    return model

### Visualization functions

COLOR_MAP = {
    # cancer
    "Cancer_Stem": "green",
    "Cancer_Prog": "gold",
    "Cancer_Senes": "orangered",

    # CD8 (blue gradient)
    "CD8_Naive": "#ade8f4",
    "CD8_Activated": "#00b4d8",
    "CD8_Memory": "#023e8a",
    "CD8_Exhausted": "#03045e",

    # CD4 (red gradient)
    "CD4_Naive": "#fc9ca2",
    "CD4_Activated": "#f92432",
    "CD4_Helper": "#9f040e",
    "CD4_Treg": "#500207",

    # myeloid
    "MDSC": "dimgray",
    "M1": "cyan",
    "M2": "magenta",
}

GRID_COLORS = [
    "white",                          # 0 empty
    COLOR_MAP["Cancer_Stem"],         # 1
    COLOR_MAP["Cancer_Prog"],         # 2
    COLOR_MAP["Cancer_Senes"],        # 3
    COLOR_MAP["CD8_Naive"],           # 4
    COLOR_MAP["CD8_Activated"],       # 5
    COLOR_MAP["CD8_Memory"],          # 6
    COLOR_MAP["CD4_Helper"],          # 7
    COLOR_MAP["CD4_Treg"],            # 8
    COLOR_MAP["MDSC"],                # 9
    COLOR_MAP["M1"],                  # 10
    COLOR_MAP["M2"],                  # 11
]

def encode_grid(model):
    grid_map = np.zeros((model.width, model.height), dtype=int)

    for x in range(model.width):
        for y in range(model.height):
            cell = model.grid[x][y]

            if cell is None:
                continue
            elif isinstance(cell, CancerCell):
                grid_map[x, y] = {
                    CancerSubtype.STEM: 1,
                    CancerSubtype.PROGENITOR: 2,
                    CancerSubtype.SENESCENT: 3
                }[cell.subtype]

            elif isinstance(cell, CD8TCell):
                grid_map[x, y] = {
                    CD8DiffState.CD8TNAIVE: 4,
                    CD8DiffState.CD8TACTIVATED: 5,
                    CD8DiffState.CD8TMEMORY: 6
                }[cell.diff_state]

            elif isinstance(cell, CD4TCell):
                if cell.diff_state == CD4DiffState.CD4THELPER:
                    grid_map[x, y] = 7
                elif cell.diff_state == CD4DiffState.CD4TREG:
                    grid_map[x, y] = 8

            elif isinstance(cell, MDSC):
                grid_map[x, y] = 9

            elif isinstance(cell, Macrophage):
                grid_map[x, y] = 10 if cell.subtype == MacSubtype.M1 else 11

    return grid_map

def plot_grid(model, output_dir):
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
    grid_map = encode_grid(model)

    # Define colormap and normalization
    cmap = mcolors.ListedColormap(GRID_COLORS)
    bounds = np.arange(-0.5, len(GRID_COLORS)-0.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 8))
    plt.imshow(grid_map.T, cmap=cmap, norm=norm, origin="lower", interpolation='nearest')
    cbar = plt.colorbar(ticks=list(range(12)), shrink=0.8)
    cbar.set_ticklabels([
        "Empty",
        "Cancer Stem",
        "Cancer Prog",
        "Cancer Senescent", 
        "CD8+ Naive",
        "CD8+ Activated",
        "CD8+ Memory",
        "CD4+ Helper",
        "CD4+ Treg",
        "MDSC",
        "M1 Mac",
        "M2 Mac"
    ])
    plt.title("Final Cell Distribution on Grid", fontsize=14, fontweight='bold')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    
    fig = plt.gcf()   # get current figure
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'final_cell_grid.png'), 
                dpi=300, bbox_inches='tight')
    
    return fig


def plot_cytokine_concentrations(model, output_dir):
    """
    Plot the 2D concentration fields for all cytokines and save them.
    """
    # All available cytokines/signals
    signals = [
        ('Arg1', 'Arginase-1'),
        ('CCL2', 'CCL2 (MCP-1)'),
        ('IFNg', 'IFN-γ'),
        ('IL2', 'IL-2'),
        ('IL10', 'IL-10'),
        ('IL12', 'IL-12'),
        ('NO', 'Nitric Oxide'),
        ('TGFb', 'TGF-β'),
        ('VEGFA', 'VEGF-A')
    ]
    
    # Create a large figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, (signal_name, display_name) in enumerate(signals):
        ax = axes[idx]
        
        if hasattr(model, f"{signal_name}_field"):
            signal_grid = getattr(model, f"{signal_name}_field")
            
            # Use log scale for better visualization if values span many orders of magnitude
            signal_max = np.max(signal_grid)
            positive_vals = signal_grid[signal_grid > 0]
            if positive_vals.size == 0:
                signal_min = signal_max * 1e-10
            else:
                signal_min = np.max([np.min(positive_vals), signal_max * 1e-10])
            
            if signal_max > 0:
                # Use log-normalized colormap if there's significant range
                if signal_max / signal_min > 100:
                    from matplotlib.colors import LogNorm
                    im = ax.imshow(signal_grid.T, origin="lower", cmap="viridis", 
                                 norm=LogNorm(vmin=signal_min, vmax=signal_max))
                else:
                    im = ax.imshow(signal_grid.T, origin="lower", cmap="viridis",
                                 vmin=0, vmax=signal_max)
            else:
                im = ax.imshow(signal_grid.T, origin="lower", cmap="viridis")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Concentration\n(molecules/grid)', fontsize=8)
            
            # Add statistics as text
            mean_val = np.mean(signal_grid)
            max_val = np.max(signal_grid)
            ax.text(0.02, 0.98, f'Max: {max_val:.1e}\nMean: {mean_val:.1e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        else:
            ax.text(0.5, 0.5, f"{signal_name}\nfield not found", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_title(display_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=10)
        ax.set_ylabel('Y Position', fontsize=10)

    plt.suptitle('Final Cytokine and Signal Molecule Concentrations', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'final_cytokine_concentrations.png'), 
                dpi=300, bbox_inches='tight')
    
    return fig

def plot_tcells(ax, df):
    # CD8
    ax.plot(df['Step'], df['CD8_Naive'], color=COLOR_MAP['CD8_Naive'], label='CD8 Naive')
    ax.plot(df['Step'], df['CD8_Activated'], color=COLOR_MAP['CD8_Activated'], linestyle='--', label='CD8 Activated')
    ax.plot(df['Step'], df['CD8_Memory'], color=COLOR_MAP['CD8_Memory'], linestyle=':', label='CD8 Memory')
    ax.plot(df['Step'], df['CD8_Exhausted'], color=COLOR_MAP['CD8_Exhausted'], linestyle='-.', label='CD8 Exhausted')

    # CD4
    ax.plot(df['Step'], df['CD4_Naive'], color=COLOR_MAP['CD4_Naive'], label='CD4 Naive')
    ax.plot(df['Step'], df['CD4_Activated'], color=COLOR_MAP['CD4_Activated'], linestyle='--', label='CD4 Activated')
    ax.plot(df['Step'], df['CD4_Helper'], color=COLOR_MAP['CD4_Helper'], linestyle=':', label='CD4 Helper')
    ax.plot(df['Step'], df['CD4_Treg'], color=COLOR_MAP['CD4_Treg'], linestyle='-.', label='CD4 Treg')

def plot_immune_population(df, output_dir=None, ax=None):
    """
    Plot CD8 and CD4 T-cell populations with differentiated states using color shades.
    """
    try:
        # Track whether we created the axis
        created_ax = ax is None

        if created_ax:
            fig, ax = plt.subplots(figsize=(20, 12))
        else:
            fig = ax.figure

        plot_tcells(ax, df)

        # Labels
        ax.set_xlabel('Step')
        ax.set_ylabel('Cell Count')
        ax.set_title('T Cell Populations Over Time', fontweight='bold')

        # Legend & grid
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)

        if created_ax and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_dir, 'immune_population_over_time.png'),
                dpi=300,
                bbox_inches='tight'
            )

        return fig

    except Exception as e:
        print(f"Error generating plot: {e}")
        return None

def plot_summary_dashboard(model, output_dir):
    """
    Create a comprehensive dashboard with cell grid, population dynamics, and key cytokines.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Cell grid (top left)
    ax1 = plt.subplot(2, 4, 1)

    grid_map = encode_grid(model)

    cmap = mcolors.ListedColormap(GRID_COLORS)
    bounds = np.arange(-0.5, len(GRID_COLORS)-0.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im1 = ax1.imshow(grid_map.T, cmap=cmap, norm=norm, origin="lower")

    ax1.set_title('Final Cell Distribution', fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # 2. Population dynamics (top middle-right)
    ax2 = plt.subplot(2, 4, (2, 4))
    try:
        df = model.datacollector.get_model_vars_dataframe()
        ax2.plot(df.index, df['CancerCellCount'], 'r-', linewidth=3, label='Cancer Cells')
        ax2.plot(df.index, df['CD8TCount'], 'b-', linewidth=2, label='CD8+ T Cells')
        ax2.plot(df.index, df['CD4TCount'], 'purple', linewidth=2, label='CD4+ T Cells')
        ax2.plot(df.index, df['MacCount'], 'cyan', linewidth=2, label='Macrophages')
        ax2.plot(df.index, df['MDSCCount'], 'gray', linewidth=2, label='MDSCs')
        ax2.set_xlabel('Simulation Step')
        ax2.set_ylabel('Cell Count')
        ax2.set_title('Population Dynamics Over Time', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    except:
        ax2.text(0.5, 0.5, 'Population data\nnot available', ha='center', va='center')
    
    # 3-6. Key cytokines (bottom row)
    key_signals = [('TGFb', 'TGF-β'), ('IL10', 'IL-10'), ('IFNg', 'IFN-γ'), ('IL2', 'IL-2')]
    
    for idx, (signal_name, display_name) in enumerate(key_signals):
        ax = plt.subplot(2, 4, 5 + idx)
        
        if hasattr(model, f"{signal_name}_field"):
            signal_grid = getattr(model, f"{signal_name}_field")
            signal_max = np.max(signal_grid)
            
            if signal_max > 0:
                im = ax.imshow(signal_grid.T, origin="lower", cmap="viridis", vmin=0, vmax=signal_max)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(np.zeros_like(signal_grid).T, origin="lower", cmap="viridis")
                
            ax.text(0.02, 0.98, f'Max: {signal_max:.1e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Data not\navailable', ha='center', va='center')
        
        ax.set_title(display_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=9)
        ax.set_ylabel('Y Position', fontsize=9)

    plt.suptitle('ABM Simulation Summary Dashboard', fontsize=18, fontweight='bold', y=0.98)

    # Leave the top 8% of the figure for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save the dashboard
    plt.savefig(os.path.join(output_dir, 'simulation_dashboard.png'), 
                dpi=300, bbox_inches='tight')
    
    return fig

def plot_results_from_csv(csv_file="cell_counts.csv", output_dir="simulation_output", show=True):
    """Plot results from saved CSV data and save to output_dir."""
    # Read CSV
    try:
        df = pd.read_csv(csv_file)

        fig = plt.figure(figsize=(15, 10))

        # Cancer cells subplot
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(df['Step'], df['Cancer_Stem'], 'g-', label='Stem', linewidth=2)
        ax1.plot(df['Step'], df['Cancer_Prog'], color='orange', label='Progenitor', linewidth=2)
        ax1.plot(df['Step'], df['Cancer_Senes'], 'r-', label='Senescent', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Cell Count')
        ax1.set_title('Cancer Cell Populations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Immune cells subplot
        ax2 = fig.add_subplot(2, 2, 2)
        plot_immune_population(df, output_dir, ax=ax2)

        # Myeloid cells subplot
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(df['Step'], df['M1'], color='cyan', label='M1 Macrophage', linewidth=2)
        ax3.plot(df['Step'], df['M2'], color='magenta', label='M2 Macrophage', linewidth=2)
        ax3.plot(df['Step'], df['MDSC'], color='gray', label='MDSC', linewidth=2)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Cell Count')
        ax3.set_title('Myeloid Cell Populations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Total populations subplot
        ax4 = fig.add_subplot(2, 2, 4)
        total_cancer = df['Cancer_Stem'] + df['Cancer_Prog'] + df['Cancer_Senes']
        total_immune = (df['CD8_Naive'] + df['CD8_Activated'] + df['CD8_Memory'] + df['CD8_Exhausted'] +
                        df['CD4_Helper'] + df['CD4_Treg'] +
                        df['M1'] + df['M2'] + df['MDSC'])
        ax4.plot(df['Step'], total_cancer, color='red', label='Total Cancer', linewidth=3)
        ax4.plot(df['Step'], total_immune, color='blue', label='Total Immune', linewidth=3)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Cell Count')
        ax4.set_title('Cancer vs Immune Populations')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()

        # Save
        fig.savefig(os.path.join(output_dir, "population_dynamics.png"), 
                    dpi=300, bbox_inches="tight")

    except ImportError:
        print("pandas not available for plotting CSV data")
    except FileNotFoundError:
        print(f"CSV file {csv_file} not found")
        return None
    return fig 
