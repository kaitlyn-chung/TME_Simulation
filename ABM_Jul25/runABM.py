
#!/usr/bin/env python3
"""
Enhanced ABM runner with detailed step-by-step output and monitoring.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import random
import time
import os

from ABM_Jul25.agents import (
    CancerCell, CancerSubtype,
    CD8TCell,
    CD4TCell, CD4TSubtype,
    MDSC,
    Macrophage, MacSubtype
)
from ABM_Jul25.model import ABM_Model
import ABM_Jul25.params as P


def print_step_summary(model, step_num, step_time=None):
    """Print detailed summary of current model state."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}")
    if step_time:
        print(f"Step execution time: {step_time:.3f}s")
    print(f"{'='*60}")
    
    # Count all cell types and subtypes
    cancer_stem = model.count_cell_type(CancerCell, subtype=CancerSubtype.STEM)
    cancer_prog = model.count_cell_type(CancerCell, subtype=CancerSubtype.PROGENITOR)
    cancer_senes = model.count_cell_type(CancerCell, subtype=CancerSubtype.SENESCENT)
    total_cancer = cancer_stem + cancer_prog + cancer_senes
    
    cd8_count = model.count_cell_type(CD8TCell)
    cd4_helper = model.count_cell_type(CD4TCell, subtype=CD4TSubtype.CD4THELPER)
    cd4_treg = model.count_cell_type(CD4TCell, subtype=CD4TSubtype.CD4TREG)
    total_cd4 = cd4_helper + cd4_treg
    
    m1_count = model.count_cell_type(Macrophage, subtype=MacSubtype.M1)
    m2_count = model.count_cell_type(Macrophage, subtype=MacSubtype.M2)
    total_mac = m1_count + m2_count
    
    mdsc_count = model.count_cell_type(MDSC)
    
    total_immune = cd8_count + total_cd4 + total_mac + mdsc_count
    total_cells = total_cancer + total_immune
    
    print(f"CELL POPULATIONS:")
    print(f"  Cancer Cells (Total: {total_cancer:3d})")
    print(f"    ├─ Stem:       {cancer_stem:3d}")
    print(f"    ├─ Progenitor: {cancer_prog:3d}")
    print(f"    └─ Senescent:  {cancer_senes:3d}")
    print(f"")
    print(f"  Immune Cells (Total: {total_immune:3d})")
    print(f"    ├─ CD8+ T:     {cd8_count:3d}")
    print(f"    ├─ CD4+ T:     {total_cd4:3d} (Helper: {cd4_helper}, Treg: {cd4_treg})")
    print(f"    ├─ Macrophage: {total_mac:3d} (M1: {m1_count}, M2: {m2_count})")
    print(f"    └─ MDSC:       {mdsc_count:3d}")
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
    
    print(f"KEY RATIOS:")
    print(f"  Immune:Cancer = {immune_cancer_ratio:.2f}")
    print(f"  CD8:Cancer    = {cd8_cancer_ratio:.2f}")
    print(f"  M1:M2         = {m1_m2_ratio:.2f}")
    print(f"  Helper:Treg   = {helper_treg_ratio:.2f}")


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


def save_step_data(model, step_num, output_dir="simulation_output"):
    """Save step data to files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save cell counts
    cancer_stem = model.count_cell_type(CancerCell, subtype=CancerSubtype.STEM)
    cancer_prog = model.count_cell_type(CancerCell, subtype=CancerSubtype.PROGENITOR)
    cancer_senes = model.count_cell_type(CancerCell, subtype=CancerSubtype.SENESCENT)
    cd8_count = model.count_cell_type(CD8TCell)
    cd4_helper = model.count_cell_type(CD4TCell, subtype=CD4TSubtype.CD4THELPER)
    cd4_treg = model.count_cell_type(CD4TCell, subtype=CD4TSubtype.CD4TREG)
    m1_count = model.count_cell_type(Macrophage, subtype=MacSubtype.M1)
    m2_count = model.count_cell_type(Macrophage, subtype=MacSubtype.M2)
    mdsc_count = model.count_cell_type(MDSC)
    
    # Append to CSV file
    csv_file = os.path.join(output_dir, "cell_counts.csv")
    if step_num == 0:
        with open(csv_file, 'w') as f:
            f.write("Step,Cancer_Stem,Cancer_Prog,Cancer_Senes,CD8,CD4_Helper,CD4_Treg,M1,M2,MDSC\n")
    
    with open(csv_file, 'a') as f:
        f.write(f"{step_num},{cancer_stem},{cancer_prog},{cancer_senes},{cd8_count},"
                f"{cd4_helper},{cd4_treg},{m1_count},{m2_count},{mdsc_count}\n")


# Default simulation scenarios
DEFAULT_SCENARIOS = {
    'small_test': {
        'steps': 20,
        'width': 20,
        'height': 20,
        'initial_tumor_cells': 20,
        'initial_CD8Tcells': 5,
        'initial_CD4Tcells': 8,
        'initial_macrophages': 8,
        'initial_MDSC': 3
    },
    'standard': {
        'steps': 100,
        'width': 30,
        'height': 30,
        'initial_tumor_cells': 80,
        'initial_CD8Tcells': 20,
        'initial_CD4Tcells': 30,
        'initial_macrophages': 30,
        'initial_MDSC': 10
    },
    'large_tumor': {
        'steps': 150,
        'width': 50,
        'height': 50,
        'initial_tumor_cells': 200,
        'initial_CD8Tcells': 40,
        'initial_CD4Tcells': 60,
        'initial_macrophages': 60,
        'initial_MDSC': 20
    },
    'immune_rich': {
        'steps': 100,
        'width': 30,
        'height': 30,
        'initial_tumor_cells': 30,
        'initial_CD8Tcells': 40,
        'initial_CD4Tcells': 50,
        'initial_macrophages': 40,
        'initial_MDSC': 5
    }
}


def run_simulation_verbose(
    scenario='standard',
    custom_params=None,
    print_every=1,
    save_data=True,
    show_molecules=False
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
    
    # Get base parameters from scenario
    if scenario not in DEFAULT_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(DEFAULT_SCENARIOS.keys())}")
    
    params = DEFAULT_SCENARIOS[scenario].copy()
    
    # Override with custom parameters if provided
    if custom_params:
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
        initial_MDSC=initial_MDSC
    )
    
    init_time = time.time() - start_time
    print(f"\nModel initialized in {init_time:.3f}s")
    
    # Print initial state
    if print_every <= 1:
        print_step_summary(model, 0)
        if show_molecules:
            print_molecular_summary(model)
    
    if save_data:
        save_step_data(model, 0)
    
    # Run simulation
    total_step_time = 0
    for step in range(1, steps + 1):
        step_start = time.time()
        
        model.step()
        
        step_time = time.time() - step_start
        total_step_time += step_time
        
        # Print progress
        if step % print_every == 0:
            print_step_summary(model, step, step_time)
            if show_molecules:
                print_molecular_summary(model)
        
        # Save data
        if save_data:
            save_step_data(model, step)
        
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
    print_step_summary(model, step)
    
    return model


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

    plt.figure(figsize=(10, 8))
    plt.imshow(grid_map.T, cmap=cmap, norm=norm, origin="lower")
    cbar = plt.colorbar(ticks=list(range(10)), shrink=0.8)
    cbar.set_ticklabels([
        "Empty",
        "Cancer Stem",
        "Cancer Prog",
        "Cancer Senescent", 
        "CD8+ T",
        "CD4+ Helper",
        "CD4+ Treg",
        "MDSC",
        "M1 Mac",
        "M2 Mac"
    ])
    plt.title("Final Cell Distribution on Grid", fontsize=14, fontweight='bold')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    return plt


def plot_cytokine_concentrations(model, output_dir="simulation_output"):
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
            signal_min = np.max([np.min(signal_grid[signal_grid > 0]), signal_max * 1e-10])
            
            if signal_max > 0:
                # Use log-normalized colormap if there's significant range
                if signal_max / signal_min > 100:
                    from matplotlib.colors import LogNorm
                    im = ax.imshow(signal_grid.T, origin="lower", cmap="plasma", 
                                 norm=LogNorm(vmin=signal_min, vmax=signal_max))
                else:
                    im = ax.imshow(signal_grid.T, origin="lower", cmap="plasma",
                                 vmin=0, vmax=signal_max)
            else:
                im = ax.imshow(signal_grid.T, origin="lower", cmap="plasma")
            
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'final_cytokine_concentrations.png'), 
                dpi=300, bbox_inches='tight')
    
    return fig


def plot_summary_dashboard(model, output_dir="simulation_output"):
    """
    Create a comprehensive dashboard with cell grid, population dynamics, and key cytokines.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Cell grid (top left)
    ax1 = plt.subplot(2, 4, 1)
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
                else:
                    grid_map[x, y] = 3
            elif isinstance(occupant, CD8TCell):
                grid_map[x, y] = 4
            elif isinstance(occupant, CD4TCell):
                if occupant.subtype == CD4TSubtype.CD4THELPER:
                    grid_map[x, y] = 5
                else:
                    grid_map[x, y] = 6
            elif isinstance(occupant, MDSC):
                grid_map[x, y] = 7
            elif isinstance(occupant, Macrophage):
                if occupant.subtype == MacSubtype.M1:
                    grid_map[x, y] = 8
                else:
                    grid_map[x, y] = 9

    cmap = mcolors.ListedColormap([
        "white", "lightgreen", "orange", "red", "blue", 
        "purple", "pink", "gray", "cyan", "magenta"
    ])
    bounds = np.arange(-0.5, 10.5, 1.0)
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
                im = ax.imshow(signal_grid.T, origin="lower", cmap="plasma", vmin=0, vmax=signal_max)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(np.zeros_like(signal_grid).T, origin="lower", cmap="plasma")
                
            ax.text(0.02, 0.98, f'Max: {signal_max:.1e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Data not\navailable', ha='center', va='center')
        
        ax.set_title(display_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=9)
        ax.set_ylabel('Y Position', fontsize=9)

    plt.suptitle('ABM Simulation Summary Dashboard', fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Save the dashboard
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'simulation_dashboard.png'), 
                dpi=300, bbox_inches='tight')
    
    return fig
    """Plot results from saved CSV data."""
    try:
        import pandas as pd
        csv_file="simulation_output/cell_counts.csv"
        df = pd.read_csv(csv_file)
        
        plt.figure(figsize=(15, 10))
        
        # Cancer cells subplot
        plt.subplot(2, 2, 1)
        plt.plot(df['Step'], df['Cancer_Stem'], 'g-', label='Stem', linewidth=2)
        plt.plot(df['Step'], df['Cancer_Prog'], 'orange', label='Progenitor', linewidth=2)
        plt.plot(df['Step'], df['Cancer_Senes'], 'r-', label='Senescent', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Cell Count')
        plt.title('Cancer Cell Populations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Immune cells subplot
        plt.subplot(2, 2, 2)
        plt.plot(df['Step'], df['CD8'], 'b-', label='CD8+ T', linewidth=2)
        plt.plot(df['Step'], df['CD4_Helper'], 'purple', label='CD4+ Helper', linewidth=2)
        plt.plot(df['Step'], df['CD4_Treg'], 'pink', label='CD4+ Treg', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Cell Count')
        plt.title('T Cell Populations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Myeloid cells subplot
        plt.subplot(2, 2, 3)
        plt.plot(df['Step'], df['M1'], 'cyan', label='M1 Macrophage', linewidth=2)
        plt.plot(df['Step'], df['M2'], 'magenta', label='M2 Macrophage', linewidth=2)
        plt.plot(df['Step'], df['MDSC'], 'gray', label='MDSC', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Cell Count')
        plt.title('Myeloid Cell Populations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Total populations subplot
        plt.subplot(2, 2, 4)
        total_cancer = df['Cancer_Stem'] + df['Cancer_Prog'] + df['Cancer_Senes']
        total_immune = df['CD8'] + df['CD4_Helper'] + df['CD4_Treg'] + df['M1'] + df['M2'] + df['MDSC']
        plt.plot(df['Step'], total_cancer, 'red', label='Total Cancer', linewidth=3)
        plt.plot(df['Step'], total_immune, 'blue', label='Total Immune', linewidth=3)
        plt.xlabel('Step')
        plt.ylabel('Cell Count')
        plt.title('Cancer vs Immune Populations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("pandas not available for plotting CSV data")
    except FileNotFoundError:
        print(f"CSV file {csv_file} not found")


def plot_results_from_csv(csv_file="simulation_output/cell_counts.csv"):
    """Plot results from saved CSV data."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        plt.figure(figsize=(15, 10))
        
        # Cancer cells subplot
        plt.subplot(2, 2, 1)
        plt.plot(df['Step'], df['Cancer_Stem'], 'g-', label='Stem', linewidth=2)
        plt.plot(df['Step'], df['Cancer_Prog'], 'orange', label='Progenitor', linewidth=2)
        plt.plot(df['Step'], df['Cancer_Senes'], 'r-', label='Senescent', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Cell Count')
        plt.title('Cancer Cell Populations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Immune cells subplot
        plt.subplot(2, 2, 2)
        plt.plot(df['Step'], df['CD8'], 'b-', label='CD8+ T', linewidth=2)
        plt.plot(df['Step'], df['CD4_Helper'], 'purple', label='CD4+ Helper', linewidth=2)
        plt.plot(df['Step'], df['CD4_Treg'], 'pink', label='CD4+ Treg', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Cell Count')
        plt.title('T Cell Populations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Myeloid cells subplot
        plt.subplot(2, 2, 3)
        plt.plot(df['Step'], df['M1'], 'cyan', label='M1 Macrophage', linewidth=2)
        plt.plot(df['Step'], df['M2'], 'magenta', label='M2 Macrophage', linewidth=2)
        plt.plot(df['Step'], df['MDSC'], 'gray', label='MDSC', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Cell Count')
        plt.title('Myeloid Cell Populations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Total populations subplot
        plt.subplot(2, 2, 4)
        total_cancer = df['Cancer_Stem'] + df['Cancer_Prog'] + df['Cancer_Senes']
        total_immune = df['CD8'] + df['CD4_Helper'] + df['CD4_Treg'] + df['M1'] + df['M2'] + df['MDSC']
        plt.plot(df['Step'], total_cancer, 'red', label='Total Cancer', linewidth=3)
        plt.plot(df['Step'], total_immune, 'blue', label='Total Immune', linewidth=3)
        plt.xlabel('Step')
        plt.ylabel('Cell Count')
        plt.title('Cancer vs Immune Populations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("pandas not available for plotting CSV data")
    except FileNotFoundError:
        print(f"CSV file {csv_file} not found")

# Main execution
if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print("Available scenarios:")
    for name, params in DEFAULT_SCENARIOS.items():
        print(f"  {name}: {params['initial_tumor_cells']} cancer, "
              f"{params['initial_CD8Tcells']+params['initial_CD4Tcells']} T cells, "
              f"{params['steps']} steps")
    
    # Option 1: Use a predefined scenario
    print(f"\nRunning 'standard' scenario...")
    model = run_simulation_verbose(
        scenario='standard',
        print_every=5,  # Print every 5 steps
        save_data=True,
        show_molecules=False
    )
    
    # Option 2: Use custom parameters
    # print(f"\nRunning custom simulation...")
    # model = run_simulation_verbose(
    #     scenario='standard',
    #     custom_params={
    #         'initial_tumor_cells': 75,  # Override just this parameter
    #         'steps': 30
    #     },
    #     print_every=1,
    #     save_data=True,
    #     show_molecules=True
    # )
    
    # Generate all visualization outputs
    print("\nGenerating comprehensive outputs...")
    
    # Generate outputs individually since generate_all_outputs wasn't defined
    if not os.path.exists("simulation_output"):
        os.makedirs("simulation_output")
    
    print("  - Final cell distribution grid...")
    fig1 = plot_grid(model)
    fig1.savefig("simulation_output/final_cell_grid.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("  - Cytokine concentration fields...")
    fig2 = plot_cytokine_concentrations(model, "simulation_output")
    plt.close(fig2)
    
    print("  - Summary dashboard...")
    fig3 = plot_summary_dashboard(model, "simulation_output")
    plt.close(fig3)
    
    print("  - Population dynamics...")
    plot_results_from_csv("simulation_output/cell_counts.csv")
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE!")
    print("="*80)
    print("Check 'simulation_output/' directory for:")
    print("  📊 population_dynamics.png - Cell counts over time")
    print("  🗺️  final_cell_grid.png - Final spatial distribution") 
    print("  🧪 final_cytokine_concentrations.png - All molecular fields")
    print("  📈 simulation_dashboard.png - Comprehensive summary")
    print("  📋 cell_counts.csv - Raw data for further analysis")
    print("="*80)