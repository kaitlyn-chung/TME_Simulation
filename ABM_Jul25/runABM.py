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

# Imports from this package
from ABM_Jul25.agents import (
    CancerCell, CancerSubtype,
    CD8TCell, CD8DiffState, CD8ExhaustionState,
    CD4TCell, CD4DiffState,
    MDSC,
    Macrophage, MacSubtype
)
from ABM_Jul25.model import ABM_Model
import ABM_Jul25.params as P
from ABM_Jul25.scenarios import (DEFAULT_SCENARIOS, get_value, confirm_value)
from ABM_Jul25.plot_simulation import (run_simulation_verbose,
                                        plot_grid,
                                        plot_cytokine_concentrations,
                                        plot_summary_dashboard,
                                        plot_results_from_csv)
import warnings
warnings.filterwarnings("ignore")

pdl1_pd1_axis = False

# Main execution
def main():
    # Create directories for outputs to land
    if not os.path.exists("simulation_output"):
        os.makedirs("simulation_output")
    output_dir = "simulation_output"

    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    print("Available scenarios:")
    for name, params in DEFAULT_SCENARIOS.items():
        if name == 'custom':
            print("  custom: You will be prompted to specify initial conditions for cancer cell, T cell, and number of steps size.")
        else:
            print(f"  {name}: {params['initial_tumor_cells']} cancer, "
                f"{params['initial_CD8Tcells']+params['initial_CD4Tcells']} T cells, "
                f"{params['steps']} steps")
    
    scenario_names = list(DEFAULT_SCENARIOS.keys())
    choice = input('Which scenario would you like to choose? ')
    while choice not in scenario_names:
        print('Invalid scenario name! Check your spelling and case senstiivity. ')
        choice = input('Which scenario would you like to choose? ')

    # Using a custom scenario, prompt the user for initializations
    if choice == 'custom':
        inputs = [
            ("steps", "Step count", "Enter number of steps: ", 2),
            ("width", "Width", "Enter width: ", 1),
            ("height", "Height", "Enter height: ", 1),
            ("initial_tumor_cells", "Initial tumor cell population", "Enter tumor cells: ", 0),
            ("initial_CD8Tcells", "Initial CD8+ T cells", "Enter CD8 count: ", 0),
            ("initial_CD4Tcells", "Initial CD4+ T cells", "Enter CD4 count: ", 0),
            ("initial_macrophages", "Initial macrophages", "Enter macrophages: ", 0),
            ("initial_MDSC", "Initial MDSCs", "Enter MDSCs: ", 0),
        ]

        param_inputs = {}
        for key, name, prompt, min_val in inputs:
            param_inputs[key] = get_value(name, prompt, min_val)
    
        model = run_simulation_verbose(
            scenario='custom',
            custom_params=param_inputs,
            print_every=5,
            save_data=True,
            show_molecules=True,
            pdl1_pd1_axis=pdl1_pd1_axis
            )

    else:
        # Use a predefined scenario
        print(f"\nRunning {choice} scenario...")
        model = run_simulation_verbose(
            scenario=choice,
            print_every=5,
            save_data=True,
            show_molecules=False,
            pdl1_pd1_axis=pdl1_pd1_axis        
        )
    
    # Generate all visualization outputs
    print("\nGenerating comprehensive outputs...")
    
    # Final cell distribution
    print("  - Final cell distribution grid...")
    fig1 = plot_grid(model, output_dir)
    plt.close(fig1)
    
    # All cytokine concentrations
    print("  - Cytokine concentration fields...")
    fig2 = plot_cytokine_concentrations(model, output_dir)
    plt.close(fig2)
    
    # Sumamry dashboard
    print("  - Summary dashboard...")
    fig3 = plot_summary_dashboard(model, output_dir)
    plt.close(fig3)
    
    # Evolution of population growth
    print("  - Population dynamics...")
    csv_path = os.path.join(output_dir, "cell_counts.csv")
    fig4 = plot_results_from_csv(csv_path, output_dir)
    plt.close(fig4)

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
    print("Move any files you want to save out of the 'simulation_output/' directory before doing a new run! They will be overridden! ")
    print("="*80)

if __name__=="__main__":
    main()