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
    CD8TCell,
    CD4TCell, CD4TSubtype,
    MDSC,
    Macrophage, MacSubtype
)
from ABM_Jul25.model import ABM_Model
import ABM_Jul25.params as P
from ABM_Jul25.scenarios import (DEFAULT_SCENARIOS, get_value)
from ABM_Jul25.plot_simulation import (run_simulation_verbose,
                                        plot_grid,
                                        plot_cytokine_concentrations,
                                        plot_summary_dashboard,
                                        plot_results_from_csv)

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
        print(f"\nRunning custom simulation...")
        steps = get_value('Step count', 'Enter the number of steps for this model. Please enter a number greater than 2: ', 2)
        width = get_value('Width', 'Enter a width for the TME for this model. Please enter a number greater than 1: ')
        height = get_value('Height', 'Enter a height for the TME for this model. Please enter a number greater than 1: ')
        int_tumor_cell = get_value('Initial tumor cell population', 'Enter an initial tumor cell population for this model. Please enter a number greater than 1: ')
        int_cd8 = get_value('Initial CD8+ T-cell population', 'Enter an initial CD8+ T-cell population for this model. Please enter a number greater than 1: ')
        int_cd4 = get_value('Initial CD4+ T-cell population', 'Enter an initial CD4+ T-cell population for this model. Please enter a number greater than 1: ')
        int_macrophage = get_value('Initial macrophage cell population', 'Enter an initial macrophage cell population for this model. Please enter a number greater than 1: ')
        int_mdsc = get_value('Initial MDSC population', 'Enter an initial MDSC population for this model. Please enter a number greater than 1: ')

        model = run_simulation_verbose(
            scenario='custom',
            custom_params={
                'steps': steps,
                'width': width,
                'height': height,
                'initial_tumor_cells': int_tumor_cell,
                'initial_CD8Tcells': int_cd8,
                'initial_CD4Tcells': int_cd4,
                'initial_macrophages': int_macrophage,
                'initial_MDSC': int_mdsc
            },
            print_every=5,
            save_data=True,
            show_molecules=True
        )

    else:
        # Use a predefined scenario
        print(f"\nRunning {choice} scenario...")
        model = run_simulation_verbose(
            scenario=choice,
            print_every=5,
            save_data=True,
            show_molecules=False
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