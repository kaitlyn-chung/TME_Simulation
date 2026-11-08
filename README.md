An agent-based model of the tumor microenvironment, adapted the tumor compartment from [Zhang et. al.](https://aacrjournals.org/cancerres/article/84/16/2734/746873/Integration-of-Clinical-Trial-Spatial-Multiomics).

Consisting of five primary agents: 
* cancer cells,
* CD8+ T-cells,
* CD4+ T-cells,
* macrophages, and
* MDSCs,
these cell types also include additional differentiation states such as:
* CD8+: PD1Neg-Effector-Terminally Exhausted / Naive-Activated-Memory axis
* CD4+: Naive-Activate-Treg or Helper axis
Cancer cells can be stem, progenitor, and senescent, while macrophages are either M1 (pro-inflammatory) or M2 (anti-inflammatory). 

There are also cytokines that evolve with the ABM, including 
* Arg1,
* CCL2,
* IFNg,
* IL2,
* IL10,
* IL12,
* TGFb, and
* VEGFA.

This repo also contains scripts related to running a Sobol sensitivity analysis, which was used in factor prioritization of parameters. These parameters were later fed into an ABC-SMC simulation, a Bayesian inference method, to fine-tune them such that the values reflect actual tumor biology based on fractional counts from [Bassez et. al.](https://www.nature.com/articles/s41591-021-01323-8).
* grid_utils.py - Defines the Laplacian and diffusion update functions.
* model.py - Defines functions that initialize and updates cytokine fields and control agent recruitment to the grid.
* params.py - Defines time steps for both cytokines and agents, and parameter values that are used to scale agent and cytokine behavior. Values are adapter from literature or previous established by the original model.
* plot_simulations.py - Defines functions for plotting and visualizing .gifs of the tumor evolving over time, population and cytokine trajectories, and cell_counts.csv that tabulate population over time.
* scenarios.py - Contains scenarios and checks for custom scenarios.
* sensitivity_analysis.py - Global sensitivity analysis using the SALib package in Python to determine variance in population fractions using the Sobol method.
* runABM.py - Main runner.
* environment.yml - Contains all the environments necessary to run these Python scripts.

The script can be run from the terminal. You will be prompted to enter your scenario before the simulation starts.
