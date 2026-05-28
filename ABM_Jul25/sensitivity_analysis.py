"""
sensitivity_analysis.py

Sobol sensitivity analysis for the tumor ABM using SALib.
Designed to work with your existing params.py and ABM runner.

USAGE:
    1. Implement run_abm() to call your ABM and return summary statistics
    2. Adjust N (sample size) based on your compute budget
    3. Run: python sensitivity_analysis.py

STRUCTURE:
    - Parameter bounds are defined as ±X% around your current values
    - Summary statistics are the model outputs you care about
    - Sobol indices tell you which parameters drive output variance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
from multiprocessing import Pool, cpu_count
import json
import os
import warnings
import shutil

# ─────────────────────────────────────────────────────────────────────────────
# 1. PARAMETER PROBLEM DEFINITION
#    Only include parameters that are (a) behavioral/kinetic and
#    (b) not fixed by biophysics (diffusion, decay, MW, release rates are fixed)
#
#    Bounds are set as [low, high] — adjust based on biological plausibility.
#    A common starting point is ±50% around your current value, but tighten
#    where you have strong prior knowledge.
# ─────────────────────────────────────────────────────────────────────────────

parameter_names = [
        # --- Cancer cell kinetics ---
        'CancerCell_stemGrowthRate',      # stem cell growth rate
        # 'CancerCell_asymDivProb',          # asymmetric division probability
        # 'CancerCell_progGrowthRate',       # progenitor growth rate
        # 'CancerCell_progDivMax',           # max progenitor divisions
        # 'CancerCell_senescentDeathRate',   # senescent death rate
        # 'CancerCell_stemMoveProb',         # stem migration probability
        # 'CancerCell_MoveProb',             # non-stem migration probability

        # # --- CD8 T cell kinetics ---
        # 'k_TCD8_killing',                  # killing rate
        # 'TCD8_div_Limit',                  # division limit
        # 'k_hill_MDSC_TCD8',               # MDSC inhibition half-max
        # 'IC50_Arg1_TCD8',                  # Arg1 inhibition of CD8
        # 'IC50_NO_TCD8',                    # NO inhibition of CD8

        # # --- CD4 T cell kinetics ---
        # 'TCD4_Treg_frac',                  # initial Treg fraction
        # 'TCD4_k_Th_diff_Treg',            # Th -> Treg differentiation rate
        # 'k_TCD4_div',                      # CD4 division rate
        # 'EC50_Arg1_Treg',                  # Arg1 effect on Treg division
        # 'k_Arg1_Treg_div',                 # Arg1-driven Treg division rate

        # # --- Macrophage polarization ---
        # 'k_Mac_M1_pol',                    # M2 -> M1 polarization rate
        # 'k_Mac_M2_pol',                    # M1 -> M2 polarization rate
        # 'EC50_TGFb_M2',                    # TGFb threshold for M2 polarization
        # 'EC50_IL10_M2',                    # IL10 threshold for M2 polarization
        # 'EC50_IFNg_M1',                    # IFNg threshold for M1 polarization
        # 'EC50_IL12_M1',                    # IL12 threshold for M1 polarization
        # 'k_M1_phago',                      # M1 phagocytosis rate

        # # --- Recruitment rates ---
        # 'k_TCD8_rec',                      # CD8 recruitment
        # 'k_TCD4_rec',                      # CD4 recruitment
        # 'k_MDSC_rec_base',                 # baseline MDSC recruitment
        # 'k_MDSC_rec_max',                  # max MDSC recruitment
        # 'k_Mac_rec',                       # macrophage recruitment

        # # --- PD-1 / PD-L1 axis ---
        # 'PDL1_baseline_mean',              # baseline PD-L1 expression
        # 'PDL1_IFNg_max',                   # max IFNg-driven PD-L1
        # 'k_PDL1_upregulation',             # rate of PD-L1 upregulation
        # 'k_exhaustion_rate',               # CD8 exhaustion accumulation rate
        # 'k_exhaustion_recovery',           # CD8 exhaustion recovery rate
        # 'exhaustion_threshold_term',       # terminal exhaustion threshold
    ]

parameter_bounds = [
        # Cancer cell kinetics — ±50% unless biologically constrained
        [1.45e-6,  4.35e-6],   # stemGrowthRate
        # [0.85,     0.99],      # asymDivProb (constrained: must stay high)
        # [2.9e-6,   8.7e-6],    # progGrowthRate
        # [5,        15],        # progDivMax (integer-valued, treat as continuous)
        # [6e-6,     1.8e-5],    # senescentDeathRate
        # [0.4,      1.0],       # stemMoveProb
        # [0.5,      1.0],       # MoveProb (constrained: already at max)

        # # CD8 kinetics
        # [5.8e-6,   1.74e-5],   # k_TCD8_killing
        # [2,        6],         # div_Limit
        # [1,        4],         # k_hill_MDSC (neighborhood count)
        # [2.65e+11, 7.95e+11],  # IC50_Arg1_TCD8
        # [3.75e-10*1.25e-10*6.022e23*0.5,
        #  3.75e-10*1.25e-10*6.022e23*1.5], # IC50_NO_TCD8

        # # CD4 kinetics
        # [0.05,     0.40],      # Treg_frac
        # [1.275e-7, 3.825e-7],  # Th->Treg diff rate
        # [5.8e-6,   1.74e-5],   # CD4 div rate
        # [9.5e+10,  2.85e+11],  # EC50_Arg1_Treg
        # [1.215e-5, 3.645e-5],  # k_Arg1_Treg_div

        # # Macrophage polarization
        # [1.6075e-7, 4.8225e-7], # k_Mac_M1_pol
        # [1.447e-6,  4.341e-6],  # k_Mac_M2_pol
        # [EC50_TGFb_M2*0.5, EC50_TGFb_M2*1.5] if False else  # placeholder
        # [8.75e-23,  2.625e-22], # EC50_TGFb_M2 (approximate)
        # [5e-24,     1.5e-23],   # EC50_IL10_M2
        # [1.8125e-22,5.4375e-22],# EC50_IFNg_M1
        # [8.75e-23,  2.625e-22], # EC50_IL12_M1
        # [2.8935e-6, 8.6805e-6], # k_M1_phago

        # # Recruitment
        # [1.25e-7,  3.75e-7],   # k_TCD8_rec
        # [6.25e-8,  1.875e-7],  # k_TCD4_rec
        # [1.215e-8, 3.645e-8],  # k_MDSC_rec_base
        # [7e-8,     2.1e-7],    # k_MDSC_rec_max
        # [1.23e-7,  3.69e-7],   # k_Mac_rec

        # # PD-1/PD-L1 
        # [0.05,     0.30],      # PDL1_baseline_mean
        # [0.50,     0.95],      # PDL1_IFNg_max
        # [5e-6,     1.5e-5],    # k_PDL1_upregulation
        # [5e-6,     1.5e-5],    # k_exhaustion_rate
        # [2.5e-7,   7.5e-7],    # k_exhaustion_recovery
        # [0.50,     0.90],      # exhaustion_threshold_term
    ]

problem = {
    'num_vars': len(parameter_names),
    'names': parameter_names,
    'bounds': parameter_bounds
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. SUMMARY STATISTICS
#    These are what your ABM outputs that you'll compare to real data.
#    Each statistic becomes a separate sensitivity analysis.
#
#    Suggested statistics based on your model and likely scRNA-seq data:
#      - Cell type proportions at final timepoint
#      - CD8 exhaustion fraction
#      - M1/M2 macrophage ratio
#      - Tumor cell count (growth)
#      - Treg fraction among CD4
#
#    Add or remove based on what your data can actually measure.
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_STAT_NAMES = [
    'tumor_cell_count',         # total tumor cells at T2
    'cd8_exhausted_fraction',   # fraction of CD8s that are terminally exhausted
    'cd8_active_fraction',      # fraction of CD8s that are active/non-exhausted
    'treg_fraction',            # Tregs as fraction of total CD4
    'm1_m2_ratio',              # M1 / (M1 + M2) macrophage ratio
    'mdsc_count',               # total MDSC count
    'tumor_immune_ratio',       # tumor cells / total immune cells (tumor burden proxy)
    'pdl1_mean_expression',     # mean PD-L1 on tumor cells
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. ABM RUNNER — IMPLEMENT THIS
#    This function receives one parameter set (as a dict) and returns
#    a dict of summary statistics.
#
#    It should:
#      1. Override your params.py values with the provided param_set
#      2. Run the ABM for the appropriate duration (your T1->T2 interval)
#      3. Return summary statistics as a dict
#
#    Run multiple replicates per parameter set and average them to
#    handle ABM stochasticity (recommend 3-5 replicates minimum).
# ─────────────────────────────────────────────────────────────────────────────

import importlib
import ABM_Jul25.params as P
from ABM_Jul25.plot_simulation import run_simulation_verbose
from ABM_Jul25.model import ABM_Model
from ABM_Jul25.scenarios import DEFAULT_SCENARIOS
import ABM_Jul25.agents as A
import random, numpy as np

def run_abm(param_set: dict, n_replicates: int = 3, seed_offset: int = 0) -> dict:
    """
    Run the ABM with a given parameter set and return summary statistics.
    Temporarily overrides P (params module) attributes, then restores them.
    """

    # 1) Save original param values so we can restore them after
    original_vals = {key: getattr(P, key) for key in param_set if hasattr(P, key)}

    # 2) Override params module in-place
    for key, val in param_set.items():
        if hasattr(P, key):
            setattr(P, key, val)

    results_across_reps = []

    try:
        for rep in range(n_replicates):
            np.random.seed(42 + seed_offset + rep)
            random.seed(42 + seed_offset + rep)

            # Pull scenario config
            scenario_cfg = DEFAULT_SCENARIOS['standard']

            # Instantiate and run model directly
            model = ABM_Model(
                width=scenario_cfg['width'],
                height=scenario_cfg['height'],
                initial_tumor_cells=scenario_cfg['initial_tumor_cells'],
                initial_CD8Tcells=scenario_cfg['initial_CD8Tcells'],
                initial_CD4Tcells=scenario_cfg['initial_CD4Tcells'],
                initial_macrophages=scenario_cfg['initial_macrophages'],
                initial_MDSC=scenario_cfg['initial_MDSC'],
                pdl1_pd1_axis=True
            )

            for _ in range(scenario_cfg['steps']):
                if not model.running:
                    break
                model.step()

            # 3) Compute summary stats from model histories + live agent counts
            stats = _compute_summary_stats(model)
            results_across_reps.append(stats)

    finally:
        # 4) Always restore original params — even if ABM crashes
        for key, val in original_vals.items():
            setattr(P, key, val)

    # 5) Average across replicates
    summary = {
        k: np.mean([r[k] for r in results_across_reps])
        for k in SUMMARY_STAT_NAMES
    }
    return summary


def _compute_summary_stats(model) -> dict:
    """
    Extract summary statistics from a completed ABM_Model instance.
    Uses model.cd8_state_counts, mean_pdl1_tumor, and live agent counts.
    """
    # --- Tumor cell count (final step) ---
    tumor_count = model.count_agents(A.CancerCell)

    # --- CD8 exhaustion (final step from tracked histories) ---
    if model.cd8_state_counts["effector"] and model.cd8_state_counts["terminal"]:
        n_effector  = model.cd8_state_counts["effector"][-1]
        n_prex      = model.cd8_state_counts["prexhausted"][-1]
        n_terminal  = model.cd8_state_counts["terminal"][-1]
        n_cd8_total = n_effector + n_prex + n_terminal
        cd8_exhausted_frac = n_terminal  / n_cd8_total if n_cd8_total > 0 else 0.0
        cd8_active_frac    = n_effector  / n_cd8_total if n_cd8_total > 0 else 0.0
    else:
        cd8_exhausted_frac = 0.0
        cd8_active_frac    = 0.0

    # --- Treg fraction among CD4 (live agents) ---
    n_treg   = model.count_cell_type(A.CD4TCell, subtype=A.CD4TSubtype.CD4TREG)
    n_thelper= model.count_cell_type(A.CD4TCell, subtype=A.CD4TSubtype.CD4THELPER)
    n_cd4_total = n_treg + n_thelper
    treg_frac = n_treg / n_cd4_total if n_cd4_total > 0 else 0.0

    # --- M1/M2 ratio ---
    n_m1 = model.count_cell_type(A.Macrophage, subtype=A.MacSubtype.M1)
    n_m2 = model.count_cell_type(A.Macrophage, subtype=A.MacSubtype.M2)
    m1_m2_ratio = n_m1 / (n_m1 + n_m2) if (n_m1 + n_m2) > 0 else 0.0

    # --- MDSC count ---
    mdsc_count = model.count_agents(A.MDSC)

    # --- Tumor/immune ratio ---
    n_immune_total = (model.count_agents(A.CD8TCell) + n_cd4_total +
                      n_m1 + n_m2 + mdsc_count)
    tumor_immune_ratio = tumor_count / n_immune_total if n_immune_total > 0 else np.nan

    # --- Mean PD-L1 on tumor cells (tracked history) ---
    pdl1_mean = model.mean_pdl1_tumor[-1] if model.mean_pdl1_tumor else 0.0

    return {
        'tumor_cell_count':       float(tumor_count),
        'cd8_exhausted_fraction': cd8_exhausted_frac,
        'cd8_active_fraction':    cd8_active_frac,
        'treg_fraction':          treg_frac,
        'm1_m2_ratio':            m1_m2_ratio,
        'mdsc_count':             float(mdsc_count),
        'tumor_immune_ratio':     tumor_immune_ratio,
        'pdl1_mean_expression':   pdl1_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAMPLING AND EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_samples(N: int = 64, seed: int = 42) -> np.ndarray:
    """
    Generate Sobol sample matrix.

    N controls total evaluations: total runs = N * (2 * num_vars + 2)
    With 35 parameters and N=256: 256 * 72 = 18,432 ABM runs.

    For expensive ABMs, start with N=64 (4,608 runs) to check sensitivity
    structure before committing to full N=256 or N=1024.

    Recommended N by ABM speed:
        < 1 min/run:  N=256 or N=512
        1-5 min/run:  N=64, use surrogate/emulator afterward
        > 5 min/run:  N=32, strongly consider surrogate model
    """
    param_values = sobol_sample.sample(problem, N, calc_second_order=True, seed=seed)
    print(f"Generated {len(param_values)} parameter sets "
          f"(N={N}, {problem['num_vars']} parameters)")
    return param_values


def evaluate_one(args):
    """
    Module-level worker function for multiprocessing.
    Must be at module level (not nested) so it can be pickled by Pool.map.
    """
    idx, params_row, n_replicates = args
    param_set = dict(zip(problem['names'], params_row))
    try:
        result = run_abm(param_set, n_replicates=n_replicates, seed_offset=idx * 100)
        return idx, [result[name] for name in SUMMARY_STAT_NAMES]
    except Exception as e:
        print(f"Warning: ABM failed at sample {idx}: {e}")
        return idx, [np.nan] * len(SUMMARY_STAT_NAMES)


def evaluate_samples(param_values: np.ndarray,
                     n_jobs: int = None,
                     n_replicates: int = 3,
                     checkpoint_file: str = 'sensitivity_checkpoint.npy') -> np.ndarray:
    """
    Evaluate the ABM across all sampled parameter sets.

    Supports checkpointing so long runs can be resumed if interrupted.
    Returns array of shape (n_samples, n_summary_stats).
    """
    n_samples = len(param_values)
    n_stats = len(SUMMARY_STAT_NAMES)

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        Y = np.load(checkpoint_file)
        start_idx = np.where(np.all(np.isnan(Y), axis=1))[0]
        start_idx = start_idx[0] if len(start_idx) > 0 else n_samples
        print(f"Resuming from checkpoint at sample {start_idx}/{n_samples}")
    else:
        Y = np.full((n_samples, n_stats), np.nan)
        start_idx = 0

    # Parallel evaluation
    n_jobs = n_jobs or max(1, cpu_count() - 1)
    args_list = [(i, param_values[i], n_replicates) for i in range(start_idx, n_samples)]

    print(f"Evaluating {len(args_list)} samples with {n_jobs} workers...")

    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            for batch_start in range(0, len(args_list), 50):
                batch = args_list[batch_start:batch_start+50]
                results = pool.map(evaluate_one, batch)
                for idx, stats in results:
                    Y[idx] = stats
                np.save(checkpoint_file, Y)
                print(f"  Checkpoint saved: {batch_start + len(batch)}/{n_samples} done")
    else:
        for i, (idx, stats) in enumerate(map(evaluate_one, args_list)):
            Y[idx] = stats
            if i % 50 == 0:
                np.save(checkpoint_file, Y)
                print(f"  {i+1}/{len(args_list)} done")

    return Y


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALYSIS AND VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def run_sobol_analysis(param_values: np.ndarray,
                       Y: np.ndarray,
                       output_dir: str = 'sensitivity_results') -> dict:
    """
    Compute Sobol indices for each summary statistic and save results.
    Returns dict of {stat_name: sobol_indices}.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    all_indices = {}

    for i, stat_name in enumerate(SUMMARY_STAT_NAMES):
        y_col = Y[:, i]

        # Skip if all NaN
        if np.all(np.isnan(y_col)):
            print(f"Skipping {stat_name} — all NaN")
            continue

        # Drop NaN rows for this statistic
        valid = ~np.isnan(y_col)
        if valid.sum() < len(y_col) * 0.9:
            print(f"Warning: {stat_name} has {(~valid).sum()} NaN runs "
                  f"({100*(~valid).mean():.1f}%) — check ABM stability")

        Si = sobol_analyze.analyze(
            problem,
            y_col,
            calc_second_order=True,
            print_to_console=False
        )
        all_indices[stat_name] = Si

        # Save to CSV
        df = pd.DataFrame({
            'parameter': problem['names'],
            'S1':  Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST':  Si['ST'],
            'ST_conf': Si['ST_conf'],
        })
        df['identifiable'] = (df['S1'] > 0.05) & (df['ST'] > 0.05)
        df.to_csv(f"{output_dir}/{stat_name}_sobol.csv", index=False)
        print(f"\n{stat_name} — top 5 parameters by total-order index:")
        print(df.nlargest(5, 'ST')[['parameter', 'S1', 'ST']].to_string(index=False))

    return all_indices


def plot_sensitivity_heatmap(all_indices: dict,
                             output_dir: str = 'sensitivity_results',
                             index_type: str = 'ST'):
    """
    Heatmap of Sobol total-order indices across all summary statistics.
    Rows = parameters, Columns = summary statistics.
    This is the key diagnostic plot for identifiability.
    """
    n_params = problem['num_vars']
    n_stats  = len(SUMMARY_STAT_NAMES)
    stats_with_data = [s for s in SUMMARY_STAT_NAMES if s in all_indices]

    matrix = np.zeros((n_params, len(stats_with_data)))
    for j, stat in enumerate(stats_with_data):
        matrix[:, j] = all_indices[stat][index_type]

    fig, ax = plt.subplots(figsize=(max(10, len(stats_with_data)*1.5),
                                    max(12, n_params*0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=f'Sobol {index_type} index')

    ax.set_xticks(range(len(stats_with_data)))
    ax.set_xticklabels(stats_with_data, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_params))
    ax.set_yticklabels(problem['names'], fontsize=8)
    ax.set_title(f'Sobol {index_type} Sensitivity Indices\n'
                 f'(Rows = parameters, Columns = summary statistics)',
                 fontsize=11, pad=15)

    # Annotate cells above threshold
    for i in range(n_params):
        for j in range(len(stats_with_data)):
            val = matrix[i, j]
            if val > 0.1:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color='black' if val < 0.6 else 'white')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_heatmap_{index_type}.pdf",
                bbox_inches='tight', dpi=150)
    plt.savefig(f"{output_dir}/sensitivity_heatmap_{index_type}.png",
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nHeatmap saved to {output_dir}/sensitivity_heatmap_{index_type}.pdf")


def plot_ranking_per_stat(all_indices: dict,
                          output_dir: str = 'sensitivity_results',
                          top_n: int = 10):
    """
    Bar chart of top-N most influential parameters for each summary statistic.
    """
    stats_with_data = [s for s in SUMMARY_STAT_NAMES if s in all_indices]
    n_stats = len(stats_with_data)
    cols = min(3, n_stats)
    rows = (n_stats + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5, rows * 4),
                             squeeze=False)
    fig.suptitle('Top Parameters by Total-Order Sobol Index', fontsize=13, y=1.01)

    for idx, stat in enumerate(stats_with_data):
        ax = axes[idx // cols][idx % cols]
        Si = all_indices[stat]
        df = pd.DataFrame({'param': problem['names'],
                           'ST': Si['ST'],
                           'ST_conf': Si['ST_conf']})
        df = df.nlargest(top_n, 'ST')

        bars = ax.barh(df['param'], df['ST'],
                       xerr=df['ST_conf'], color='steelblue',
                       error_kw={'linewidth': 0.8, 'capsize': 2})
        ax.axvline(0.05, color='red', linestyle='--', linewidth=0.8,
                   label='0.05 threshold')
        ax.set_xlabel('Total-order Sobol index (ST)')
        ax.set_title(stat.replace('_', ' '), fontsize=9)
        ax.invert_yaxis()
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide empty subplots
    for idx in range(len(stats_with_data), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_rankings.pdf",
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Rankings saved to {output_dir}/parameter_rankings.pdf")


def summarize_identifiability(all_indices: dict,
                              output_dir: str = 'sensitivity_results',
                              ST_threshold: float = 0.05):
    """
    Produce a summary table of which parameters are influential on
    at least one output (identifiable candidates) vs. non-influential
    (candidates for fixing at nominal values during fitting).
    """
    records = []
    for param in problem['names']:
        max_ST = 0.0
        influential_outputs = []
        for stat, Si in all_indices.items():
            param_idx = problem['names'].index(param)
            st = Si['ST'][param_idx]
            if st > ST_threshold:
                influential_outputs.append(stat)
            max_ST = max(max_ST, st)
        records.append({
            'parameter': param,
            'max_ST': round(max_ST, 4),
            'n_outputs_influenced': len(influential_outputs),
            'influential_outputs': ', '.join(influential_outputs),
            'recommendation': 'FIT' if max_ST > ST_threshold else 'FIX at nominal'
        })

    df = pd.DataFrame(records).sort_values('max_ST', ascending=False)
    df.to_csv(f"{output_dir}/identifiability_summary.csv", index=False)

    print("\n" + "="*60)
    print("IDENTIFIABILITY SUMMARY")
    print("="*60)
    fit_params = df[df['recommendation'] == 'FIT']
    fix_params = df[df['recommendation'] == 'FIX at nominal']
    print(f"\nParameters to FIT ({len(fit_params)}):")
    print(fit_params[['parameter', 'max_ST', 'n_outputs_influenced']].to_string(index=False))
    print(f"\nParameters to FIX ({len(fix_params)}):")
    print(fix_params[['parameter', 'max_ST']].to_string(index=False))
    print(f"\nFull summary saved to {output_dir}/identifiability_summary.csv")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description='Sobol sensitivity analysis for tumor ABM')
    parser.add_argument('--N', type=int, default=64,
                        help='Sobol sample size (total runs = N*(2*params+2)). '
                             'Start with 64, increase to 256+ for final analysis.')
    parser.add_argument('--n-replicates', type=int, default=3,
                        help='ABM replicates per parameter set (handles stochasticity)')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Parallel workers (default: n_cpu - 1)')
    parser.add_argument('--output-dir', type=str, default='sensitivity_results',
                        help='Directory for outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip ABM evaluation, load from checkpoint and re-analyze')
    args = parser.parse_args()

    print(f"\nTumor ABM Sensitivity Analysis")
    print(f"  Parameters: {problem['num_vars']}")
    print(f"  Summary statistics: {len(SUMMARY_STAT_NAMES)}")
    print(f"  N = {args.N}  →  {args.N * (2*problem['num_vars'] + 2)} ABM runs "
          f"× {args.n_replicates} replicates = "
          f"{args.N * (2*problem['num_vars'] + 2) * args.n_replicates} total simulations\n")

    # Step 1: Sample
    param_values = generate_samples(N=args.N, seed=args.seed)

    # Step 1.5: Create the output directory, checking that it exists
    if os.path.exists(args.output_dir):   # Will hold onto cache of previously run simulations if we don't clear what already exists
        shutil.rmtree(args.output_dir)
                                                  
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 2: Evaluate
    checkpoint = f"{args.output_dir}/checkpoint_N{args.N}.npy"
    if args.skip_eval and os.path.exists(checkpoint):
        print(f"Loading evaluations from {checkpoint}")
        Y = np.load(checkpoint)
    else:
        Y = evaluate_samples(param_values,
                             n_jobs=args.n_jobs,
                             n_replicates=args.n_replicates,
                             checkpoint_file=checkpoint)

    # Step 3: Sobol analysis
    all_indices = run_sobol_analysis(param_values, Y, output_dir=args.output_dir)

    # Step 4: Visualize
    plot_sensitivity_heatmap(all_indices, output_dir=args.output_dir, index_type='ST')
    plot_sensitivity_heatmap(all_indices, output_dir=args.output_dir, index_type='S1')
    plot_ranking_per_stat(all_indices, output_dir=args.output_dir, top_n=10)

    # Step 5: Identifiability summary
    summary_df = summarize_identifiability(all_indices, output_dir=args.output_dir)

    print(f"\nAll outputs written to: {args.output_dir}/")