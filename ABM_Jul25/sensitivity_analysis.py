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

import ABM_Jul25.params as P

# PARAM_SPECS: 
#   - param_name  : must exactly match an attribute in params.py
#   - range_type  : sampling method
#                       - "cenetered" value * (1 -/+ pct_range)
#                       - "explicit" [low_value, high_value] sampling range
#                       - "shifted" shift_center * (1 -/+ pct_range) where shift_center = shift_amount + value
#   - pct_range   : bounds are set to value * (1 -/+ pct_range)
#   - lower_clip / upper_clip : hard biological limits (None = no clip);
#                   used for probabilities/fractions that must stay in [0, 1]
#   - is_integer  : round sampled values to the nearest int before use
#                   (division counts, Hill neighborhood counts, etc.)
# centered:     (param_name, "centered", pct_range, lo_clip, hi_clip, is_int)
# shifted:      (param_name, "shifted", shift_center, pct_range, lo_clip, hi_clip, is_int)
# explicit:     (param_name, "explicit", low_value, high_value, lo_clip, hi_clip, is_int)

# NOTE: diffusion coefficients, decay rates, molecular weights, and mass-based
# release rates are excluded — those are fixed by biophysics/literature, not
# fitted behavioral parameters.

PARAM_SPECS = [

    # --- Cancer cell kinetics ---
    ('CancerCell_stemGrowthRate',      'centered', 0.5,  0.0,  None, False),
    ('CancerCell_asymDivProb',         'centered', 0.15, 0.7,  1.0,  False),
    ('CancerCell_progGrowthRate',      'centered', 0.5,  0.0,  None, False),
    ('CancerCell_progDivMax',          'centered', 0.5,  2,    15,   True),
    ('CancerCell_senescentDeathRate',  'centered', 0.5,  0.0,  None, False),
    ('CancerCell_stemMoveProb',        'centered', 0.3,  0.0,  1.0,  False),

    # --- CD8 T cell kinetics ---
    ('cd8_commitment_delay',           'centered', 0.5,  0.0,  None, False),

    # current = 0.05, explore around 0.20
    ('p_memory_diff',                  'shifted',  0.20, 0.5,  0.0,  1.0, False),

    ('TCD8_prolif_IL2th',              'centered', 0.5,  0.0,  None, False),
    ('TCD8_div_Interval',              'centered', 0.5,  0.0,  None, False),
    ('TCD8_div_limit_effector',        'centered', 0.5,  2,    20,   True),
    ('TCD8_div_limit_memory',          'centered', 0.5,  1,    10,   True),

    # current = 1e-4, explore one log fold up
    ('k_TCR_activation',               'shifted',  1e-3, 1.0,  0.0,  None, False),

    # current = 1e-5, explore one log fold up
    ('k_cytokine_CD8',                 'shifted',  1e-4, 1.0,  0.0,  None, False),

    ('k_hill_MDSC_TCD8',               'centered', 0.5,  1,    6,    True),

    ('TCD8_kill_cooldown',             'centered', 0.5,  0.0,  None, False),

    # --- CD4 T cell kinetics ---
    ('cd4_commitment_delay',           'centered', 0.5,  0.0,  None, False),
    ('TCD4_Treg_frac',                 'centered', 0.5,  0.0,  1.0,  False),
    ('k_TCD4_div',                     'centered', 0.5,  0.0,  None, False),

    # current = 2.55e-7, explore one log fold up
    ('TCD4_k_Th_diff_Treg',            'shifted',  2.5e-6, 1.0, 0.0, None, False),

    # current = 1.9e11, explore one log fold up
    ('EC50_Arg1_Treg',                 'shifted',  2e12, 1.0, 0.0, None, False),

    ('TCD4_div_Interval',              'centered', 0.5,  0.0,  None, False),
    ('k_Arg1_Treg_div',                'centered', 0.5,  0.0,  None, False),

    # --- Macrophage polarization / phagocytosis ---
    ('Mac_moveProb',                   'centered', 0.5,  0.0,  1.0,  False),

    # current = 3.215e-7, explore one log fold up
    ('k_Mac_M1_pol',                   'shifted',  3e-6, 1.0, 0.0, None, False),

    # current = 2.894e-6, explore one log fold up
    ('k_Mac_M2_pol',                   'shifted',  3e-5, 1.0, 0.0, None, False),

    ('EC50_TGFb_M2',                   'shifted',  10 * P.EC50_TGFb_M2, 1.0, 0.0, None, False),
    ('EC50_IL10_M2',                   'shifted',  10 * P.EC50_IL10_M2, 1.0, 0.0, None, False),
    ('EC50_IFNg_M1',                   'shifted',  10 * P.EC50_IFNg_M1, 1.0, 0.0, None, False),
    ('EC50_IL12_M1',                   'shifted',  10 * P.EC50_IL12_M1, 1.0, 0.0, None, False),

    ('k_M1_phago',                     'centered', 0.5,  0.0,  None, False),

    ('IC50_IL10_phago',                'shifted',  10 * P.IC50_IL10_phago, 1.0, 0.0, None, False),

    # --- Recruitment rates ---
    ('k_TCD8_rec',                     'shifted',  2.5e-6, 1.0, 0.0, None, False),
    ('k_TCD4_rec',                     'shifted',  1.25e-6, 1.0, 0.0, None, False),
    ('k_MDSC_rec_base',                'shifted',  2.4e-7, 1.0, 0.0, None, False),
    ('k_MDSC_rec_max',                 'shifted',  1.4e-6, 1.0, 0.0, None, False),
    ('k_Mac_rec',                      'shifted',  2.5e-6, 1.0, 0.0, None, False),

    ('M1_M2_rec_ratio',                'centered', 0.3,  0.0,  1.0,  False),

    # --- PD-1 / PD-L1 axis ---
    ('PDL1_baseline_mean',             'centered', 0.5,  0.0,  1.0,  False),
    ('PDL1_IFNg_max',                  'centered', 0.3,  0.0,  1.0,  False),
    ('k_PDL1_upregulation',            'centered', 0.5,  0.0,  None, False),
    ('k_PD1_upregulation',             'centered', 0.5,  0.0,  None, False),

    # current = 2e-5
    ('k_PD1_signaling',                'shifted',  2e-4, 1.0, 0.0, None, False),

    # current = 1e-5
    ('k_exhaustion_rate',              'shifted',  1e-4, 1.0, 0.0, None, False),

    # current = 5e-7
    ('k_exhaustion_recovery',          'shifted',  5e-6, 1.0, 0.0, None, False),

    ('exhaustion_threshold_eff',       'centered', 0.3,  0.05, 0.95, False),
    ('exhaustion_threshold_term',      'centered', 0.3,  0.05, 0.95, False),
    ('how_exhausted',                  'centered', 1.0,  0.0,  1.0,  False),
]

def build_problem(param_specs=PARAM_SPECS):
    """
    Build the SALib 'problem' dict directly from the live values in params.py,
    so bounds always stay consistent with whatever is currently in params.py.
    Also returns a dict flagging which parameters should be rounded to int.
    """
    names, bounds, integer_flags = [], [], {}

    for spec in param_specs:

        name = spec[0]
        range_type = spec[1]

        if not hasattr(P, name):
            warnings.warn(
                f"params.py has no attribute '{name}' "
                f"— skipping parameter."
            )
            continue

        val = float(getattr(P, name))

        # --------------------------------------------------
        # Relative to current params.py value
        # --------------------------------------------------
        if range_type == "centered":

            pct_range, lo_clip, hi_clip, is_int = spec[2:]

            lo = val * (1 - pct_range)
            hi = val * (1 + pct_range)

            if val < 0:
                lo, hi = hi, lo

        # --------------------------------------------------
        # Relative to user-specified center
        # --------------------------------------------------
        elif range_type == "shifted":

            shift_center, pct_range, lo_clip, hi_clip, is_int = spec[2:]

            lo = shift_center * (1 - pct_range)
            hi = shift_center * (1 + pct_range)

            if shift_center < 0:
                lo, hi = hi, lo

        # --------------------------------------------------
        # User-specified bounds
        # --------------------------------------------------
        elif range_type == "explicit":

            lo, hi, lo_clip, hi_clip, is_int = spec[2:]

        else:
            raise ValueError(
                f"Unknown range_type '{range_type}' "
                f"for parameter '{name}'."
            )

        # --------------------------------------------------
        # Apply hard clips
        # --------------------------------------------------
        if lo_clip is not None:
            if (lo_clip == 0) and (lo < 0.1):
                lo = lo_clip # If the low value is close to low clip (0) snap to 0
            else:
                lo = max(lo, lo_clip) # Take the greater of the two values for lower bound
        if hi_clip is not None:
            if hi > hi_clip * 0.9:
                hi = hi_clip # If the value is close to high clip, snap to that value
            else:
                hi = min(hi, hi_clip) # Take the smaller of the two values for higher bound

        # --------------------------------------------------
        # Prevent degenerate ranges
        # --------------------------------------------------
        if lo >= hi:

            if val == 0:
                lo, hi = -1e-6, 1e-6

            else:
                lo, hi = sorted([
                    val * 0.9,
                    val * 1.1
                ])

        names.append(name)
        bounds.append([lo, hi])
        integer_flags[name] = is_int

    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }

    return problem, integer_flags

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

RAW_STAT_NAMES = [
    'tumor_cell_count',
    'cd8_exhausted_fraction',
    'cd8_active_fraction',
    'treg_fraction',
    'm1_m2_ratio',
    'mdsc_count',
    'tumor_immune_ratio',
    'pdl1_expression',
    'tumor_elimination_step',
    'tumor_extinct'
]

SUMMARY_STAT_NAMES = [

    # --- Tumor burden ---
    'tumor_cell_count_mean',
    'tumor_cell_count_sd',

    # --- CD8 states ---
    'cd8_exhausted_fraction_mean',
    'cd8_exhausted_fraction_sd',

    'cd8_active_fraction_mean',
    'cd8_active_fraction_sd',

    # --- CD4/Treg ---
    'treg_fraction_mean',
    'treg_fraction_sd',

    # --- Macrophages ---
    'm1_m2_ratio_mean',
    'm1_m2_ratio_sd',

    # --- MDSCs ---
    'mdsc_count_mean',
    'mdsc_count_sd',

    # --- Tumor : immune balance ---
    'tumor_immune_ratio_mean',
    'tumor_immune_ratio_sd',

    # --- PD-L1 ---
    'pdl1_expression_mean',
    'pdl1_expression_sd',
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

    # 2) Override params module in-place (rounding integer-valued params,
    #    e.g. division counts, Hill neighborhood counts)
    for key, val in param_set.items():
        if hasattr(P, key):
            if INTEGER_PARAMS.get(key, False):
                val = int(round(val))
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
                initial_MDSC=scenario_cfg['initial_MDSC']
            )

            tumor_elimination_step = np.nan
            for step in range(scenario_cfg['steps']):

                model.step()

                if model.count_agents(A.CancerCell) == 0:
                    tumor_elimination_step = step
                    break

                if not model.running:
                    break

            # 3) Compute summary stats from model histories + live agent counts
            stats = _compute_summary_stats(model)
            stats['tumor_elimination_step'] = tumor_elimination_step
            results_across_reps.append(stats)

    finally:
        # 4) Always restore original params — even if ABM crashes
        for key, val in original_vals.items():
            setattr(P, key, val)

    # 5) Average across replicates
    summary = {}

    for k in RAW_STAT_NAMES:

        vals = [r[k] for r in results_across_reps]

        summary[f"{k}_mean"] = np.nanmean(vals)
        summary[f"{k}_sd"] = np.nanstd(vals, ddof=1) # sample standard deviation

    return summary


def _compute_summary_stats(model) -> dict:
    # ----------------------------
    # Cancer cells
    # ----------------------------
    n_cancer = model.count_agents(A.CancerCell)

    # ----------------------------
    # Macrophages
    # ----------------------------
    n_m1 = model.count_cell_type(
        A.Macrophage,
        subtype=A.MacSubtype.M1
    )

    n_m2 = model.count_cell_type(
        A.Macrophage,
        subtype=A.MacSubtype.M2
    )

    # ----------------------------
    # CD8 states
    # ----------------------------
    n_cd8_naive = model.count_cell_type(
        A.CD8TCell,
        exhaustion_state=A.CD8ExhaustionState.CD8TPD1NEG
    )

    n_cd8_effector = model.count_cell_type(
        A.CD8TCell,
        exhaustion_state=A.CD8ExhaustionState.CD8TEFFECTOR
    )

    n_cd8_exhausted = model.count_cell_type(
        A.CD8TCell,
        exhaustion_state=A.CD8ExhaustionState.CD8TTERMINAL
    )

    # ----------------------------
    # CD4 states
    # ----------------------------
    n_cd4_helper = model.count_cell_type(
        A.CD4TCell,
        diff_state=A.CD4DiffState.CD4THELPER
    )

    n_cd4_treg = model.count_cell_type(
        A.CD4TCell,
        diff_state=A.CD4DiffState.CD4TREG
    )

    # Remaining CD4s assumed naïve/uncommitted
    n_cd4_total = model.count_agents(A.CD4TCell)
    n_cd4_naive = max(
        0,
        n_cd4_total - n_cd4_helper - n_cd4_treg
    )

    # ----------------------------
    # Total population
    # ----------------------------
    total_population = (
        n_cancer +
        n_m1 +
        n_m2 +
        n_cd8_naive +
        n_cd8_effector +
        n_cd8_exhausted +
        n_cd4_helper +
        n_cd4_treg +
        n_cd4_naive
    )

    if total_population == 0:
        return {
            'Cancer_cells':   0.0,
            'M1_like':        0.0,
            'M2_like':        0.0,
            'CD8_Effector':   0.0,
            'CD8_Exhausted':  0.0,
            'CD8_Naive':      0.0,
            'CD4_Helper':     0.0,
            'CD4_Treg':       0.0,
            'CD4_Naive':      0.0,
        }

    return {
        'Cancer_cells':  n_cancer / total_population,
        'M1_like':       n_m1 / total_population,
        'M2_like':       n_m2 / total_population,
        'CD8_Effector':  n_cd8_effector / total_population,
        'CD8_Exhausted': n_cd8_exhausted / total_population,
        'CD8_Naive':     n_cd8_naive / total_population,
        'CD4_Helper':    n_cd4_helper / total_population,
        'CD4_Treg':      n_cd4_treg / total_population,
        'CD4_Naive':     n_cd4_naive / total_population,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAMPLING AND EVALUATION
# ─────────────────────────────────────────────────────────────────────────────


problem, INTEGER_PARAMS = build_problem(param_specs=PARAM_SPECS)

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
    param_values = sobol_sample.sample(problem, N, calc_second_order=second_order_toggle, seed=seed)
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


def get_chunk_indices(n_samples: int, n_chunks: int, chunk_index: int) -> np.ndarray:
    """
    Split range(n_samples) into n_chunks contiguous, near-equal pieces and
    return the indices belonging to chunk_index (0-indexed).
    Used to shard the Sobol sample matrix across independent HPC array tasks.
    """
    if not (0 <= chunk_index < n_chunks):
        raise ValueError(f"chunk_index {chunk_index} out of range for n_chunks {n_chunks}")
    bounds = np.linspace(0, n_samples, n_chunks + 1).astype(int)
    return np.arange(bounds[chunk_index], bounds[chunk_index + 1])


def evaluate_chunk(param_values: np.ndarray,
                    n_chunks: int,
                    chunk_index: int,
                    n_jobs: int = None,
                    n_replicates: int = 3,
                    output_dir: str = 'sensitivity_results') -> str:
    """
    Evaluate only this chunk's slice of the full sample matrix and save the
    partial results to their own file. Designed to be called once per SLURM
    array task (task ID = chunk_index), so many chunks run concurrently
    across separate nodes — this is the actual HPC parallelization axis;
    n_jobs below only threads across cores *within* one chunk's node.

    Safe to re-run: if this chunk's file already exists and is fully
    populated (no NaNs), it is skipped so a resubmitted array job doesn't
    redo finished work.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_file = os.path.join(output_dir, f"chunk_{chunk_index:04d}_of_{n_chunks:04d}.npz")
    idx = get_chunk_indices(len(param_values), n_chunks, chunk_index)

    if os.path.exists(chunk_file):
        existing = np.load(chunk_file)
        if not np.isnan(existing['Y']).any():
            print(f"Chunk {chunk_index}/{n_chunks} already complete at {chunk_file} — skipping.")
            return chunk_file
        print(f"Chunk {chunk_index}/{n_chunks} found partially complete — resuming within chunk.")
        Y_chunk = existing['Y']
    else:
        Y_chunk = np.full((len(idx), len(SUMMARY_STAT_NAMES)), np.nan)

    n_jobs = n_jobs or max(1, cpu_count() - 1)
    remaining = [i for i, local_i in enumerate(range(len(idx))) if np.isnan(Y_chunk[local_i]).all()]
    args_list = [(int(idx[i]), param_values[idx[i]], n_replicates) for i in remaining]

    print(f"Chunk {chunk_index}/{n_chunks}: evaluating {len(args_list)} of {len(idx)} "
          f"assigned samples (global indices {idx[0]}..{idx[-1]}) with {n_jobs} workers...")

    def _save():
        np.savez(chunk_file, global_indices=idx, Y=Y_chunk)

    if n_jobs > 1 and len(args_list) > 1:
        with Pool(n_jobs) as pool:
            for batch_start in range(0, len(args_list), 50):
                batch = args_list[batch_start:batch_start + 50]
                results = pool.map(evaluate_one, batch)
                for global_idx, stats in results:
                    local_i = int(np.where(idx == global_idx)[0][0])
                    Y_chunk[local_i] = stats
                _save()
                print(f"  chunk {chunk_index}: {batch_start + len(batch)}/{len(args_list)} newly done")
    else:
        for i, (global_idx, stats) in enumerate(map(evaluate_one, args_list)):
            local_i = int(np.where(idx == global_idx)[0][0])
            Y_chunk[local_i] = stats
            if i % 25 == 0:
                _save()
                print(f"  chunk {chunk_index}: {i + 1}/{len(args_list)} newly done")

    _save()
    print(f"Chunk {chunk_index}/{n_chunks} complete -> {chunk_file}")
    return chunk_file


def merge_chunks(n_samples: int,
                  n_chunks: int,
                  output_dir: str = 'sensitivity_results',
                  checkpoint_file: str = None,
                  archive_merged: bool = True) -> np.ndarray:
    """
    Stitch all chunk_*.npz files in output_dir back into one (n_samples, n_stats)
    array, in the same format evaluate_samples()/the --skip-eval path expects.
    Raises with a clear message listing any missing/incomplete chunks rather
    than silently returning partial data. Saves merged files to a temp subdir to avoid clutter if archive_merged=True.
    """
    n_stats = len(SUMMARY_STAT_NAMES)
    Y = np.full((n_samples, n_stats), np.nan)

    missing, incomplete = [], []
    merged_files = []

    for chunk_index in range(n_chunks):
        chunk_file = os.path.join(output_dir, f"chunk_{chunk_index:04d}_of_{n_chunks:04d}.npz")
        if not os.path.exists(chunk_file):
            missing.append(chunk_index)
            continue
        data = np.load(chunk_file)
        gidx, Yc = data['global_indices'], data['Y']
        if np.isnan(Yc).any():
            incomplete.append(chunk_index)
        Y[gidx] = Yc
        merged_files.append(chunk_file)

    if missing:
        print(f"WARNING: {len(missing)} chunk file(s) not found: {missing}")
    if incomplete:
        print(f"WARNING: {len(incomplete)} chunk file(s) still have NaN rows "
              f"(unfinished evaluation): {incomplete}")
    n_done = np.sum(~np.isnan(Y).any(axis=1))
    print(f"Merged {n_done}/{n_samples} samples from {n_chunks} chunk file(s) in {output_dir}/")
    
    if checkpoint_file:
        np.save(checkpoint_file, Y)
        print(f"Saved merged results to {checkpoint_file}")

    if archive_merged and merged_files:
        archive_dir = os.path.join(output_dir, "merged_chunks_archive")
        os.makedirs(archive_dir, exist_ok=True)

        for f in merged_files:
            shutil.move(f, archive_dir)
        print(f"Moved {len(merged_files)} chunk files to {archive_dir} to avoid clutter.")

    return Y



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

        # SALib's Sobol estimator divides by Var(y); a statistic that came
        # back constant (e.g. an outcome that never varied across this
        # sample, such as a summary stat that stayed at 0 for every run)
        # produces Var=0 -> inf/NaN partway through analyze() and crashes
        # the whole job. Skip it with a clear message instead of losing
        # every other statistic's results on an HPC run.
        if np.any(~valid):
            print(f"Skipping {stat_name} — contains NaN values from failed ABM runs; "
                  f"cannot run Sobol analysis on an incomplete sample. Investigate "
                  f"the failed runs (see per-sample error messages during evaluation) "
                  f"before re-running this statistic.")
            continue
        if np.nanvar(y_col) == 0:
            print(f"Skipping {stat_name} — zero variance across all samples "
                  f"(value was constant: {y_col[0]:.4g}). No sensitivity to compute.")
            continue

        # SALib's Sobol estimator can still raise internally (not just via
        # the variance/NaN guards above) when a *subset* of the resampled
        # A/B/AB matrices happens to be constant — this is more likely with
        # small N or with sparse/skewed outcomes (e.g. a fraction that's
        # mostly 0). Isolate each statistic so one degenerate case doesn't
        # take down the whole (potentially multi-day) analysis run.
        try:
            Si = sobol_analyze.analyze(
                problem,
                y_col,
                calc_second_order=second_order_toggle,
                print_to_console=False
            )
        except Exception as e:
            print(f"Skipping {stat_name} — SALib Sobol analysis failed ({e}). "
                  f"This usually means the outcome was too degenerate/sparse at this "
                  f"sample size; consider a larger N or check for a modeling issue.")
            continue
        all_indices[stat_name] = Si

        # Save to CSV
        df = pd.DataFrame({
            'parameter': problem['names'],
            'S1':  Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST':  Si['ST'],
            'ST_conf': Si['ST_conf'],
        })
        df["relative_ST_conf"] = (
                df["ST_conf"] /
                np.maximum(df["ST"], 1e-12)
            )
        df["top_parameter"] = ((df["relative_ST_conf"] < 0.25) & (df["ST"] > 0.05))
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

def save_run_database(param_values, Y,
                      output_dir='sensitivity_results'):

    df = pd.DataFrame(
        param_values,
        columns=problem['names']
    )

    for i, stat in enumerate(SUMMARY_STAT_NAMES):
        df[stat] = Y[:, i]

    df.to_csv(
        f"{output_dir}/all_runs.csv",
        index=False
    )

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(
        description='Sobol sensitivity analysis for tumor ABM',
        epilog=(
            "HPC usage (recommended): run in three phases so the expensive "
            "evaluation phase can be sharded across a SLURM array —\n"
            "  1) --mode sample    (once)\n"
            "  2) --mode evaluate-chunk --n-chunks K --chunk-index $SLURM_ARRAY_TASK_ID  "
            "(K array tasks, run concurrently)\n"
            "  3) --mode merge     then   --mode analyze   (once, after all chunks finish)\n"
            "Single-machine usage: --mode full runs all phases sequentially on one node."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'sample', 'evaluate-chunk', 'merge', 'analyze'],
                        help="Pipeline phase to run. 'full' = original single-node behavior.")
    parser.add_argument('--N', type=int, default=64,
                        help='Sobol sample size (total runs = N*(2*params+2)). '
                             'Start with 64, increase to 256+ for final analysis.')
    parser.add_argument('--n-replicates', type=int, default=3,
                        help='ABM replicates per parameter set (handles stochasticity)')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Parallel workers WITHIN this task/node (default: n_cpu - 1)')
    parser.add_argument('--n-chunks', type=int, default=1,
                        help='Total number of shards to split the sample matrix into '
                             '(set this to your SLURM array size)')
    parser.add_argument('--chunk-index', type=int, default=0,
                        help='Which shard this task evaluates, 0-indexed '
                             '(pass $SLURM_ARRAY_TASK_ID here)')
    parser.add_argument('--output-dir', type=str, default='sensitivity_results',
                        help='Directory for outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-eval', action='store_true',
                        help="('full' mode only) Skip ABM evaluation entirely, load Y from "
                             "checkpoint and only re-run analysis/plots")
    parser.add_argument('--resume', action='store_true',
                        help="('full' mode only) Resume evaluation from an existing checkpoint "
                             "in --output-dir instead of wiping it and starting over.")
    args = parser.parse_args()

    second_order_toggle = False

    print(f"\nTumor ABM Sensitivity Analysis  [mode={args.mode}]")
    print(f"  Parameters: {problem['num_vars']}")
    print(f"  Summary statistics: {len(SUMMARY_STAT_NAMES)}")
    if second_order_toggle is False:
        n_total_samples_formula = args.N * (problem['num_vars'] + 2)
    else:
        n_total_samples_formula = args.N * (2 * problem['num_vars'] + 2)
    print(f"  N = {args.N}  →  {n_total_samples_formula} parameter sets "
          f"× {args.n_replicates} replicates = "
          f"{n_total_samples_formula * args.n_replicates} total ABM runs\n")

    samples_file = os.path.join(args.output_dir, f"param_values_N{args.N}.npy")
    checkpoint = os.path.join(args.output_dir, f"checkpoint_N{args.N}.npy")

    # ---- Phase: sample ----
    # Generates the Saltelli sample matrix once and saves it to disk so every
    # downstream array task (and the merge/analyze phase) reads the exact
    # same samples. Run this as a single, short, non-array job.
    if args.mode == 'sample':
        os.makedirs(args.output_dir, exist_ok=True)
        param_values = generate_samples(N=args.N, seed=args.seed)
        np.save(samples_file, param_values)
        n = len(param_values)
        print(f"Saved {n} parameter sets to {samples_file}")
        print(f"Suggested --n-chunks for an array job: something that divides "
              f"cleanly into ~{max(1, n // 20)}-{max(1, n // 5)} samples per chunk "
              f"depending on your per-run walltime budget.")

    # ---- Phase: evaluate-chunk ----
    # One call per SLURM array task. Each task independently evaluates only
    # its shard of the sample matrix and writes its own chunk file — this is
    # the actual cross-node parallelization axis. --n-jobs only threads
    # across cores within this single task/node.
    elif args.mode == 'evaluate-chunk':
        if not os.path.exists(samples_file):
            raise FileNotFoundError(
                f"{samples_file} not found — run --mode sample first (once) "
                f"before submitting the evaluate-chunk array job."
            )
        param_values = np.load(samples_file)
        evaluate_chunk(param_values,
                       n_chunks=args.n_chunks,
                       chunk_index=args.chunk_index,
                       n_jobs=args.n_jobs,
                       n_replicates=args.n_replicates,
                       output_dir=args.output_dir)

    # ---- Phase: merge ----
    # Run once after all array tasks finish. Stitches chunk_*.npz files back
    # into a single checkpoint_N{N}.npy in the same format --mode full uses,
    # so --mode analyze (or --skip-eval under --mode full) can read it.
    elif args.mode == 'merge':
        if not os.path.exists(samples_file):
            raise FileNotFoundError(f"{samples_file} not found — run --mode sample first.")
        param_values = np.load(samples_file)
        merge_chunks(n_samples=len(param_values),
                    n_chunks=args.n_chunks,
                    output_dir=args.output_dir,
                    checkpoint_file=checkpoint)

    # ---- Phase: analyze ----
    # Run once, after merge. Loads the merged checkpoint and produces the
    # Sobol indices, heatmaps, rankings, and identifiability summary.
    elif args.mode == 'analyze':
        if not (os.path.exists(samples_file) and os.path.exists(checkpoint)):
            raise FileNotFoundError(
                f"Need both {samples_file} and {checkpoint} — run --mode sample, "
                f"then evaluate-chunk for all shards, then --mode merge, before --mode analyze."
            )
        param_values = np.load(samples_file)
        Y = np.load(checkpoint)
        all_indices = run_sobol_analysis(param_values, Y, output_dir=args.output_dir)
        plot_sensitivity_heatmap(all_indices, output_dir=args.output_dir, index_type='ST')
        plot_sensitivity_heatmap(all_indices, output_dir=args.output_dir, index_type='S1')
        plot_ranking_per_stat(all_indices, output_dir=args.output_dir, top_n=10)
        summarize_identifiability(all_indices, output_dir=args.output_dir)
        save_run_database(param_values, Y, output_dir=args.output_dir)
        print(f"\nAll outputs written to: {args.output_dir}/")

    # ---- Phase: full ----
    # Original single-node behavior: sample -> evaluate (via internal
    # multiprocessing.Pool) -> analyze -> plot, all in one process. Fine for
    # small N / a quick local check; use the sharded modes above for the
    # real HPC run.
    else:
        param_values = generate_samples(N=args.N, seed=args.seed)

        # Only wipe the output dir on a genuinely fresh run — --resume or
        # --skip-eval need the existing checkpoint to survive so a preempted
        # HPC job can pick back up instead of losing all completed work.
        if os.path.exists(args.output_dir) and not (args.resume or args.skip_eval):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

        if args.skip_eval and os.path.exists(checkpoint):
            print(f"Loading evaluations from {checkpoint}")
            Y = np.load(checkpoint)
        else:
            Y = evaluate_samples(param_values,
                                 n_jobs=args.n_jobs,
                                 n_replicates=args.n_replicates,
                                 checkpoint_file=checkpoint)

        all_indices = run_sobol_analysis(param_values, Y, output_dir=args.output_dir)
        plot_sensitivity_heatmap(all_indices, output_dir=args.output_dir, index_type='ST')
        plot_sensitivity_heatmap(all_indices, output_dir=args.output_dir, index_type='S1')
        plot_ranking_per_stat(all_indices, output_dir=args.output_dir, top_n=10)
        summarize_identifiability(all_indices, output_dir=args.output_dir)
        save_run_database(param_values, Y, output_dir=args.output_dir)
        print(f"\nAll outputs written to: {args.output_dir}/")