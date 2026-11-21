"""
Simple ABC-SMC calibration of the TME ABM with pyABC, one patient per run.

    python abc_smc_tme.py 0          # fit patient in row 0 of observed.csv
    python abc_smc_tme.py            # fit all patients one after another

observed.csv: columns Patient_ID + the nine fraction columns in KEYS below.
abm_adapter.py must provide run_model(parameters) -> dict with the nine KEYS.
"""
import os
import sys
import random
import multiprocessing as mp
from multiprocessing import Pool
import time
# import cProfile
# import pstats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabc
from pyabc.visualization import (plot_epsilons, plot_effective_sample_sizes,
                                 plot_acceptance_rates_trajectory, plot_kde_matrix_highlevel)

import random
import numpy as np

import ABM_Jul25.params as P
import ABM_Jul25.agents as A
from ABM_Jul25.model import ABM_Model
from ABM_Jul25.scenarios import DEFAULT_SCENARIOS

# ---------------------------------------------------------------- settings
DATA = "/athena/cayuga_0014/scratch/kch4018/abm-files/ABM_Validation/Bassez_counts/observed_fractions.csv"
OUT = "/athena/cayuga_0014/scratch/kch4018/abm-files/ABM_Validation/SCM_results_exhaustion"   # new dir: prior changed, don't resume SCM_results/*/abc.db
PRIOR_CHECK_ONLY = os.environ.get("PRIOR_CHECK_ONLY") == "1"   # 1 -> run only the prior predictive check, skip ABC
POP_SIZE = 500                # particles per generation
MAX_GENERATIONS = 8
# MIN_EPSILON = 0.02
ESS_THRESHOLD = POP_SIZE / 2  # resample if effective sample size falls below this
MIN_ACCEPTANCE = 0.01         # stop if acceptance rate falls below this
MAX_SIMS = 50000              # hard cap on ABM runs per patient
N_PPC = 256                    # fresh simulations for the posterior predictive check
N_PRIOR_PC = 200               # simulations from the prior, to flag unreachable observed values before ABC
N_PROCS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

KEYS = ["Cancer_cells", "M1_like", "M2_like", "CD8_Effector", "CD8_Exhausted",
        "CD8_Naive", "CD4_Helper", "CD4_Treg", "CD4_Naive"]

# ---------------------------------------------------------------- priors (all uniform)
# Rate constants are sampled as log10(value) because they span decades.
# Initial-count bounds sum to 850 so they fit the 30x30 grid.

# Using 13 parameters from the sensitivity analysis where median_ST > 0.05

PRIOR = {
    "N_tumor_init": (100, 140), 
    "N_CD8_init": (10, 40), 
    "N_CD4_init": (40, 70),
    "N_Mac_init": (25,45), 
    "N_MDSC_init": (10, 30),
    "log10_CancerCell_progGrowthRate": (-7, -6),
    "log10_CancerCell_stemGrowthRate": (-7, -6),
    "how_exhausted": (0.6, 1),
    "log10_k_TCD8_rec": (-8, -5),  
    "log10_k_TCD4_rec": (-8, -6),
    "log10_k_TCR_activation": (-6, -4),
    "M1_M2_rec_ratio": (1, 2),   # M1:M2 recruitment ratio (literature 1.3:1)
    "TCD8_moveProb": (0.5, 1),
    "log10_cd8_commitment_delay": (3, 7),
    "log10_k_Mac_rec": (-10, -8),
    "PDL1_baseline_mean": (0, 1),
    "log10_k_PDL1_upregulation": (-8, -6),
    "log10_TCD4_death_rate": (-6, -5),
    "exhaustion_threshold_eff": (0.1, 0.35),   # default 0.3; boundary for being counted exhausted; max < term min
    "exhaustion_threshold_term": (0.4, 0.8),   # default 0.70; proliferation arrest; must stay > exhaustion_threshold_eff
    "log10_k_PD1_signaling": (-6, -4),         # default 2e-5
}

prior = pyabc.Distribution(**{k: pyabc.RV("uniform", lo, hi - lo) for k, (lo, hi) in PRIOR.items()})

def to_params(par):
    out = {}

    for k in PRIOR:

        if k.startswith("log10_"):
            out[k[6:]] = 10 ** par[k]
        else:
            out[k] = int(round(par[k])) if "init" in k else par[k]

    return out


def get_final_fractions(model):

    n_cancer = model.count_agents(A.CancerCell)

    n_m1 = model.count_cell_type(
        A.Macrophage,
        subtype=A.MacSubtype.M1
    )

    n_m2 = model.count_cell_type(
        A.Macrophage,
        subtype=A.MacSubtype.M2
    )

    # Naive = differentiation state (PD1NEG also includes activated, low-exhaustion cells).
    # Naive cells never accumulate exhaustion, so naive and terminal never overlap.
    n_cd8_naive = model.count_cell_type(
        A.CD8TCell,
        diff_state=A.CD8DiffState.CD8TNAIVE
    )

    # Exhausted = intermediate (PD-1+, still dividing) + terminal, to match the data's
    # CD8_EX + CD8_EX_Proliferating. Effector = remaining non-naive (CD8_EM/EMRA).
    # Counted by exhaustion_level, not the CD8TEFFECTOR enum: non-naive cells are created in
    # CD8TEFFECTOR with level 0 (and can_exhaust=False cells never leave it).
    n_cd8_exhausted = sum(
        1 for a in model.schedule.agents
        if isinstance(a, A.CD8TCell) and getattr(a, "alive", True)
        and a.diff_state != A.CD8DiffState.CD8TNAIVE
        and a.exhaustion_level > P.exhaustion_threshold_eff
    )

    n_cd8_effector = max(
        0,
        model.count_agents(A.CD8TCell) - n_cd8_naive - n_cd8_exhausted
    )

    n_cd4_helper = model.count_cell_type(
        A.CD4TCell,
        diff_state=A.CD4DiffState.CD4THELPER
    )

    n_cd4_treg = model.count_cell_type(
        A.CD4TCell,
        diff_state=A.CD4DiffState.CD4TREG
    )

    n_cd4_total = model.count_agents(A.CD4TCell)

    n_cd4_naive = max(
        0,
        n_cd4_total - n_cd4_helper - n_cd4_treg
    )

    total = (
        n_cancer +
        n_m1 +
        n_m2 +
        n_cd8_effector +
        n_cd8_exhausted +
        n_cd8_naive +
        n_cd4_helper +
        n_cd4_treg +
        n_cd4_naive
    )

    if total == 0:
        raise ValueError("No cells remain")

    return {
        "Cancer_cells": n_cancer / total,
        "M1_like": n_m1 / total,
        "M2_like": n_m2 / total,
        "CD8_Effector": n_cd8_effector / total,
        "CD8_Exhausted": n_cd8_exhausted / total,
        "CD8_Naive": n_cd8_naive / total,
        "CD4_Helper": n_cd4_helper / total,
        "CD4_Treg": n_cd4_treg / total,
        "CD4_Naive": n_cd4_naive / total,
    }


# ---------------------------------------------------------------- model and distance
def model(par):
    # print(f"ENTER model PID={os.getpid()}", flush=True)
    seed = int.from_bytes(os.urandom(4), "little")

    random.seed(seed)
    np.random.seed(seed)

    params = to_params(par)
    # print(f"PID={os.getpid()} params={params}", flush=True)

    ABC_INIT_PARAMS = {
        "N_tumor_init",
        "N_CD8_init",
        "N_CD4_init",
        "N_Mac_init",
        "N_MDSC_init",
    }

    for k in params:
        if k not in ABC_INIT_PARAMS and not hasattr(P, k):
            raise ValueError(
                f"ABC parameter '{k}' not found in params.py"
            )

    original_vals = {
        k: getattr(P, k)
        for k in params
        if hasattr(P, k)
    }

    try:

        # Override kinetic parameters
        for k, v in params.items():

            if hasattr(P, k):
                setattr(P, k, v)

        scenario = DEFAULT_SCENARIOS["standard"]

        n_total = (
            params["N_tumor_init"] +
            params["N_CD8_init"] +
            params["N_CD4_init"] +
            params["N_Mac_init"] +
            params["N_MDSC_init"]
        )

        capacity = scenario["width"] * scenario["height"]

        if n_total > capacity:
            raise ValueError(
                f"Initial population {n_total} exceeds grid capacity {capacity}"
            )

        # IMPORTANT:
        # Initial conditions come from ABC particle
        model = ABM_Model(
            width=scenario["width"],
            height=scenario["height"],
            initial_tumor_cells=params["N_tumor_init"],
            initial_CD8Tcells=params["N_CD8_init"],
            initial_CD4Tcells=params["N_CD4_init"],
            initial_macrophages=params["N_Mac_init"],
            initial_MDSC=params["N_MDSC_init"]
        )

        # print("ABM initialized", flush=True)

        for step in range(scenario["steps"]):
            t0 = time.time()
            model.step()
            dt = time.time() - t0
            if not model.running:
                break

        return get_final_fractions(model)

    except Exception as e:

        print(f"ABM failure: {e}")

        return {
            k: -1.0
            for k in KEYS
        }

    finally:
        # restore params.py
        for k, v in original_vals.items():
            setattr(P, k, v)

def distance(x, x_0):
    """Hellinger distance between the simulated and observed compositions (0..1)."""
    p = np.array([x[k] for k in KEYS])
    q = np.array([x_0[k] for k in KEYS])
    if p.min() < 0:
        return 1e3
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def weighted_median(x, w):
    """Value at which the cumulative normalised weight first reaches 0.5."""
    order = np.argsort(x)
    cum = np.cumsum(w[order]) / np.sum(w)
    return x[order][np.searchsorted(cum, 0.5)]


def prior_predictive_check(obs, d):
    """Simulate from the prior; flag observed values outside the range the model can produce."""
    out = os.path.join(d, "prior_predictive_summary.csv")
    if os.path.exists(out):                # already done on an earlier (resumed) run
        return
    draws = [dict(prior.rvs()) for _ in range(N_PRIOR_PC)]
    with Pool(N_PROCS) as pool:
        sims = pd.DataFrame(pool.map(model, draws))
    sims = sims[(sims >= 0).all(axis=1)]
    sims.to_csv(os.path.join(d, "prior_predictive_simulations.csv"), index=False)
    summary = pd.DataFrame({"observed": pd.Series(obs), "min": sims.min(),
                            "q05": sims.quantile(0.05), "median": sims.median(),
                            "q95": sims.quantile(0.95), "max": sims.max()})
    summary["reachable"] = (summary.observed >= summary["min"]) & (summary.observed <= summary["max"])
    summary["inside_90"] = (summary.observed >= summary.q05) & (summary.observed <= summary.q95)
    summary.to_csv(out)
    print("Prior predictive check:\n", summary, flush=True)
    bad = summary.index[~summary.reachable].tolist()
    if bad:
        print(f"WARNING: observed values outside all {len(sims)} prior simulations: {bad}. "
              "PPC coverage cannot reach these without model/prior changes.", flush=True)


# ---------------------------------------------------------------- one patient
def fit_patient(pid, obs):
    d = os.path.join(OUT, pid)
    os.makedirs(d, exist_ok=True)
    db = os.path.join(d, "abc.db")
    prior_predictive_check(obs, d)
    if PRIOR_CHECK_ONLY:
        return

    abc = pyabc.ABCSMC(
            model,
            prior,
            distance,
            population_size=pyabc.AdaptivePopulationSize(
                start_nr_particles=POP_SIZE,
                max_population_size=1000,
                min_population_size=POP_SIZE
            ),
            # LocalTransition: per-particle kernel from its nearest neighbours instead of one
            # global Gaussian -> proposals track the posterior shape, which limits weight collapse
            transitions=pyabc.LocalTransition(k_fraction=0.25),
            # median-based schedule: shrinks epsilon faster, so weights do less corrective work
            eps=pyabc.QuantileEpsilon(alpha=0.5),
            sampler=pyabc.sampler.MulticoreEvalParallelSampler(
                n_procs=N_PROCS
            )
        )
    if os.path.exists(db):                 # job was killed earlier: continue from the saved generations
        abc.load("sqlite:///" + db, 1)
    else:
        abc.new("sqlite:///" + db, obs)
    h = abc.run(
        # minimum_epsilon=MIN_EPSILON, 
        max_nr_populations=MAX_GENERATIONS,
                min_acceptance_rate=MIN_ACCEPTANCE, max_total_nr_simulations=MAX_SIMS)

    # --- convergence diagnostics (pyABC plots + a table)
    pops = h.get_all_populations()
    pops = pops[pops.t >= 0].copy()
    # population size is adaptive (500-1000), so use the actual accepted count per generation
    pops["acceptance_rate"] = pops["particles"] / pops["samples"]
    pops.to_csv(os.path.join(d, "generations.csv"), index=False)
    for name, fn in [("epsilon", plot_epsilons), ("ess", plot_effective_sample_sizes),
                     ("acceptance", plot_acceptance_rates_trajectory)]:
        fn([h]).figure.savefig(os.path.join(d, name + ".png"), dpi=120)
        plt.close("all")
    plot_kde_matrix_highlevel(h, m=0, t=h.max_t)
    plt.savefig(os.path.join(d, "posterior_kde.png"), dpi=120)
    plt.close("all")

    # --- posterior samples (equal-weight resample of the final generation)
    df, w = h.get_distribution(m=0, t=h.max_t)
    for t in range(h.max_t + 1):

        pop, w = h.get_distribution(m=0, t=t)

        w = np.asarray(w)
        w = w / w.sum()

        ess = 1.0 / np.sum(w**2)

        print(
            f"Generation {t}: "
            f"ESS={ess:.2f} "
            f"ESS/N={ess/len(w):.3f} "
            f"max_w={w.max():.3f}",
            flush = True
        )
    w = np.asarray(w)
    w = w / w.sum()

    map_summary = []

    for param in df.columns:

        plt.figure(figsize=(5, 4))

        counts, bins, _ = plt.hist(
            df[param],
            bins=30,
            weights=w,
            density=True,
            alpha=0.7
        )

        peak_idx = np.argmax(counts)

        map_est = 0.5 * (
            bins[peak_idx]
            + bins[peak_idx + 1]
        )
        
        map_summary.append({
            "parameter": param,
            "MAP": map_est,
            "mean": np.average(df[param], weights=w),
            "median": weighted_median(df[param].to_numpy(), w)
        })

        plt.axvline(
            map_est,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"MAP = {map_est:.3g}"
        )

        plt.xlabel(param)
        plt.ylabel("Posterior density")
        plt.title(f"Posterior: {param}")
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(d, f"{param}_posterior.png"),
            dpi=150
        )

        plt.close()

    pd.DataFrame(map_summary).to_csv(
        os.path.join(d, "posterior_modes.csv"),
        index=False
    )
    
    post = df.iloc[np.random.choice(len(df), 1000, p=w / w.sum())].reset_index(drop=True)
    for k in PRIOR:
        if k.startswith("log10_"):
            post[k[6:]] = 10 ** post[k]
    post.to_csv(os.path.join(d, "posterior_samples.csv"), index=False)
    print(post.describe().T[["mean", "std", "min", "max"]])

    # --- posterior predictive check: fresh simulations from posterior draws
    draw_idx = np.random.choice(len(post), N_PPC, replace=True)
    draws = [post.iloc[i][list(PRIOR)].to_dict() for i in draw_idx]

    with Pool(N_PROCS) as pool:
        sims = pd.DataFrame(pool.map(model, draws))
    sims = sims[(sims >= 0).all(axis=1)]
    sims.to_csv(os.path.join(d, "ppc_simulations.csv"), index=False)
    lo, med, hi = (sims.quantile(q) for q in (0.05, 0.5, 0.95))
    summary = pd.DataFrame({"observed": pd.Series(obs), "pred_median": med, "pred_q05": lo, "pred_q95": hi})
    summary["inside_90"] = (summary.observed >= summary.pred_q05) & (summary.observed <= summary.pred_q95)
    summary.to_csv(os.path.join(d, "ppc_summary.csv"))
    print(summary)
    print(f"{pid}: {h.total_nr_simulations} sims, final eps {pops.epsilon.iloc[-1]:.3f}, "
          f"PPC coverage {summary.inside_90.mean():.2f}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True) # Forces forking the process state, not starting a new interpreter
    print("Starting script", flush=True)
    print("N_PROCS =", N_PROCS, flush=True)


    # start = time.time()
    # test_par = prior.rvs()
    # model(test_par)
    # print(f"Wall time = {time.time() - start:.1f} s")

    # cProfile.runctx(
    #     "model(test_par)",
    #     globals(),
    #     locals(),
    #     "profile.out"
    # )

    # stats = pstats.Stats("profile.out")
    # stats.sort_stats("cumtime")
    # stats.print_stats(30)

    # sys.exit()
    ####
    data = pd.read_csv(DATA)
    data[KEYS] = data[KEYS].div(data[KEYS].sum(axis=1), axis=0)   # renormalise over the nine categories
    rows = [int(sys.argv[1])] if len(sys.argv) > 1 else range(len(data))
    for i in rows:
        r = data.iloc[i]
        fit_patient(str(r["Patient_ID"]), {k: float(r[k]) for k in KEYS})