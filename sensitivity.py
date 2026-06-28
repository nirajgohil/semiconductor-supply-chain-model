"""
Global sensitivity analysis (Sobol variance decomposition) on the SD model.
Output of interest: effective semiconductor output index at 2042 (+20yr horizon)
under the ecosystem-aligned policy, and the 2025 output under baseline.

Uses SALib Saltelli sampling + Sobol indices.
"""
import numpy as np
from dataclasses import asdict
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from sd_model import Params, SemiconductorSD


# Parameters to vary, with +/- ranges around calibrated central values
PROBLEM = {
    'num_vars': 8,
    'names': [
        'tau_fab',        # fab construction delay
        'tau_pkg_base',   # packaging build delay (pre-2023)
        'tau_w',          # workforce formation delay
        'beta',           # learning speed
        'delta_fab',      # fab depreciation
        'delta_pkg',      # packaging depreciation
        'demand_growth',  # baseline demand growth
        'ai_surge_amp',   # AI packaging-demand surge amplitude
    ],
    'bounds': [
        [3.0, 5.0],       # tau_fab
        [2.0, 3.5],       # tau_pkg_base
        [5.0, 10.0],      # tau_w
        [0.10, 0.30],     # beta
        [0.05, 0.12],     # delta_fab
        [0.04, 0.10],     # delta_pkg
        [0.05, 0.10],     # demand_growth
        [0.8, 2.0],       # ai_surge_amp
    ],
}


def t_index(r, yr):
    return np.argmin(np.abs(r['t'] - yr))


def evaluate(param_vec, policy='ecosystem', year=2042):
    p = Params()
    p.dt = 0.125  # coarser step for speed; dt-convergence already verified <1%
    names = PROBLEM['names']
    for name, val in zip(names, param_vec):
        setattr(p, name, float(val))
    if policy == 'ecosystem':
        p.p_capital, p.p_workforce, p.p_ecosystem = 1.0, 1.0, 1.0
    r = SemiconductorSD(p).simulate(to_horizon=True)
    i22 = t_index(r, 2022); q22 = r['Q_eff'][i22]
    return r['Q_eff'][t_index(r, year)] / q22 * 100


def run_sobol(n_base=256, policy='ecosystem', year=2042):
    param_values = sobol_sample.sample(PROBLEM, n_base, calc_second_order=False)
    Y = np.array([evaluate(pv, policy, year) for pv in param_values])
    Si = sobol_analyze.analyze(PROBLEM, Y, calc_second_order=False, print_to_console=False)
    return Si, Y


if __name__ == '__main__':
    print("Running Sobol sensitivity analysis (ecosystem policy, 2042 output)...")
    Si, Y = run_sobol(n_base=64)
    print(f"\nOutput range across samples: [{Y.min():.1f}, {Y.max():.1f}], "
          f"mean={Y.mean():.1f}, std={Y.std():.1f}\n")
    print(f"{'Parameter':16s} {'S1 (first-order)':>18s} {'ST (total)':>14s}")
    print("-" * 50)
    order = np.argsort(-Si['ST'])
    for idx in order:
        name = PROBLEM['names'][idx]
        print(f"{name:16s} {Si['S1'][idx]:>18.3f} {Si['ST'][idx]:>14.3f}")

    # save for figure
    np.savez('/home/claude/semi_paper/model/sobol_results.npz',
             names=np.array(PROBLEM['names']),
             S1=Si['S1'], ST=Si['ST'],
             S1_conf=Si['S1_conf'], ST_conf=Si['ST_conf'],
             Y=Y)
    print("\nSaved sobol_results.npz")
