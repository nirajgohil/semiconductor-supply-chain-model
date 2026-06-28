"""
Robustness analysis for the two headline qualitative conclusions:
  (C1) Advanced packaging is the binding constraint in the AI era (2024-2030).
  (C2) Capital-only and capital+workforce policy produce negligible effective-output
       gain (< 2%) at the 20-year horizon, while ecosystem-aligned policy produces
       a large gain (> 50%).

We sample the 8 key structural parameters over plausible ranges (the same ranges
used in the Sobol analysis) via Latin-hypercube sampling, and for each draw we
check whether C1 and C2 hold. We report the fraction of the parameter space in
which each conclusion is preserved. A conclusion that holds across the great
majority of the space is "structurally robust" rather than an artifact of the
central calibration.
"""
import numpy as np
from dataclasses import asdict
from scipy.stats import qmc
from sd_model import Params, SemiconductorSD


PROBLEM = {
    'names': ['tau_fab', 'tau_pkg_base', 'tau_w', 'beta',
              'delta_fab', 'delta_pkg', 'demand_growth', 'ai_surge_amp'],
    'bounds': [
        [3.0, 5.0],    # tau_fab
        [2.0, 3.5],    # tau_pkg_base
        [5.0, 10.0],   # tau_w
        [0.10, 0.30],  # beta
        [0.05, 0.12],  # delta_fab
        [0.04, 0.10],  # delta_pkg
        [0.05, 0.10],  # demand_growth
        [0.8, 2.0],    # ai_surge_amp
    ],
}


def ti(r, yr):
    return int(np.argmin(np.abs(r['t'] - yr)))


def build_params(vec, policy=(0, 0, 0)):
    p = Params()
    p.dt = 0.125
    for name, val in zip(PROBLEM['names'], vec):
        setattr(p, name, float(val))
    p.p_capital, p.p_workforce, p.p_ecosystem = policy
    return p


def check_draw(vec):
    """Return (C1_holds, C2_holds, gains) for one parameter draw."""
    # baseline
    base = SemiconductorSD(build_params(vec, (0, 0, 0))).simulate(to_horizon=True)

    # C1: packaging binding in the AI era (check 2024, 2027, 2030)
    ai_years = [2024, 2027, 2030]
    c1 = all(base['binding'][ti(base, y)] == 'packaging' for y in ai_years)

    # policy scenarios
    cap = SemiconductorSD(build_params(vec, (1, 0, 0))).simulate(to_horizon=True)
    capw = SemiconductorSD(build_params(vec, (1, 1, 0))).simulate(to_horizon=True)
    eco = SemiconductorSD(build_params(vec, (1, 1, 1))).simulate(to_horizon=True)

    i22 = ti(base, 2022); i42 = ti(base, 2042)
    qb22 = base['Q_eff'][i22]
    def gain(r):
        # gain at 2042 vs baseline, both indexed to own 2022
        rb = base['Q_eff'][i42] / qb22
        rr = r['Q_eff'][i42] / (r['Q_eff'][i22])
        return (rr - rb) / rb * 100
    g_cap = gain(cap)
    g_capw = gain(capw)
    g_eco = gain(eco)

    # C2: capital-only and capital+workforce negligible (<2%), ecosystem large (>50%)
    c2 = (abs(g_cap) < 2.0) and (abs(g_capw) < 2.0) and (g_eco > 50.0)

    return c1, c2, (g_cap, g_capw, g_eco)


def run(n=200, seed=42):
    sampler = qmc.LatinHypercube(d=len(PROBLEM['names']), seed=seed)
    unit = sampler.random(n)
    lb = np.array([b[0] for b in PROBLEM['bounds']])
    ub = np.array([b[1] for b in PROBLEM['bounds']])
    samples = lb + unit * (ub - lb)

    c1_count = 0
    c2_count = 0
    both_count = 0
    cond_cap_ineffective = 0
    eco_gains = []
    cap_gains = []
    eco_when_binding = []
    for vec in samples:
        c1, c2, (gc, gcw, ge) = check_draw(vec)
        c1_count += int(c1)
        c2_count += int(c2)
        both_count += int(c1 and c2)
        eco_gains.append(ge)
        cap_gains.append(max(abs(gc), abs(gcw)))
        if c1:
            eco_when_binding.append(ge)
            if abs(gc) < 2.0 and abs(gcw) < 2.0:
                cond_cap_ineffective += 1

    eco_gains = np.array(eco_gains)
    cap_gains = np.array(cap_gains)
    eco_when_binding = np.array(eco_when_binding)
    n_binding = int(c1_count)

    # central region: +/-20% around calibrated values
    p0 = Params()
    base_vals = np.array([getattr(p0, nm) for nm in PROBLEM['names']])
    s2 = qmc.LatinHypercube(d=len(PROBLEM['names']), seed=seed + 1)
    u2 = s2.random(n)
    lo = base_vals * 0.8; hi = base_vals * 1.2
    samp2 = lo + u2 * (hi - lo)
    c1c = 0; c2c = 0
    for vec in samp2:
        c1, c2, _ = check_draw(vec)
        c1c += int(c1); c2c += int(c2)

    print("=" * 72)
    print(f"ROBUSTNESS ANALYSIS  (Latin-hypercube, n={n} draws over 8 parameters)")
    print("=" * 72)
    print("FULL plausible parameter space:")
    print(f"  C1  packaging binding in AI era (2024-2030):  "
          f"{c1_count}/{n} = {100*c1_count/n:.1f}%")
    print(f"  C2  capital ineffective (<2%) & ecosystem effective (>50%):  "
          f"{c2_count}/{n} = {100*c2_count/n:.1f}%")
    print("CENTRAL region (+/-20% of calibrated values):")
    print(f"  C1: {c1c}/{n} = {100*c1c/n:.1f}%    C2: {c2c}/{n} = {100*c2c/n:.1f}%")
    print("CONDITIONAL (the deterministic mechanism):")
    print(f"  Given packaging binding, capital ineffective: "
          f"{cond_cap_ineffective}/{n_binding} = {100*cond_cap_ineffective/max(n_binding,1):.1f}%")
    print(f"  Ecosystem gain when packaging binds: "
          f"min={eco_when_binding.min():.0f}% median={np.median(eco_when_binding):.0f}% "
          f"max={eco_when_binding.max():.0f}%")
    print("=" * 72)

    np.savez('/home/claude/semi_paper/model/robustness_results.npz',
             eco_gains=eco_gains, cap_gains=cap_gains,
             eco_when_binding=eco_when_binding,
             c1_frac=c1_count/n, c2_frac=c2_count/n, both_frac=both_count/n,
             c1_central=c1c/n, c2_central=c2c/n,
             cond_cap_ineffective=cond_cap_ineffective/max(n_binding, 1),
             n=n, n_binding=n_binding)
    return c1_count/n, c2_count/n, both_count/n, eco_gains, cap_gains


if __name__ == '__main__':
    run(n=200)
