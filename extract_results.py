"""Extract all numbers needed for manuscript tables/text into a single dump."""
import json
import numpy as np
from sd_model import Params, SemiconductorSD, make_scenarios

def ti(r, yr):
    return int(np.argmin(np.abs(r['t'] - yr)))

out = {}

# --- historical output index, baseline (1990=100) ---
p = Params()
rb = SemiconductorSD(p).simulate()
q0 = rb['Q_eff'][0]
out['historical_baseline_index'] = {
    str(yr): round(float(rb['Q_eff'][ti(rb, yr)] / q0 * 100), 1)
    for yr in [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2022, 2025]
}
out['historical_yield'] = {
    str(yr): round(float(rb['Y'][ti(rb, yr)]), 3)
    for yr in [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]
}

# --- forward horizon scenarios: COMMON BASE (baseline 2022 = 100 for all) ---
# This ensures the index columns and the gain column are mutually consistent:
# gain = (scenario_2042 - baseline_2042) / baseline_2042 computed from these indices.
scen = make_scenarios()
base_run = SemiconductorSD(scen['Baseline']).simulate(to_horizon=True)
q22_base = base_run['Q_eff'][ti(base_run, 2022)]   # single common normalizer
fwd = {}
for name, sp in scen.items():
    r = SemiconductorSD(sp).simulate(to_horizon=True)
    fwd[name] = {
        str(yr): round(float(r['Q_eff'][ti(r, yr)] / q22_base * 100), 1)
        for yr in [2022, 2024, 2027, 2032, 2037, 2042]
    }
out['forward_scenarios_index_commonbase'] = fwd

# --- 5yr and 20yr gains vs baseline ---
base = SemiconductorSD(scen['Baseline']).simulate(to_horizon=True)
gains = {}
for name in ['Capital-only', 'Capital+Workforce', 'Ecosystem-aligned']:
    r = SemiconductorSD(scen[name]).simulate(to_horizon=True)
    for yr, key in [(2027, '5yr'), (2032, '10yr'), (2042, '20yr')]:
        i = ti(base, yr)
        g = (r['Q_eff'][i] - base['Q_eff'][i]) / base['Q_eff'][i] * 100
        gains.setdefault(name, {})[key] = round(float(g), 1)
out['gains_vs_baseline_pct'] = gains

# --- binding constraint sequence ---
binding = {}
rb_h = SemiconductorSD(Params()).simulate(to_horizon=True)
for yr in [1995, 2000, 2005, 2010, 2015, 2018, 2020, 2022, 2024, 2026, 2028, 2030, 2035, 2040]:
    binding[str(yr)] = rb_h['binding'][ti(rb_h, yr)]
out['binding_sequence'] = binding

# --- sobol results ---
data = np.load('sobol_results.npz', allow_pickle=True)
sob = {}
for i, nm in enumerate(data['names']):
    sob[str(nm)] = {
        'S1': round(float(data['S1'][i]), 3),
        'ST': round(float(data['ST'][i]), 3),
        'ST_conf': round(float(data['ST_conf'][i]), 3),
    }
out['sobol'] = sob
out['sobol_output_range'] = {
    'min': round(float(data['Y'].min()), 1),
    'max': round(float(data['Y'].max()), 1),
    'mean': round(float(data['Y'].mean()), 1),
    'std': round(float(data['Y'].std()), 1),
}

# --- parameter table (calibrated central values) ---
p = Params()
out['parameters'] = {
    'tau_fab': p.tau_fab, 'tau_pkg_base': p.tau_pkg_base, 'tau_pkg_fast': p.tau_pkg_fast,
    'tau_w': p.tau_w, 'delta_fab': p.delta_fab, 'delta_pkg': p.delta_pkg,
    'delta_w': p.delta_w, 'y_min': p.y_min, 'y_max': p.y_max, 'beta': p.beta,
    'demand_growth': p.demand_growth, 'ai_surge_amp': p.ai_surge_amp,
    'ai_surge_year': p.ai_surge_year, 'ai_surge_width': p.ai_surge_width,
    'n_fab': p.n_fab, 'n_pkg': p.n_pkg, 'n_w': p.n_w,
    'a_eq': p.a_eq, 'util_target': p.util_target, 'dt': p.dt,
}

# --- implied learning rate per doubling from beta ---
# Y = ymin + (ymax-ymin)(1 - exp(-beta * E/E0)). Translate to a cost-style LR proxy:
# We report the experience needed to close x% of the yield gap.
beta = p.beta
e_half = -np.log(0.5) / beta   # normalized experience to close half the gap
out['learning'] = {
    'beta': beta,
    'norm_experience_to_close_half_gap': round(float(e_half), 2),
    'note': 'memory-chip learning curves cluster near the 72% type (Irwin-Klenow); ~28% LR'
}

with open('results_dump.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
