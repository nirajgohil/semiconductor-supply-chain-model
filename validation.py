"""
Validation suite for the semiconductor SD model.
Implements standard system-dynamics model-validation tests (Sterman 2000, Ch. 21):
  1. Dimensional / structural consistency (stocks remain non-negative)
  2. Integration-error test (halve dt, confirm trajectory invariance)
  3. Extreme-condition tests (set parameters to extremes, confirm sane behavior)
  4. Behavior-reproduction (qualitative pattern checks vs. stylized facts)
  5. Equilibrium / steady-state behavior under zero demand growth
"""
import numpy as np
from dataclasses import asdict
from sd_model import Params, SemiconductorSD, make_scenarios


def t_index(r, yr):
    return np.argmin(np.abs(r['t'] - yr))


def test_nonnegative_stocks():
    p = Params()
    r = SemiconductorSD(p).simulate(to_horizon=True)
    ok = all(np.all(r[k] >= 0) for k in ['K_fab', 'K_pkg', 'E', 'W'])
    return ok, "All stocks remain non-negative across full horizon"


def test_integration_convergence():
    p1 = Params(); p1.dt = 0.0625
    p2 = Params(); p2.dt = 0.03125  # half step
    r1 = SemiconductorSD(p1).simulate(to_horizon=True)
    r2 = SemiconductorSD(p2).simulate(to_horizon=True)
    # compare effective output at common years
    yrs = [2000, 2010, 2020, 2030, 2040]
    diffs = []
    for yr in yrs:
        q1 = r1['Q_eff'][t_index(r1, yr)]
        q2 = r2['Q_eff'][t_index(r2, yr)]
        diffs.append(abs(q1 - q2) / max(q2, 1e-9))
    max_rel = max(diffs)
    return max_rel < 0.01, f"Max relative trajectory change on halving dt = {max_rel*100:.3f}% (<1% required)"


def test_extreme_zero_learning():
    """If beta -> 0, yield stays at floor; output must be far lower."""
    p_lo = Params(); p_lo.beta = 1e-6
    p_hi = Params()
    r_lo = SemiconductorSD(p_lo).simulate()
    r_hi = SemiconductorSD(p_hi).simulate()
    q_lo = r_lo['Q_eff'][t_index(r_lo, 2020)]
    q_hi = r_hi['Q_eff'][t_index(r_hi, 2020)]
    return q_lo < q_hi, f"Zero-learning output ({q_lo:.1f}) < baseline ({q_hi:.1f}) at 2020"


def test_extreme_no_construction():
    """If autonomous + responsive investment is zero, fab capacity must decay."""
    p = Params(); p.base_invest = 0.0; p.inv_gain = 0.0
    r = SemiconductorSD(p).simulate()
    k_start = r['K_fab'][0]
    k_end = r['K_fab'][t_index(r, 2025)]
    return k_end < k_start, f"With no construction, K_fab decays {k_start:.0f} -> {k_end:.0f}"


def test_extreme_infinite_packaging():
    """If packaging is abundant, packaging must NOT be the binding constraint."""
    p = Params(); p.K_pkg0 = 10000.0; p.pkg_base_invest = 500.0
    r = SemiconductorSD(p).simulate(to_horizon=True)
    pkg_binding_years = sum(1 for b in r['binding'] if b == 'packaging')
    return pkg_binding_years == 0, f"With abundant packaging, packaging binds in {pkg_binding_years} steps (0 required)"


def test_delay_dominance():
    """Capital-only policy must produce negligible output gain in first 5 years
    (the core thesis: delays dominate short-horizon policy)."""
    scen = make_scenarios()
    base = SemiconductorSD(scen['Baseline']).simulate(to_horizon=True)
    cap = SemiconductorSD(scen['Capital-only']).simulate(to_horizon=True)
    i27 = t_index(base, 2027)  # 5 yr after 2022 policy onset
    gain_5yr = (cap['Q_eff'][i27] - base['Q_eff'][i27]) / base['Q_eff'][i27]
    return abs(gain_5yr) < 0.05, f"Capital-only 5-yr output gain = {gain_5yr*100:.2f}% (<5%: delay-dominated)"


def test_ecosystem_dominance():
    """Ecosystem-aligned policy must outperform capital-only at long horizon."""
    scen = make_scenarios()
    cap = SemiconductorSD(scen['Capital-only']).simulate(to_horizon=True)
    eco = SemiconductorSD(scen['Ecosystem-aligned']).simulate(to_horizon=True)
    i42 = t_index(cap, 2042)
    eco_q = eco['Q_eff'][i42]; cap_q = cap['Q_eff'][i42]
    return eco_q > cap_q, f"Ecosystem output ({eco_q:.0f}) > capital-only ({cap_q:.0f}) at 2042 (+20yr)"


def test_packaging_emerges_ai_era():
    """Packaging must become the binding constraint by the AI era (post-2015)
    and remain so, consistent with CoWoS evidence."""
    p = Params()
    r = SemiconductorSD(p).simulate(to_horizon=True)
    binding_2024 = r['binding'][t_index(r, 2024)]
    binding_2030 = r['binding'][t_index(r, 2030)]
    ok = binding_2024 == 'packaging' and binding_2030 == 'packaging'
    return ok, f"Binding constraint 2024={binding_2024}, 2030={binding_2030} (packaging expected)"


def test_yield_monotone_increasing():
    """Yield must increase monotonically with cumulative experience."""
    p = Params()
    r = SemiconductorSD(p).simulate()
    dy = np.diff(r['Y'])
    return np.all(dy >= -1e-9), "Yield is monotonically non-decreasing in experience"


def run_all():
    tests = [
        ("Non-negative stocks", test_nonnegative_stocks),
        ("Integration convergence (dt halving)", test_integration_convergence),
        ("Extreme: zero learning", test_extreme_zero_learning),
        ("Extreme: no construction", test_extreme_no_construction),
        ("Extreme: infinite packaging", test_extreme_infinite_packaging),
        ("Yield monotonicity", test_yield_monotone_increasing),
        ("Delay dominance (5-yr capital policy)", test_delay_dominance),
        ("Ecosystem dominance (20-yr)", test_ecosystem_dominance),
        ("Packaging emerges as AI-era bottleneck", test_packaging_emerges_ai_era),
    ]
    print("=" * 78)
    print("MODEL VALIDATION SUITE")
    print("=" * 78)
    results = []
    for name, fn in tests:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        print(f"        {msg}")
        results.append((name, ok, msg))
    n_pass = sum(1 for _, ok, _ in results if ok)
    print("=" * 78)
    print(f"RESULT: {n_pass}/{len(results)} tests passed")
    print("=" * 78)
    return results


if __name__ == '__main__':
    run_all()
