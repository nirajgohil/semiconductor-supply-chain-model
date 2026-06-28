"""
System Dynamics Model of the U.S. Semiconductor Supply Chain
=============================================================
Implements the stock-flow structure specified in the manuscript (Section: Model).
Integrated numerically with an explicit RK4 scheme on an annual grid (with
sub-annual integration steps for numerical stability of the delay chains).

State variables (stocks):
    K_fab    Installed advanced-node fabrication capacity   [index, 1990 = 100]
    K_pkg    Advanced packaging (CoWoS-class) capacity       [index, 1990 = 100]
    E        Cumulative manufacturing experience             [normalized wafer-units]
    W        Skilled manufacturing workforce                 [index, 1990 = 100]

Delay chains (material/pipeline delays implemented as 3rd-order Erlang stages):
    Fab construction pipeline   (mean tau_fab  ~ 4.0 yr)
    Packaging build pipeline    (mean tau_pkg  ~ 2.5 yr, post-2023 ~1.75 yr)
    Workforce training pipeline (mean tau_w    ~ 6.0 yr)

Calibration anchors (all from public sources; see manuscript Table: Calibration):
    US capacity share 37% (1990) -> 12% (2020) -> 10% (2022)  [SIA/SEMI/VLSI]
    Learning rate ~ 28% per doubling (memory LC ~ 72% type)    [Irwin-Klenow / Applied Econ 1994]
    CoWoS build time 3-5 yr -> 1.5-2 yr after 2023             [TrendForce / TSMC]
    CoWoS capacity ~10x growth 2023-2026                        [TrendForce / SemiWiki]

Author: Niraj Gohil
"""

import numpy as np
from dataclasses import dataclass, field, asdict


# ----------------------------------------------------------------------------
# Parameter container
# ----------------------------------------------------------------------------
@dataclass
class Params:
    # --- Time grid ---
    t0: float = 1990.0
    t_end: float = 2025.0
    dt: float = 0.0625            # 1/16 yr integration step (16 steps/yr)
    horizon_end: float = 2042.0   # forward projection horizon (20 yr post-CHIPS)

    # --- Fabrication capacity ---
    delta_fab: float = 0.08       # annual depreciation/obsolescence of fab capacity
    tau_fab: float = 4.0          # mean fab construction + qualification delay (yr)
    n_fab: int = 3                # Erlang order for construction pipeline

    # --- Advanced packaging ---
    delta_pkg: float = 0.07       # annual depreciation of packaging capacity
    tau_pkg_base: float = 2.5     # mean packaging build delay pre-2023 (yr)
    tau_pkg_fast: float = 1.75    # mean packaging build delay post-2023 (yr)
    n_pkg: int = 3                # Erlang order
    pkg_year_break: float = 2023.0

    # --- Workforce ---
    tau_w: float = 6.0            # mean workforce formation delay (yr)
    n_w: int = 3                  # Erlang order
    delta_w: float = 0.04         # annual attrition rate

    # --- Yield learning (saturating exponential in cumulative experience) ---
    y_min: float = 0.20           # floor yield at node introduction
    y_max: float = 0.92           # asymptotic mature yield
    beta: float = 0.18            # learning speed (per unit normalized experience)
    # Learning rate per doubling implied by beta is reported in the manuscript.

    # --- Demand (exogenous driver, normalized index) ---
    demand_growth: float = 0.072  # baseline annual growth of perceived demand
    demand_0: float = 95.0
    # AI-era demand acceleration for advanced packaging (logistic surge)
    ai_surge_year: float = 2022.5  # inflection year of the AI packaging surge
    ai_surge_width: float = 1.2    # logistic width (yr); smaller = sharper jump
    ai_surge_amp: float = 1.2      # amplitude: post-surge multiplier = 1 + amp

    # --- Investment behavior (bounded rationality, anchor-and-adjust) ---
    inv_gain: float = 0.55        # responsiveness of construction starts to gap
    util_target: float = 0.85     # desired utilization
    smooth_time: float = 2.0      # perception smoothing time for utilization (yr)
    base_invest: float = 7.0      # autonomous construction starts (index units/yr)

    # --- Packaging investment (scales with AI-era demand signal) ---
    pkg_inv_gain: float = 0.5
    pkg_base_invest: float = 2.6

    # --- Workforce hiring ---
    hire_gain: float = 0.5
    w_target_ratio: float = 1.0   # desired workforce relative to capacity

    # --- Equipment availability (exogenous, captures EUV/tool concentration) ---
    a_eq: float = 0.92            # baseline tool availability fraction

    # --- Policy levers (set per scenario; 0 = no policy) ---
    p_capital: float = 0.0        # capital-cost reduction -> boosts construction starts
    p_workforce: float = 0.0      # workforce program -> boosts hiring + training speed
    p_ecosystem: float = 0.0      # ecosystem coordination -> boosts pkg + equipment + materials
    policy_start: float = 2022.0  # CHIPS Act enactment year
    cap_strength: float = 1.6     # capital lever -> construction-start multiplier strength
    eco_equip_strength: float = 0.07  # ecosystem lever -> equipment availability lift
    # Secondary couplings (imperfectly targeted policy partially relieves co-binding
    # fab/yield/utilization constraints even when packaging dominates):
    cap_util_lift: float = 0.04   # capital lever -> small utilization lift
    wf_yield_lift: float = 0.03   # workforce lever -> small effective-yield lift
    eco_pkg_throughput: float = 0.18  # ecosystem lever -> packaging throughput lift

    # --- Initial conditions (1990 = 100 index baseline) ---
    K_fab0: float = 100.0
    K_pkg0: float = 165.0
    E0: float = 50.0
    W0: float = 100.0


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class SemiconductorSD:
    """Stock-flow system dynamics model with explicit delay pipelines."""

    def __init__(self, p: Params):
        self.p = p

    # --- auxiliary functions ---
    def yield_fn(self, E):
        """Saturating learning curve: Y(E) = Ymin + (Ymax-Ymin)(1-exp(-beta E_norm))."""
        p = self.p
        E_norm = E / p.E0          # normalize so 1990 experience ~ 1
        return p.y_min + (p.y_max - p.y_min) * (1.0 - np.exp(-p.beta * E_norm))

    def demand(self, t):
        p = self.p
        return p.demand_0 * np.exp(p.demand_growth * (t - p.t0))

    def demand_pkg(self, t):
        """Packaging-bound demand: baseline demand times an AI-era surge multiplier.
        The surge is modeled as a logistic ramp centered on the generative-AI
        inflection (~2022-2023), capturing the discontinuous jump in
        heterogeneous-integration / HBM / CoWoS demand that made advanced
        packaging the emergent binding constraint."""
        p = self.p
        base = self.demand(t)
        # logistic surge multiplier: ~1 before inflection, -> (1+amp) after
        x = (t - p.ai_surge_year) / p.ai_surge_width
        logistic = 1.0 / (1.0 + np.exp(-x))
        mult = 1.0 + p.ai_surge_amp * logistic
        return base * mult

    def tau_pkg(self, t):
        p = self.p
        return p.tau_pkg_fast if t >= p.pkg_year_break else p.tau_pkg_base

    def policy_active(self, t):
        return 1.0 if t >= self.p.policy_start else 0.0

    # --- right-hand side ---
    def derivs(self, t, s):
        """
        State vector s layout:
          s[0]            K_fab
          s[1]            K_pkg
          s[2]            E
          s[3]            W
          s[4]            U_perceived (smoothed utilization)
          s[5:5+n_fab]    fab construction pipeline stages
          s[..+n_pkg]     packaging build pipeline stages
          s[..+n_w]       workforce training pipeline stages
        """
        p = self.p
        K_fab, K_pkg, E, W, U_perc = s[0], s[1], s[2], s[3], s[4]
        K_fab = max(K_fab, 1e-6)
        K_pkg = max(K_pkg, 1e-6)
        W = max(W, 1e-6)

        i = 5
        fab_pipe = s[i:i + p.n_fab]; i += p.n_fab
        pkg_pipe = s[i:i + p.n_pkg]; i += p.n_pkg
        w_pipe = s[i:i + p.n_w]; i += p.n_w

        pa = self.policy_active(t)

        # --- effective output (constraint-limited: min operator) ---
        Y = self.yield_fn(E)
        # workforce lever gives a small effective-yield lift (better process control)
        Y_eff = np.clip(Y + p.wf_yield_lift * p.p_workforce * pa, 0.0, 0.99)
        # workforce adequacy scales achievable utilization
        w_adequacy = np.clip(W / (p.w_target_ratio * K_fab), 0.0, 1.0)
        # capital lever gives a small utilization lift (better tool/logistics support)
        U = (p.util_target + p.cap_util_lift * p.p_capital * pa) * w_adequacy
        U = np.clip(U, 0.0, 1.0)
        Q_fab = Y_eff * U * K_fab
        # equipment availability, lifted by ecosystem policy
        a_eq_eff = np.clip(p.a_eq + p.eco_equip_strength * p.p_ecosystem * pa, 0.0, 0.99)
        Q_equip = a_eq_eff * K_fab
        # packaging availability fraction (throughput per unit packaging capacity).
        # Advanced packaging has structurally lower throughput-per-capacity than
        # front-end fab (lower economies of scale, heavy customization, substrate
        # dependence). This is what makes it the binding constraint under the AI surge.
        # Ecosystem policy lifts packaging throughput (substrate + OSAT coordination).
        a_pkg = 0.62 + p.eco_pkg_throughput * p.p_ecosystem * pa
        Q_pkg = a_pkg * K_pkg
        Q_eff = min(Q_fab, Q_equip, Q_pkg)

        # --- demand / gap signals ---
        D = self.demand(t)
        # utilization = output relative to capacity
        util_inst = np.clip(Q_eff / K_fab, 0.0, 1.5)
        dU_perc = (util_inst - U_perc) / p.smooth_time

        # capacity gap drives investment (demand vs effective output)
        gap = (D - Q_eff) / max(D, 1e-6)
        gap = np.clip(gap, -0.5, 1.0)

        # --- fab construction starts (bounded rational + policy) ---
        capital_boost = 1.0 + p.cap_strength * p.p_capital * pa
        starts_fab = (p.base_invest + p.inv_gain * gap * K_fab * 0.1) * capital_boost
        starts_fab = max(starts_fab, 0.0)

        # fab construction pipeline (Erlang: n stages each with rate n/tau)
        rate_fab = p.n_fab / p.tau_fab
        d_fab_pipe = np.zeros(p.n_fab)
        inflow = starts_fab
        for k in range(p.n_fab):
            outflow = rate_fab * fab_pipe[k]
            d_fab_pipe[k] = inflow - outflow
            inflow = outflow
        fab_completion = rate_fab * fab_pipe[-1]

        # --- packaging build starts ---
        # AI-era demand signal: packaging gap grows as demand outpaces pkg capacity
        D_pkg = self.demand_pkg(t)
        pkg_gap = (D_pkg - Q_pkg) / max(D_pkg, 1e-6)
        pkg_gap = np.clip(pkg_gap, -0.5, 1.2)
        eco_boost = 1.0 + 1.2 * p.p_ecosystem * pa
        starts_pkg = (p.pkg_base_invest + p.pkg_inv_gain * pkg_gap * K_pkg * 0.12) * eco_boost
        starts_pkg = max(starts_pkg, 0.0)

        rate_pkg = p.n_pkg / self.tau_pkg(t)
        d_pkg_pipe = np.zeros(p.n_pkg)
        inflow = starts_pkg
        for k in range(p.n_pkg):
            outflow = rate_pkg * pkg_pipe[k]
            d_pkg_pipe[k] = inflow - outflow
            inflow = outflow
        pkg_completion = rate_pkg * pkg_pipe[-1]

        # --- workforce hiring/training ---
        w_gap = (p.w_target_ratio * K_fab - W) / max(K_fab, 1e-6)
        w_gap = np.clip(w_gap, -0.5, 1.0)
        wf_boost = 1.0 + 1.0 * p.p_workforce * pa
        hires = (p.hire_gain * w_gap * K_fab * 0.15 + 2.0) * wf_boost
        hires = max(hires, 0.0)

        # workforce training pipeline (policy can shorten tau_w)
        tau_w_eff = p.tau_w * (1.0 - 0.35 * p.p_workforce * pa)
        tau_w_eff = max(tau_w_eff, 1.0)
        rate_w = p.n_w / tau_w_eff
        d_w_pipe = np.zeros(p.n_w)
        inflow = hires
        for k in range(p.n_w):
            outflow = rate_w * w_pipe[k]
            d_w_pipe[k] = inflow - outflow
            inflow = outflow
        w_completion = rate_w * w_pipe[-1]

        # --- stock derivatives ---
        dK_fab = fab_completion - p.delta_fab * K_fab
        dK_pkg = pkg_completion - p.delta_pkg * K_pkg
        dE = Q_eff                              # experience accumulates with output
        dW = w_completion - p.delta_w * W

        ds = np.concatenate([
            [dK_fab, dK_pkg, dE, dW, dU_perc],
            d_fab_pipe, d_pkg_pipe, d_w_pipe
        ])
        return ds

    # --- initial state ---
    def initial_state(self):
        p = self.p
        # seed pipelines at steady-state-ish values to avoid startup transient
        fab_ss = p.base_invest * (p.tau_fab / p.n_fab)
        pkg_ss = p.pkg_base_invest * (p.tau_pkg_base / p.n_pkg)
        w_ss = 2.0 * (p.tau_w / p.n_w)
        s0 = np.concatenate([
            [p.K_fab0, p.K_pkg0, p.E0, p.W0, p.util_target],
            np.full(p.n_fab, fab_ss),
            np.full(p.n_pkg, pkg_ss),
            np.full(p.n_w, w_ss),
        ])
        return s0

    # --- RK4 integration ---
    def simulate(self, to_horizon=False):
        p = self.p
        end = p.horizon_end if to_horizon else p.t_end
        n_steps = int(round((end - p.t0) / p.dt))
        t = p.t0
        s = self.initial_state()

        times = [t]
        states = [s.copy()]
        outputs = [self._record_outputs(t, s)]

        for _ in range(n_steps):
            k1 = self.derivs(t, s)
            k2 = self.derivs(t + p.dt / 2, s + p.dt / 2 * k1)
            k3 = self.derivs(t + p.dt / 2, s + p.dt / 2 * k2)
            k4 = self.derivs(t + p.dt, s + p.dt * k3)
            s = s + (p.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            s[:4] = np.maximum(s[:4], 1e-6)   # keep stocks non-negative
            t += p.dt
            times.append(t)
            states.append(s.copy())
            outputs.append(self._record_outputs(t, s))

        return self._package(times, states, outputs)

    def _record_outputs(self, t, s):
        p = self.p
        K_fab, K_pkg, E, W = s[0], s[1], s[2], s[3]
        K_fab = max(K_fab, 1e-6)
        pa = self.policy_active(t)
        Y = self.yield_fn(E)
        Y_eff = np.clip(Y + p.wf_yield_lift * p.p_workforce * pa, 0.0, 0.99)
        w_adequacy = np.clip(W / (p.w_target_ratio * K_fab), 0.0, 1.0)
        U = (p.util_target + p.cap_util_lift * p.p_capital * pa) * w_adequacy
        U = np.clip(U, 0.0, 1.0)
        Q_fab = Y_eff * U * K_fab
        a_eq_eff = np.clip(p.a_eq + p.eco_equip_strength * p.p_ecosystem * pa, 0.0, 0.99)
        Q_equip = a_eq_eff * K_fab
        a_pkg = 0.62 + p.eco_pkg_throughput * p.p_ecosystem * pa
        Q_pkg = a_pkg * K_pkg
        Q_eff = min(Q_fab, Q_equip, Q_pkg)
        # identify binding constraint
        vals = {'fab': Q_fab, 'equipment': Q_equip, 'packaging': Q_pkg}
        binding = min(vals, key=vals.get)
        return dict(Y=Y, U=U, Q_fab=Q_fab, Q_equip=Q_equip, Q_pkg=Q_pkg,
                    Q_eff=Q_eff, binding=binding, demand=self.demand(t))

    def _package(self, times, states, outputs):
        times = np.array(times)
        states = np.array(states)
        out = {k: np.array([o[k] for o in outputs]) if k != 'binding'
               else [o['binding'] for o in outputs]
               for k in outputs[0]}
        result = {
            't': times,
            'K_fab': states[:, 0],
            'K_pkg': states[:, 1],
            'E': states[:, 2],
            'W': states[:, 3],
        }
        result.update(out)
        return result


# ----------------------------------------------------------------------------
# Scenario definitions
# ----------------------------------------------------------------------------
def make_scenarios():
    base = Params()

    baseline = Params(**asdict(base))
    baseline.p_capital = 0.0
    baseline.p_workforce = 0.0
    baseline.p_ecosystem = 0.0

    capital_only = Params(**asdict(base))
    capital_only.p_capital = 1.0
    capital_only.p_workforce = 0.0
    capital_only.p_ecosystem = 0.0

    capital_workforce = Params(**asdict(base))
    capital_workforce.p_capital = 1.0
    capital_workforce.p_workforce = 1.0
    capital_workforce.p_ecosystem = 0.0

    ecosystem = Params(**asdict(base))
    ecosystem.p_capital = 1.0
    ecosystem.p_workforce = 1.0
    ecosystem.p_ecosystem = 1.0

    return {
        'Baseline': baseline,
        'Capital-only': capital_only,
        'Capital+Workforce': capital_workforce,
        'Ecosystem-aligned': ecosystem,
    }


if __name__ == '__main__':
    scenarios = make_scenarios()
    print("=== Effective output index (1990 = 100), historical horizon ===")
    for name, p in scenarios.items():
        model = SemiconductorSD(p)
        r = model.simulate()
        q0 = r['Q_eff'][0]
        idx = lambda yr: r['Q_eff'][np.argmin(np.abs(r['t'] - yr))] / q0 * 100
        print(f"{name:20s} | 2000={idx(2000):6.1f} 2010={idx(2010):6.1f} "
              f"2020={idx(2020):6.1f} 2025={idx(2025):6.1f}")

    print("\n=== Forward horizon (policy onset 2022 -> +20yr), index 2022 = 100 ===")
    for name, p in scenarios.items():
        model = SemiconductorSD(p)
        r = model.simulate(to_horizon=True)
        i22 = np.argmin(np.abs(r['t'] - 2022))
        q22 = r['Q_eff'][i22]
        idx = lambda yr: r['Q_eff'][np.argmin(np.abs(r['t'] - yr))] / q22 * 100
        print(f"{name:20s} | 2027={idx(2027):6.1f} 2032={idx(2032):6.1f} "
              f"2037={idx(2037):6.1f} 2042={idx(2042):6.1f}")

    print("\n=== Binding constraint over time (baseline) ===")
    model = SemiconductorSD(scenarios['Baseline'])
    r = model.simulate(to_horizon=True)
    for yr in [1995, 2005, 2015, 2022, 2025, 2030, 2035]:
        i = np.argmin(np.abs(r['t'] - yr))
        print(f"  {yr}: binding = {r['binding'][i]:12s}  Y={r['Y'][i]:.3f}  "
              f"Kfab={r['K_fab'][i]:6.1f}  Kpkg={r['K_pkg'][i]:6.1f}")
