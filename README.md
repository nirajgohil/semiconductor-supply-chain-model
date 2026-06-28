# Dynamic Modeling of the U.S. Semiconductor Supply Chain

Replication code and data for the paper:

**"Dynamic Modeling of the U.S. Semiconductor Supply Chain: Advanced Packaging as the
Binding Constraint and the Limits of Capacity-Only Policy"**
Niraj Gohil, 2026.

A continuous-time system dynamics model of advanced-node semiconductor supply (1990–2042)
in which effective output is governed by a binding-constraint (minimum) operator across
fabrication, equipment, and advanced packaging, with capacity forming through multi-stage
(Erlang) delay pipelines and endogenous yield learning.

---

## Key findings

1. **The binding constraint migrates endogenously** from yield-limited fabrication in the
   1990s to advanced packaging in the post-2015 AI era, reproducing the observed CoWoS
   bottleneck.
2. **Capital-only policy is ineffective under a packaging-bound regime:** capital-only and
   capital-plus-workforce policy produce 0% change in effective output at 5/10/20 years;
   only ecosystem-aligned policy that relieves packaging moves output (+215% at 20 years).
   This holds in 100% of sampled parameter draws where packaging binds.
3. **Long-run output is dominated by the fabrication depreciation rate** (Sobol' total-order
   index S_T = 0.76), not learning speed or demand growth.

All reported magnitudes are scenario indices under documented assumptions, not point
forecasts — the contribution is structural.

---

## Files

**Model code**
- `sd_model.py` — core system dynamics model (RK4 integrator, Erlang delays, yield
  learning, binding-constraint min-operator, 4 policy scenarios)
- `validation.py` — 9-test validation battery (all pass)
- `sensitivity.py` — global Sobol' variance-based sensitivity analysis (SALib)
- `robustness.py` — Latin-hypercube robustness test of the headline conclusions
- `realdata_validation.py` — comparison of model dynamics against real FRED series
- `make_figures.py` — regenerates all figures
- `extract_results.py` — dumps every number used in the paper's tables and text

**Data** (public domain, from FRED — Federal Reserve Bank of St. Louis)
- `IPG3344S.csv` — US semiconductor & electronic-component production index
- `CAPG3344S.csv` — companion capacity index
- `IPG3344SQ.csv` — production index, quarterly

**Figures** — `fig_*.pdf`, the 9 figures from the paper.

---

## Reproduce the results

Requires Python 3.10+.

```bash
pip install numpy scipy matplotlib SALib

python validation.py          # 9-test validation battery (all PASS)
python sd_model.py            # baseline + 4 policy scenarios
python sensitivity.py        # Sobol' sensitivity indices (~2-4 min)
python robustness.py         # Latin-hypercube robustness of the conclusions
python make_figures.py        # regenerates all figures
python extract_results.py     # writes every paper number to results_dump.json
python realdata_validation.py # model vs. real FRED series
```

> Note on `realdata_validation.py`: this compares the model's normalized output against the
> real FRED production/capacity indices. The model is a *normalized structural* index, not a
> forecast of physical output volume, so the two are not on a common scale (real US
> production grew ~560x over 1990–2025 in the Moore's-law era; the model's normalized index
> grows ~3x). The comparison is therefore of *dynamics and direction*, not levels — see the
> "Relationship to Real Aggregate Data" section of the paper. The real data corroborates the
> paper's qualitative claims (the offshoring dynamic, and declining implied utilization from
> 0.89 in 2000 to 0.76 in 2025) without being used as a statistical fit.

---

## Data sources

All data is public domain, from FRED (Federal Reserve Bank of St. Louis):
- [IPG3344S](https://fred.stlouisfed.org/series/IPG3344S) — Industrial Production:
  Semiconductor and Other Electronic Component (NAICS 3344)
- [CAPG3344S](https://fred.stlouisfed.org/series/CAPG3344S) — Industrial Capacity, same sector

---

## License

MIT License — see `LICENSE`.

## Citation

If you use this code or model, please cite the paper (arXiv reference to be added once posted).
