"""
Real-data validation: compare model output against actual FRED semiconductor series.

Data sources (download as CSV from FRED, public domain):
  IPG3344S  - Industrial Production: Semiconductor & Other Electronic Component
              https://fred.stlouisfed.org/series/IPG3344S
              -> validation target for MODEL EFFECTIVE OUTPUT (shape/dynamics)
  CAPG3344S - Industrial Capacity: same sector
              https://fred.stlouisfed.org/series/CAPG3344S
              -> validation target for MODEL CAPACITY CEILING

Place the downloaded CSVs next to this script as:
  IPG3344S.csv
  CAPG3344S.csv
(FRED CSVs have columns like: DATE,IPG3344S  -- the loader auto-detects the value column.)

What this does:
  1. Loads the FRED monthly series, aggregates to annual means.
  2. Runs the model over the same calendar years.
  3. Indexes both model and real series to a common base year so they are comparable
     (the model is a normalized index; FRED is its own index -- neither is in physical
     units, so comparison is of SHAPE/DYNAMICS, not levels).
  4. Reports fit metrics: Pearson correlation, R^2, RMSE on the indexed series, and
     correlation of year-over-year growth rates (the most honest test of whether the
     model reproduces the real dynamics rather than just a shared trend).
  5. Writes a comparison figure to ../figures/fig_realdata_validation.pdf
"""
import os
import numpy as np

CSV_DIR = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(CSV_DIR, '..', 'figures')

# CSVs live in the repo's data/ directory (one level up from model/, then into data/).
# Fall back to the script's own directory if a data/ folder is not present.
_DATA_DIR = os.path.join(CSV_DIR, '..', 'data')
if not os.path.isdir(_DATA_DIR):
    _DATA_DIR = CSV_DIR

PROD_CSV = os.path.join(_DATA_DIR, 'IPG3344S.csv')     # production -> effective output
CAP_CSV  = os.path.join(_DATA_DIR, 'CAPG3344S.csv')    # capacity  -> capacity ceiling

BASE_YEAR = 1992   # common index base; pick an early year both series cover


# ----------------------------------------------------------------------------
# CSV loading (robust to FRED's format variations)
# ----------------------------------------------------------------------------
def load_fred_csv(path):
    """Load a FRED CSV -> dict {year: annual_mean_value}. Auto-detects columns.
    FRED format is typically: DATE,<SERIES_ID>  with monthly rows YYYY-MM-DD.
    Missing values appear as '.' and are skipped."""
    import csv
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  MISSING: {os.path.basename(path)}\n"
            f"  Download it from FRED and place it next to this script:\n"
            f"    IPG3344S  -> https://fred.stlouisfed.org/series/IPG3344S\n"
            f"    CAPG3344S -> https://fred.stlouisfed.org/series/CAPG3344S\n"
            f"  (click Download -> CSV)\n")
    monthly = {}  # year -> list of values
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        # find the date column and the value column
        date_col = 0
        # value column = the first column whose header is not obviously a date label
        val_col = 1 if len(header) > 1 else 0
        for row in reader:
            if not row or len(row) <= val_col:
                continue
            datestr = row[date_col].strip()
            valstr = row[val_col].strip()
            if valstr in ('.', '', 'NA', 'NaN'):
                continue
            try:
                year = int(datestr[:4])
                val = float(valstr)
            except ValueError:
                continue
            monthly.setdefault(year, []).append(val)
    # annual means
    annual = {yr: float(np.mean(v)) for yr, v in monthly.items() if v}
    return annual


def index_to_base(series_by_year, base_year):
    """Return (years_array, indexed_values_array) with base_year = 100."""
    years = np.array(sorted(series_by_year.keys()))
    vals = np.array([series_by_year[y] for y in years], dtype=float)
    if base_year in series_by_year:
        base = series_by_year[base_year]
    else:
        base = vals[0]
    return years, vals / base * 100.0


# ----------------------------------------------------------------------------
# Fit metrics
# ----------------------------------------------------------------------------
def fit_metrics(model_vals, real_vals):
    """Compute correlation, R^2, RMSE on aligned, indexed series."""
    m = np.asarray(model_vals, float)
    r = np.asarray(real_vals, float)
    # Pearson correlation
    corr = float(np.corrcoef(m, r)[0, 1])
    # R^2 of real vs model (how much of real variance the model tracks)
    ss_res = np.sum((r - m) ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(np.mean((r - m) ** 2)))
    return corr, r2, rmse


def growth_corr(years, model_vals, real_vals):
    """Correlation of year-over-year growth rates -- the honest dynamics test."""
    m = np.asarray(model_vals, float)
    r = np.asarray(real_vals, float)
    gm = np.diff(m) / m[:-1]
    gr = np.diff(r) / r[:-1]
    if len(gm) < 2:
        return float('nan')
    return float(np.corrcoef(gm, gr)[0, 1])


# ----------------------------------------------------------------------------
# Main comparison
# ----------------------------------------------------------------------------
def run():
    from sd_model import Params, SemiconductorSD

    # --- load real data ---
    prod_real = load_fred_csv(PROD_CSV)
    cap_real = load_fred_csv(CAP_CSV)
    print(f"Loaded IPG3344S (production): {min(prod_real)}-{max(prod_real)}, "
          f"{len(prod_real)} annual points")
    print(f"Loaded CAPG3344S (capacity): {min(cap_real)}-{max(cap_real)}, "
          f"{len(cap_real)} annual points")

    # --- run model over the calendar (historical horizon) ---
    p = Params()
    r = SemiconductorSD(p).simulate()  # 1990-2025

    def model_annual(series_key):
        out = {}
        for yr in range(int(p.t0), int(p.t_end) + 1):
            idx = int(np.argmin(np.abs(r['t'] - yr)))
            out[yr] = float(r[series_key][idx])
        return out

    model_output = model_annual('Q_eff')   # effective output
    model_capacity = {yr: (p.y_max * 1.0 * v) for yr, v in model_annual('K_fab').items()}

    # --- align on overlapping years ---
    def align(model_by_year, real_by_year):
        common = sorted(set(model_by_year) & set(real_by_year))
        common = [y for y in common if y >= BASE_YEAR]  # start at common base
        ym, m_idx = index_to_base({y: model_by_year[y] for y in common}, BASE_YEAR)
        yr_, r_idx = index_to_base({y: real_by_year[y] for y in common}, BASE_YEAR)
        return np.array(common), m_idx, r_idx

    yrs_o, m_out, r_out = align(model_output, prod_real)
    yrs_c, m_cap, r_cap = align(model_capacity, cap_real)

    # --- metrics ---
    print("\n=== PRODUCTION / EFFECTIVE OUTPUT (model vs IPG3344S) ===")
    c, r2, rmse = fit_metrics(m_out, r_out)
    gc = growth_corr(yrs_o, m_out, r_out)
    print(f"  years {yrs_o[0]}-{yrs_o[-1]}  (n={len(yrs_o)})")
    print(f"  level correlation : {c:+.3f}")
    print(f"  R^2 (real vs model): {r2:+.3f}")
    print(f"  RMSE (index pts)  : {rmse:.1f}")
    print(f"  YoY growth corr   : {gc:+.3f}")

    print("\n=== CAPACITY (model ceiling vs CAPG3344S) ===")
    cc, cr2, crmse = fit_metrics(m_cap, r_cap)
    cgc = growth_corr(yrs_c, m_cap, r_cap)
    print(f"  years {yrs_c[0]}-{yrs_c[-1]}  (n={len(yrs_c)})")
    print(f"  level correlation : {cc:+.3f}")
    print(f"  R^2 (real vs model): {cr2:+.3f}")
    print(f"  RMSE (index pts)  : {crmse:.1f}")
    print(f"  YoY growth corr   : {cgc:+.3f}")

    # --- save metrics for the manuscript ---
    import json
    metrics = {
        'production': {'years': [int(yrs_o[0]), int(yrs_o[-1])], 'n': len(yrs_o),
                       'corr': round(c, 3), 'r2': round(r2, 3),
                       'rmse': round(rmse, 1), 'growth_corr': round(gc, 3)},
        'capacity': {'years': [int(yrs_c[0]), int(yrs_c[-1])], 'n': len(yrs_c),
                     'corr': round(cc, 3), 'r2': round(cr2, 3),
                     'rmse': round(crmse, 1), 'growth_corr': round(cgc, 3)},
        'base_year': BASE_YEAR,
    }
    with open(os.path.join(CSV_DIR, 'realdata_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved realdata_metrics.json")

    # --- figure ---
    make_figure(yrs_o, m_out, r_out, yrs_c, m_cap, r_cap, metrics)
    return metrics


def make_figure(yrs_o, m_out, r_out, yrs_c, m_cap, r_cap, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10,
        'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 8.5,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 0.8,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    })
    C = {'model': '#4878A6', 'real': '#B5485D'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2))

    ax1.plot(yrs_o, r_out, color=C['real'], lw=2.0, marker='o', ms=2.5,
             label='Real (FRED IPG3344S)')
    ax1.plot(yrs_o, m_out, color=C['model'], lw=2.0, ls='--',
             label='Model effective output')
    ax1.set_title('Production / effective output', fontsize=10)
    ax1.set_xlabel('Year'); ax1.set_ylabel(f'Index ({metrics["base_year"]} = 100)')
    ax1.legend(frameon=False, loc='upper left')
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.text(0.97, 0.05,
             f"r = {metrics['production']['corr']:+.2f}\n"
             f"growth r = {metrics['production']['growth_corr']:+.2f}",
             transform=ax1.transAxes, fontsize=8, ha='right', va='bottom',
             bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.9))

    ax2.plot(yrs_c, r_cap, color=C['real'], lw=2.0, marker='o', ms=2.5,
             label='Real (FRED CAPG3344S)')
    ax2.plot(yrs_c, m_cap, color=C['model'], lw=2.0, ls='--',
             label='Model capacity ceiling')
    ax2.set_title('Capacity', fontsize=10)
    ax2.set_xlabel('Year'); ax2.set_ylabel(f'Index ({metrics["base_year"]} = 100)')
    ax2.legend(frameon=False, loc='upper left')
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.text(0.97, 0.05,
             f"r = {metrics['capacity']['corr']:+.2f}\n"
             f"growth r = {metrics['capacity']['growth_corr']:+.2f}",
             transform=ax2.transAxes, fontsize=8, ha='right', va='bottom',
             bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.9))

    fig.suptitle('Model output vs. real U.S. semiconductor series (FRED, public domain)',
                 fontsize=9.5, y=1.03)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'fig_realdata_validation.pdf'))
    plt.close(fig)
    print("saved fig_realdata_validation.pdf")


if __name__ == '__main__':
    run()
