# Dynamic Modeling of the U.S. Semiconductor Supply Chain — FINAL PACKAGE

**Title:** Dynamic Modeling of the U.S. Semiconductor Supply Chain: Advanced Packaging
as the Binding Constraint and the Limits of Capacity-Only Policy

**Author:** Niraj Gohil  |  **Contact:** nirajgohil60@gmail.com

Complete, final deliverable. 11-page IEEE-format paper plus full replication code and data.
Everything is current: final title, neutral preprint header, real email, fixed
indicator-function rendering, clean non-overlapping figures, impersonal voice, the honest
real-data validation section, and all reviewer corrections applied.

---

## Folder layout

### 01_arxiv_submission/  ← upload THIS to arXiv
- **arxiv_submission.tar.gz** — the file you upload to arXiv. Flattened LaTeX source with
  name-matched main.bbl and all 9 figures. Verified to recompile cleanly (11 pages).
- **arxiv_abstract.txt** — paste into arXiv's abstract field (under the 1920-char limit).
- **ARXIV_SUBMISSION_GUIDE.md** — categories (eess.SY primary, econ.GN cross-list),
  endorsement status (already cleared via your two eess.SY papers), checklist.
- main.tex, main.bbl, references.bib — loose source (also inside the tarball).

### 02_paper/
- **main.pdf** — the compiled 11-page paper (read / share / journal submission).
- main.tex, references.bib — LaTeX source.

### 03_model_code/  ← replication code that produced every number and figure
- sd_model.py, validation.py, sensitivity.py, robustness.py, make_figures.py,
  extract_results.py, realdata_validation.py
- IPG3344S.csv, CAPG3344S.csv, IPG3344SQ.csv — real FRED data
- cached results: results_dump.json, sobol_results.npz, robustness_results.npz,
  realdata_metrics.json

### 04_figures/
- All 9 publication figures as vector PDFs.

### 05_github_repo/  ← ready to publish on GitHub
- A clean, repo-ready copy (model/, data/, figures/, paper/, README.md, LICENSE,
  requirements.txt, .gitignore) plus **github_repo.zip**.
- To publish: github.com -> New repository (Public, add README) -> Add file -> Upload
  files -> drag in the contents of github_repo.zip -> Commit.
- After it is live, the paper's Replication section can be updated to link to it.

---

## To submit to arXiv (quick version)
1. arxiv.org -> Submit -> upload `01_arxiv_submission/arxiv_submission.tar.gz`
2. Confirm the preview compiles (11 pages, figures + bibliography render)
3. Primary category **eess.SY**, cross-list **econ.GN**
4. Paste the title and the contents of `arxiv_abstract.txt`
5. Set license, proofread metadata, Submit

## To reproduce the results
```
cd 03_model_code
pip install numpy scipy matplotlib SALib
python validation.py        # 9/9 tests pass
python sd_model.py          # baseline + scenarios
python sensitivity.py       # Sobol indices
python robustness.py        # robustness analysis
python make_figures.py      # regenerate all figures
python realdata_validation.py  # model vs. real FRED series
```

## Key findings
1. The binding constraint migrates endogenously from yield-limited fabrication (1990s) to
   advanced packaging (post-2015 AI era), reproducing the observed CoWoS bottleneck.
2. Under a packaging-bound regime, capital-only and capital+workforce policy yield 0%
   effective-output gain at 5/10/20 years; only ecosystem-aligned policy moves output
   (+215% at 20 years). Holds in 100% of sampled draws where packaging binds.
3. Long-run output is dominated by the fabrication depreciation rate (Sobol' S_T = 0.76).

All magnitudes are scenario indices under documented assumptions, not point forecasts.
