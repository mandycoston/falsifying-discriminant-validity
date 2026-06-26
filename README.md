<h1 align="center">Falsifying Discriminant Validity of Predictive Algorithms</h1>
<h3 align="center">
<a href="https://dl.acm.org/doi/10.1145/3805689.3812310">📄 Paper</a>
&nbsp;•&nbsp;
<a href="https://arxiv.org/pdf/2601.17146">
  <sub>
    <img
      width="17"
      alt="arxiv"
      src="https://github.com/user-attachments/assets/42d07f6e-46b8-4570-a0f5-b3588a2bfc83"
    />
  </sub>
  ArXiv
</a>
</h3>

This directory contains the code to replicate every empirical result (figures,
tables, and reported statistics) in the **main paper** and the **appendix**.

The analysis covers two datasets — **LSAC** (Law School Admission Council) and
**COMPAS** — and the following models: an unconstrained **logistic regression**,
two **LAFTR** representations (one with a race adversary, one with a gender
adversary), the proprietary **COMPAS decile risk scores**, and a logistic
regression trained on COMPAS two-year recidivism.

## 1. Requirements

### R packages
```r
install.packages(c("tidyverse", "glmnet", "caret", "boot"))
```

### Python (only needed for the LAFTR models)
```bash
pip install -r requirements_laftr.txt
```
(torch, numpy, pandas, scikit-learn.)

## 2. Data

No data files are shipped with this release; they are downloaded at runtime.

- **LSAC**: downloaded automatically from
  `https://raw.githubusercontent.com/damtharvey/law-school-dataset/master/law_dataset.csv`
  by both the R (`load_lsac_data.R`) and Python scripts. Requires internet access.
- **COMPAS**: ProPublica's data. The R COMPAS scripts will `git clone`
  `https://github.com/propublica/compas-analysis.git` into a `compas-analysis/`
  subdirectory automatically the first time they run.
  - The Python script (`train_laftr_models.py`) does **not** clone; it expects
    `compas-analysis/compas-scores-two-years.csv` to already exist. Run one R
    COMPAS script first (e.g. `run_compas_existing_scores.R`), or clone manually:
    ```bash
    git clone https://github.com/propublica/compas-analysis.git
    ```

## 3. File inventory

### Core library (sourced by everything)
| File | Contents |
|------|----------|
| `falsification_methods.R` | `platt_scaling()`, `log_loss()`, `compute_loss()`, `falsify_single_proxy()` (Algorithm 1), `falsify_multiple_proxy()` (Algorithm 2) |
| `load_lsac_data.R` | Loads and prepares the LSAC dataset (downloads from GitHub) |

### LSAC analysis
| File | Produces |
|------|----------|
| `run_lsac.R` | Logistic-regression falsification (gender & race p-values); LSAC performance-metrics table; the logistic-regression rows of the Platt-scaling ablation |
| `train_laftr_models.py` | Trains LAFTR (race adversary) → `predictions_lsac_laftr.csv` (also writes `predictions_compas_laftr.csv`, unused here) |
| `compute_laftr_table1_metrics.py` | Trains LAFTR (gender adversary) → `predictions_lsac_laftr_gender.csv`; LAFTR performance-metrics table |
| `run_lsac_laftr.R` | LAFTR (race adversary) falsification test |
| `run_lsac_laftr_gender.R` | LAFTR (gender adversary) falsification test |
| `compute_correlation_tables.R` | Pearson / Spearman / Kendall correlation tables (LSAC and COMPAS) |
| `plot_rank_distribution.R` | **Main-paper figure**: rank distribution of impermissible proxies |

### COMPAS analysis
| File | Produces |
|------|----------|
| `run_compas_existing_scores.R` | COMPAS decile-score falsification (age < 25); COMPAS performance-metrics table |
| `run_compas_existing_scores_race.R` | COMPAS decile-score falsification (race) |
| `run_compas.R` | Logistic regression trained on two-year recidivism (appendix) |
| `plot_compas_paired_diff.R` | **Appendix figure**: paired-difference distributions (age & race) |

### Appendix robustness / ablations
| File | Produces |
|------|----------|
| `run_lsac_negative_control.R` | Negative-control check |
| `run_lsac_family_income.R` | Family income as an alternative impermissible proxy |
| `ablation_laftr_platt.R` | Platt-scaling ablation (LAFTR rows) |
| `ablation_loss_function.R` | Log-loss vs. Brier-score ablation |

### Appendix bar-passage table (manual one-line change)
The appendix table reporting performance metrics for a model trained to predict
**bar passage** instead of first-year GPA does not have its own script. To
reproduce it, edit `run_lsac.R` so the model is fit on `pass_bar` instead of
`fygpa_binary` — change the `glm(...)` call (around line 87) from
`glm(fygpa_binary ~ lsat + ugpa, ...)` to `glm(pass_bar ~ lsat + ugpa, ...)` —
and re-run the script. All other tables come from the scripts as-is.

> Note: the figure scripts write to `../figures/` (a `figures/` directory beside
> the script folder). Create that directory, or edit the `png(...)` path in
> `plot_rank_distribution.R` and `plot_compas_paired_diff.R`, before running them.

## 4. Run order

Set your working directory to this folder. Several steps share a single
**train/calibration/evaluation split (50/25/25, `set.seed(42)`)**. The R scripts
regenerate that split deterministically; the Python scripts read it from
`lsac_fairlearn_indices.csv`. **You must create that index file with an R script
before running the Python scripts**, otherwise Python falls back to a different
split and the LAFTR predictions will be misaligned with the R evaluation set
(producing silently incorrect results).

### LSAC pipeline (LAFTR results)
```r
# Step 1 — write the shared split, then stop (predictions not yet available)
source("run_lsac_laftr.R")        # creates lsac_fairlearn_indices.csv, then stops
```
```bash
# Step 2 — train LAFTR models using that split
python train_laftr_models.py            # writes predictions_lsac_laftr.csv
python compute_laftr_table1_metrics.py  # writes predictions_lsac_laftr_gender.csv + metrics
```
```r
# Step 3 — run the LAFTR falsification tests and correlations
source("run_lsac_laftr.R")        # now runs the full test
source("run_lsac_laftr_gender.R")
source("compute_correlation_tables.R")
```

### LSAC pipeline (logistic regression + appendix)
```r
source("run_lsac.R")                 # main LR falsification + metrics tables
source("run_lsac_negative_control.R")
source("run_lsac_family_income.R")
source("ablation_laftr_platt.R")    # needs the LAFTR prediction CSVs from above
source("ablation_loss_function.R")
```

### COMPAS pipeline
```r
source("run_compas_existing_scores.R")       # also auto-clones ProPublica data
source("run_compas_existing_scores_race.R")
source("run_compas.R")
```

### Figures
```r
source("plot_rank_distribution.R")    # main-paper figure
source("plot_compas_paired_diff.R")   # appendix figure
```

## 5. Notes on reproducibility

- All splits use `set.seed(42)` and a 50% train / 25% calibration / 25%
  evaluation partition; the Python LAFTR scripts reuse the R split via
  `lsac_fairlearn_indices.csv`.
- Platt scaling is applied per-outcome before computing losses; this calibration
  step is essential — without it, some tests flip qualitative conclusions (see
  the Platt-scaling ablation).
- Algorithm 1 (`falsify_single_proxy`) is used when there is a single permissible
  proxy; Algorithm 2 (`falsify_multiple_proxy`) is used with multiple permissible
  proxies and reports the rank-based test visualized by `plot_rank_distribution.R`.

## Citation

If you use this code, please cite the paper as follows:

<pre style="white-space: pre; overflow-x: auto;">
@inproceedings{coston2026falsifying,
  title = {Falsifying Discriminant Validity of Predictive Algorithms},
  author = {Coston, Amanda Lee},
  year = {2026},
  isbn = {9798400725968},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3805689.3812310},
  doi = {10.1145/3805689.3812310},
  booktitle = {Proceedings of the 2026 ACM Conference on Fairness, Accountability, and Transparency},
  pages = {3105–3128},
  numpages = {24},
  series = {FAccT '26}
}
</pre>
