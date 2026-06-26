# COMPAS Paired Difference Histograms (Appendix Figure)
#
# Plots the distribution of (re-arrest loss - impermissible proxy loss)
# for age (left) and race (right) using Algorithm 1 on COMPAS decile scores.

source("falsification_methods.R")

set.seed(42)

# ============================================================================
# Load COMPAS data
# ============================================================================

repo_dir <- "compas-analysis"
data_file <- file.path(repo_dir, "compas-scores-two-years.csv")
if (!file.exists(data_file)) stop("COMPAS data not found at ", data_file)
compas_data <- read.csv(data_file, stringsAsFactors = FALSE)

compas_data$two_year_recid <- as.numeric(compas_data$two_year_recid)
compas_data$age_lt25 <- as.numeric(compas_data$age < 25)
compas_data$race_black <- as.numeric(
  grepl("African|Black", compas_data$race, ignore.case = TRUE))

# Convert decile scores to [0,1]
compas_scores <- (compas_data$decile_score - 1) / 9
compas_scores <- pmax(pmin(compas_scores, 1), 0)

keep <- complete.cases(compas_data[, c("two_year_recid", "age_lt25",
                                        "race_black", "decile_score")])
compas_data <- compas_data[keep, ]
compas_scores <- compas_scores[keep]

# Split: 20% calibration, 80% evaluation
n <- nrow(compas_data)
calib_size <- floor(n * 0.2)
calib_idx <- sample(n, calib_size)
eval_idx  <- setdiff(1:n, calib_idx)

# ============================================================================
# Compute paired loss differences for each impermissible proxy
# ============================================================================

compute_paired_diffs <- function(predictions, imp_labels, perm_labels,
                                  calib_idx, eval_idx) {
  platt_imp  <- platt_scaling(predictions, imp_labels, calib_idx)
  platt_perm <- platt_scaling(predictions, perm_labels, calib_idx)

  loss_imp  <- log_loss(platt_imp$calibrated[eval_idx],
                        imp_labels[eval_idx])
  loss_perm <- log_loss(platt_perm$calibrated[eval_idx],
                        perm_labels[eval_idx])

  # Delta = permissible loss - impermissible loss
  # (re-arrest loss - impermissible proxy loss)
  loss_perm - loss_imp
}

delta_age <- compute_paired_diffs(compas_scores, compas_data$age_lt25,
                                   compas_data$two_year_recid,
                                   calib_idx, eval_idx)

delta_race <- compute_paired_diffs(compas_scores, compas_data$race_black,
                                    compas_data$two_year_recid,
                                    calib_idx, eval_idx)

# ============================================================================
# Compute p-values for annotation
# ============================================================================

# For age: test Delta_i = L_imp - L_perm > 0 (i.e., -delta_age here)
r_age  <- wilcox.test(-delta_age, alternative = "greater", mu = 0)
r_race <- wilcox.test(-delta_race, alternative = "greater", mu = 0)

# ============================================================================
# Plot
# ============================================================================

png("../figures/compas_paired_diff_combined.png",
    width = 1200, height = 500, res = 150)
par(mfrow = c(1, 2), mar = c(5, 5, 3, 1), cex.axis = 1.2, cex.lab = 1.3,
    cex.main = 1.4)

# Left: Age
hist(delta_age, breaks = 40, col = "steelblue", border = "white",
     main = "Age (Impermissible)",
     xlab = "Re-arrest loss \u2212 Age loss",
     ylab = "Frequency")
abline(v = 0, lty = 2, lwd = 2)
mtext(sprintf("p \u2248 %.3f", r_age$p.value), side = 3, line = -1.5,
      adj = 0.95, cex = 1.1)

# Right: Race
hist(delta_race, breaks = 40, col = "indianred", border = "white",
     main = "Race (Impermissible)",
     xlab = "Re-arrest loss \u2212 Race loss",
     ylab = "Frequency")
abline(v = 0, lty = 2, lwd = 2)
mtext(sprintf("p = %.3f", r_race$p.value), side = 3, line = -1.5,
      adj = 0.95, cex = 1.1)

dev.off()
cat("Figure saved to ../figures/compas_paired_diff_combined.png\n")
