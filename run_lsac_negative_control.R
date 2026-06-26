# Negative Control Test (Appendix)
#
# Validation of the falsification procedure: treat cumulative GPA
# (a permissible proxy) as if it were impermissible, with first-year GPA
# as the permissible proxy. A correctly functioning procedure should
# return INDISCRIMINANT.

source("falsification_methods.R")
source("load_lsac_data.R")

set.seed(42)

# ============================================================================
# Load and split data (same split as main LSAC analysis)
# ============================================================================

lsac_data <- load_lsac_data()
n <- nrow(lsac_data)
all_indices <- sample(n)
train_size <- floor(n * 0.5)
calib_size <- floor(n * 0.25)
train_indices <- all_indices[1:train_size]
calib_indices <- all_indices[(train_size + 1):(train_size + calib_size)]
eval_indices  <- all_indices[(train_size + calib_size + 1):n]

train_data <- lsac_data[train_indices, ]
calib_eval_data <- lsac_data[c(calib_indices, eval_indices), ]

# Train logistic regression (same as main analysis)
model <- glm(fygpa_binary ~ lsat + ugpa, data = train_data, family = binomial,
             control = list(maxit = 1000))
predictions_raw <- predict(model, newdata = calib_eval_data, type = "response")

calib_rel <- 1:length(calib_indices)
eval_rel  <- (length(calib_indices) + 1):nrow(calib_eval_data)

# ============================================================================
# Negative control: cumulative GPA as "impermissible", first-year GPA as
# permissible. Both are permissible proxies, so the test should return
# INDISCRIMINANT.
# ============================================================================

cat(paste0(rep("=", 70), collapse = ""), "\n")
cat("NEGATIVE CONTROL TEST\n")
cat("  'Impermissible' proxy: cumulative GPA (gpa_binary)\n")
cat("  Permissible proxy:     first-year GPA (fygpa_binary)\n")
cat(paste0(rep("=", 70), collapse = ""), "\n\n")

impermissible_labels <- calib_eval_data$gpa_binary
permissible_labels   <- calib_eval_data$fygpa_binary

result <- falsify_single_proxy(
  predictions          = predictions_raw,
  impermissible_labels = impermissible_labels,
  permissible_labels   = permissible_labels,
  alpha = 0.05,
  calib_indices = calib_rel,
  eval_indices  = eval_rel
)

cat("Result:", result$result, "\n")
cat("P-value:", format_pvalue(result$p_value, z = result$z_statistic), "\n")
cat("Test used:", result$test_used, "\n")
cat("Mean loss difference:", sprintf("%.4f", result$delta_mean), "\n\n")

if (grepl("INDISCRIMINANT", result$result)) {
  cat("Negative control PASSED: procedure correctly returns INDISCRIMINANT\n")
  cat("when a permissible proxy is treated as impermissible.\n")
} else {
  cat("WARNING: Negative control FAILED — expected INDISCRIMINANT.\n")
}

# Save result
write.csv(data.frame(
  Test = "Negative control",
  Impermissible = "gpa_binary",
  Permissible   = "fygpa_binary",
  P_Value       = result$p_value,
  Mean_Diff     = result$delta_mean,
  Result        = result$result,
  stringsAsFactors = FALSE
), "lsac_negative_control_results.csv", row.names = FALSE)
cat("\nResults saved to lsac_negative_control_results.csv\n")
