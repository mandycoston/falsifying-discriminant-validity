# LSAC: Family Income as Impermissible Proxy (Appendix)
#
# The paper says: "Family income in the LSAC dataset is encoded as an ordinal
# variable with categories 1-5. We binarize this variable by defining high
# income as the highest category (category 5), which comprises approximately
# 8% of applicants."
#
# Tests Algorithm 2 (multiple permissible proxies) with family income as
# the impermissible proxy.

source("falsification_methods.R")
source("load_lsac_data.R")

set.seed(42)

# ============================================================================
# Load and split data (same as main LSAC analysis)
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
# Create family income impermissible proxy
# ============================================================================

if (!"fam_inc" %in% colnames(calib_eval_data)) {
  stop("Column 'fam_inc' not found in LSAC dataset.")
}

income_data <- as.numeric(as.character(calib_eval_data$fam_inc))
cat("Family income variable (fam_inc):\n")
cat("  Categories:", paste(sort(unique(income_data[!is.na(income_data)])),
                           collapse = ", "), "\n")

# Binarize: high income = category 5 (paper definition)
income_binary <- as.numeric(income_data == 5)
cat("  High income (category 5): proportion =",
    sprintf("%.2f%%", 100 * mean(income_binary, na.rm = TRUE)), "\n\n")

# ============================================================================
# Algorithm 1: Family income vs single permissible proxy
# ============================================================================

cat(paste0(rep("=", 70), collapse = ""), "\n")
cat("ALGORITHM 1: Family income (impermissible) vs GPA (permissible)\n")
cat(paste0(rep("=", 70), collapse = ""), "\n\n")

result_single <- falsify_single_proxy(
  predictions          = predictions_raw,
  impermissible_labels = income_binary,
  permissible_labels   = calib_eval_data$gpa_binary,
  alpha = 0.05,
  calib_indices = calib_rel,
  eval_indices  = eval_rel
)

cat("  Result:", result_single$result, "\n")
cat("  P-value:", format_pvalue(result_single$p_value,
                                z = result_single$z_statistic), "\n")
cat("  Mean loss difference:", sprintf("%.4f", result_single$delta_mean), "\n\n")

# ============================================================================
# Algorithm 2: Family income vs multiple permissible proxies
# ============================================================================

cat(paste0(rep("=", 70), collapse = ""), "\n")
cat("ALGORITHM 2: Family income (impermissible) vs multiple permissible proxies\n")
cat(paste0(rep("=", 70), collapse = ""), "\n\n")

permissible_matrix <- cbind(
  fygpa    = calib_eval_data$fygpa_binary,
  gpa      = calib_eval_data$gpa_binary,
  pass_bar = calib_eval_data$pass_bar
)

result_multiple <- falsify_multiple_proxy(
  predictions          = predictions_raw,
  impermissible_labels = income_binary,
  permissible_labels_matrix = permissible_matrix,
  alpha = 0.05,
  calib_indices = calib_rel,
  eval_indices  = eval_rel
)

cat("  Result:", result_multiple$result, "\n")
cat("  P-value:", format_pvalue(result_multiple$conditional_rank_pvalue,
                                z = result_multiple$z_statistic), "\n\n")

# ============================================================================
# Save results
# ============================================================================

results_df <- data.frame(
  Test    = c("Single Proxy (Alg 1)", "Multiple Proxy (Alg 2)"),
  Impermissible = "Family Income (cat 5)",
  P_Value = c(result_single$p_value, result_multiple$conditional_rank_pvalue),
  Result  = c(result_single$result, result_multiple$result),
  stringsAsFactors = FALSE
)
write.csv(results_df, "lsac_family_income_results.csv", row.names = FALSE)
cat("Results saved to lsac_family_income_results.csv\n")
