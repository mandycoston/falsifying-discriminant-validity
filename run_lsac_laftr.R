# ============================================================================
# LSAC Dataset: Falsification Tests on LAFTR Model (Madras et al. 2018)
# ============================================================================
# Modeled outcome: fygpa_binary (first-year GPA above/below median)
# Impermissible proxy: race (White vs non-White)
# Permissible proxies: fygpa_binary, gpa_binary, pass_bar
# Predictions from train_laftr_models.py (PyTorch LAFTR, DP adversary).
# ============================================================================

source("falsification_methods.R")
source("load_lsac_data.R")

set.seed(42)

# ============================================================================
# Load and Prepare Data
# ============================================================================

cat("Loading LSAC dataset...\n")
lsac_data <- load_lsac_data()
cat("Data dimensions:", dim(lsac_data), "\n\n")

required_cols <- c("lsat", "gpa", "fygpa_binary", "gpa_binary", "pass_bar", "race", "male")
missing_cols <- setdiff(required_cols, colnames(lsac_data))
if (length(missing_cols) > 0) stop("Missing columns: ", paste(missing_cols, collapse = ", "))

# ============================================================================
# Split (same as run_lsac_fairlearn.R: 50/25/25, seed 42)
# ============================================================================

n <- nrow(lsac_data)
train_prop <- 0.5; calib_prop <- 0.25
all_indices <- sample(n)
train_size <- floor(n * train_prop)
calib_size <- floor(n * calib_prop)
train_indices <- all_indices[1:train_size]
calib_indices <- all_indices[(train_size + 1):(train_size + calib_size)]
eval_indices  <- all_indices[(train_size + calib_size + 1):n]

cat("Split: train =", length(train_indices),
    " calib =", length(calib_indices),
    " eval =", length(eval_indices), "\n")

# Export indices for Python (reuse if present)
idx_file <- "lsac_fairlearn_indices.csv"
if (!file.exists(idx_file)) {
  write.csv(data.frame(index = c(calib_indices, eval_indices)), idx_file, row.names = FALSE)
  cat("Wrote", idx_file, "\n")
}

# ============================================================================
# Load LAFTR Predictions
# ============================================================================

pred_file <- "predictions_lsac_laftr.csv"
if (!file.exists(pred_file)) {
  stop("LAFTR predictions not found. Run first:\n  python train_laftr_models.py\nThen run this script again.")
}

pred_df <- read.csv(pred_file)
predictions_raw <- as.numeric(pred_df$prediction)
expected_len <- length(calib_indices) + length(eval_indices)
if (length(predictions_raw) != expected_len) {
  stop("Prediction length ", length(predictions_raw), " != calib+eval ", expected_len)
}

calib_eval_data <- lsac_data[c(calib_indices, eval_indices), ]
calib_indices_rel <- 1:length(calib_indices)
eval_indices_rel  <- (length(calib_indices) + 1):length(predictions_raw)

# ============================================================================
# Performance summary
# ============================================================================

eval_pred   <- predictions_raw[eval_indices_rel]
eval_fygpa  <- calib_eval_data$fygpa_binary[eval_indices_rel]
eval_race   <- as.numeric(tolower(as.character(calib_eval_data$race[eval_indices_rel])) %in%
                            c("white", "1"))

cat("\nLAFTR model (LSAC) eval-set summary:\n")
cat("  Mean prediction:", sprintf("%.4f", mean(eval_pred)), "\n")
cat("  Log loss (fygpa):", sprintf("%.4f", mean(log_loss(eval_pred, eval_fygpa))), "\n\n")

# ============================================================================
# Falsification: Algorithm 1 (race as impermissible proxy)
# ============================================================================

impermissible_proxy <- as.numeric(tolower(as.character(calib_eval_data$race)) %in%
                                    c("white", "1"))
permissible_proxy   <- calib_eval_data$gpa_binary

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("FALSIFICATION (Alg 1): LAFTR vs race\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

result_single <- tryCatch({
  falsify_single_proxy(
    predictions          = predictions_raw,
    impermissible_labels = impermissible_proxy,
    permissible_labels   = permissible_proxy,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices  = eval_indices_rel
  )
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
  list(result = "ERROR", p_value = NA, delta_mean = NA, test_used = "error", z_statistic = NA)
})

cat("  Result:", result_single$result, "\n")
cat("  P-value:", format_pvalue(result_single$p_value, z = result_single$z_statistic), "\n")
cat("  Test used:", result_single$test_used, "\n\n")

# ============================================================================
# Falsification: Algorithm 2 (multiple permissible proxies)
# ============================================================================

permissible_matrix <- cbind(
  fygpa    = calib_eval_data$fygpa_binary,
  gpa      = calib_eval_data$gpa_binary,
  pass_bar = calib_eval_data$pass_bar
)

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("FALSIFICATION (Alg 2): LAFTR vs race (multiple permissible proxies)\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

result_multiple <- tryCatch({
  falsify_multiple_proxy(
    predictions          = predictions_raw,
    impermissible_labels = impermissible_proxy,
    permissible_labels_matrix = permissible_matrix,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices  = eval_indices_rel
  )
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
  list(result = "ERROR", conditional_rank_pvalue = NA, z_statistic = NA)
})

cat("  Result:", result_multiple$result, "\n")
cat("  P-value:", format_pvalue(result_multiple$conditional_rank_pvalue,
                                z = result_multiple$z_statistic), "\n\n")

# ============================================================================
# Save Results
# ============================================================================

results_df <- data.frame(
  Test    = c("Single Proxy", "Multiple Proxy"),
  P_Value = c(result_single$p_value, result_multiple$conditional_rank_pvalue),
  Result  = c(result_single$result, result_multiple$result),
  stringsAsFactors = FALSE
)
write.csv(results_df, "lsac_laftr_results_race.csv", row.names = FALSE)
cat("Results saved to lsac_laftr_results_race.csv\n\n")
cat("LSAC LAFTR analysis complete.\n")
