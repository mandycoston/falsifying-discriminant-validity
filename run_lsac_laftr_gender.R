# ============================================================================
# LSAC Dataset: Falsification Tests on LAFTR Model (adversary = gender)
# ============================================================================
# Modeled outcome: fygpa_binary (first-year GPA above/below median)
# Impermissible proxies: race, gender
# Permissible proxies: fygpa_binary, gpa_binary, pass_bar
# Predictions from compute_laftr_table1_metrics.py (LAFTR, DP adversary on gender).
# ============================================================================

source("falsification_methods.R")
source("load_lsac_data.R")

set.seed(42)

cat("Loading LSAC dataset...\n")
lsac_data <- load_lsac_data()

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
    " eval =", length(eval_indices), "\n\n")

# Load LAFTR gender-adversary predictions
pred_file <- "predictions_lsac_laftr_gender.csv"
if (!file.exists(pred_file)) {
  stop("Run compute_laftr_table1_metrics.py first to generate ", pred_file)
}
pred_df <- read.csv(pred_file)
predictions_raw <- as.numeric(pred_df$prediction)

calib_eval_data <- lsac_data[c(calib_indices, eval_indices), ]
calib_indices_rel <- 1:length(calib_indices)
eval_indices_rel  <- (length(calib_indices) + 1):length(predictions_raw)

# ============================================================================
# Race as impermissible proxy
# ============================================================================

race_proxy <- as.numeric(tolower(as.character(calib_eval_data$race)) %in% c("white", "1"))
permissible_matrix <- cbind(
  fygpa    = calib_eval_data$fygpa_binary,
  gpa      = calib_eval_data$gpa_binary,
  pass_bar = calib_eval_data$pass_bar
)

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("FALSIFICATION (Alg 1): LAFTR (adv=gender) vs RACE\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

result_race_single <- tryCatch({
  falsify_single_proxy(
    predictions = predictions_raw,
    impermissible_labels = race_proxy,
    permissible_labels = calib_eval_data$gpa_binary,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices = eval_indices_rel
  )
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
  list(result = "ERROR", p_value = NA, z_statistic = NA, test_used = "error")
})
cat("  Result:", result_race_single$result, "\n")
cat("  P-value:", format_pvalue(result_race_single$p_value, z = result_race_single$z_statistic), "\n\n")

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("FALSIFICATION (Alg 2): LAFTR (adv=gender) vs RACE (multiple permissible)\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

result_race_multi <- tryCatch({
  falsify_multiple_proxy(
    predictions = predictions_raw,
    impermissible_labels = race_proxy,
    permissible_labels_matrix = permissible_matrix,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices = eval_indices_rel
  )
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
  list(result = "ERROR", conditional_rank_pvalue = NA, z_statistic = NA)
})
cat("  Result:", result_race_multi$result, "\n")
cat("  P-value:", format_pvalue(result_race_multi$conditional_rank_pvalue, z = result_race_multi$z_statistic), "\n\n")

# ============================================================================
# Gender as impermissible proxy
# ============================================================================

gender_proxy <- as.numeric(calib_eval_data$male)

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("FALSIFICATION (Alg 1): LAFTR (adv=gender) vs GENDER\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

result_gender_single <- tryCatch({
  falsify_single_proxy(
    predictions = predictions_raw,
    impermissible_labels = gender_proxy,
    permissible_labels = calib_eval_data$gpa_binary,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices = eval_indices_rel
  )
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
  list(result = "ERROR", p_value = NA, z_statistic = NA, test_used = "error")
})
cat("  Result:", result_gender_single$result, "\n")
cat("  P-value:", format_pvalue(result_gender_single$p_value, z = result_gender_single$z_statistic), "\n\n")

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("FALSIFICATION (Alg 2): LAFTR (adv=gender) vs GENDER (multiple permissible)\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

result_gender_multi <- tryCatch({
  falsify_multiple_proxy(
    predictions = predictions_raw,
    impermissible_labels = gender_proxy,
    permissible_labels_matrix = permissible_matrix,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices = eval_indices_rel
  )
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
  list(result = "ERROR", conditional_rank_pvalue = NA, z_statistic = NA)
})
cat("  Result:", result_gender_multi$result, "\n")
cat("  P-value:", format_pvalue(result_gender_multi$conditional_rank_pvalue, z = result_gender_multi$z_statistic), "\n\n")

# ============================================================================
# Save
# ============================================================================

results_df <- data.frame(
  impermissible_proxy = c("race", "race", "gender", "gender"),
  algorithm = c("Alg1", "Alg2", "Alg1", "Alg2"),
  p_value = c(result_race_single$p_value, result_race_multi$conditional_rank_pvalue,
              result_gender_single$p_value, result_gender_multi$conditional_rank_pvalue),
  result = c(result_race_single$result, result_race_multi$result,
             result_gender_single$result, result_gender_multi$result),
  stringsAsFactors = FALSE
)
write.csv(results_df, "lsac_laftr_gender_results.csv", row.names = FALSE)
cat("Results saved to lsac_laftr_gender_results.csv\n")
