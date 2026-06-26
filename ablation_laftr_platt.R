# Ablation: Effect of Platt Scaling on Falsification Test Results (Table A1)
#
# Tests all models with and without Platt scaling calibration:
#   LSAC: Logistic regression (Alg 1), LAFTR race-adversary (Alg 2),
#          LAFTR gender-adversary (Alg 2)
#   COMPAS: Existing decile scores (Alg 1)

source("falsification_methods.R")
source("load_lsac_data.R")

set.seed(42)

# ============================================================================
# LSAC: Load and split data
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
calib_indices_rel <- 1:length(calib_indices)
eval_indices_rel  <- (length(calib_indices) + 1):(length(calib_indices) + length(eval_indices))

race_proxy   <- as.numeric(tolower(as.character(calib_eval_data$race)) %in% c("white", "1"))
gender_proxy <- as.numeric(calib_eval_data$male)
perm_proxy   <- calib_eval_data$gpa_binary  # single permissible proxy for Alg 1

permissible_matrix <- cbind(
  fygpa    = calib_eval_data$fygpa_binary,
  gpa      = calib_eval_data$gpa_binary,
  pass_bar = calib_eval_data$pass_bar
)

# ============================================================================
# Helper: run Alg 1 WITHOUT Platt scaling (raw predictions)
# ============================================================================

falsify_single_no_platt <- function(predictions, impermissible_labels,
                                     permissible_labels, eval_indices,
                                     alpha = 0.05) {
  pred_eval <- predictions[eval_indices]
  pred_eval <- pmax(pmin(pred_eval, 1 - 1e-15), 1e-15)

  loss_imp  <- log_loss(pred_eval, impermissible_labels[eval_indices])
  loss_perm <- log_loss(pred_eval, permissible_labels[eval_indices])

  delta <- loss_imp - loss_perm
  test_result <- wilcox.test(delta, alternative = "greater", mu = 0)

  list(result = ifelse(test_result$p.value <= alpha, "DISCRIMINANT",
                       "INDISCRIMINANT"),
       p_value = test_result$p.value,
       delta_mean = mean(delta))
}

# ============================================================================
# Helper: run Alg 2 WITHOUT Platt scaling (raw predictions)
# ============================================================================

falsify_multiple_no_platt <- function(predictions, impermissible_labels,
                                       permissible_labels_matrix,
                                       eval_indices, alpha = 0.05) {
  M <- ncol(permissible_labels_matrix)
  pred_eval <- predictions[eval_indices]
  pred_eval <- pmax(pmin(pred_eval, 1 - 1e-15), 1e-15)

  loss_imp <- log_loss(pred_eval, impermissible_labels[eval_indices])
  loss_perm <- matrix(NA, nrow = length(eval_indices), ncol = M)
  for (j in 1:M) {
    loss_perm[, j] <- log_loss(pred_eval, permissible_labels_matrix[eval_indices, j])
  }

  loss_matrix <- cbind(loss_imp, loss_perm)
  ranks_matrix <- t(apply(loss_matrix, 1, function(x) rank(x, ties.method = "average")))
  imp_ranks <- ranks_matrix[, 1]

  test_stat <- mean(imp_ranks)
  expected  <- (M + 2) / 2
  var_rank  <- ((M + 1)^2 - 1) / 12
  se        <- sqrt(var_rank / length(eval_indices))
  z         <- (test_stat - expected) / se
  p         <- 1 - pnorm(z)

  delta_mean <- mean(loss_imp - rowMeans(loss_perm))

  list(result = ifelse(p <= alpha, "DISCRIMINANT", "INDISCRIMINANT"),
       p_value = p, z_statistic = z, test_statistic = test_stat,
       delta_mean = delta_mean)
}

# ============================================================================
# Helper: run Alg 2 WITH Platt scaling (existing function)
# ============================================================================

run_alg2_with_platt <- function(predictions, impermissible_labels,
                                 permissible_matrix, calib_indices_rel,
                                 eval_indices_rel) {
  result <- falsify_multiple_proxy(
    predictions = predictions,
    impermissible_labels = impermissible_labels,
    permissible_labels_matrix = permissible_matrix,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices = eval_indices_rel
  )
  list(result = result$result,
       p_value = result$conditional_rank_pvalue,
       z_statistic = result$z_statistic,
       test_statistic = result$test_statistic,
       delta_mean = NA)
}

results <- data.frame(
  Model = character(), Impermissible = character(), Calibration = character(),
  Mean_Diff = numeric(), P_Value = numeric(), Result = character(),
  stringsAsFactors = FALSE
)

# ============================================================================
# LSAC: Logistic regression (Algorithm 1, single permissible proxy)
# ============================================================================

model_lr <- glm(fygpa_binary ~ lsat + ugpa, data = train_data, family = binomial,
                control = list(maxit = 1000))
preds_lr <- predict(model_lr, newdata = calib_eval_data, type = "response")

for (imp_info in list(list("Race", race_proxy), list("Gender", gender_proxy))) {
  imp_name <- imp_info[[1]]
  imp_labels <- imp_info[[2]]

  cat("=== Logistic reg. vs", imp_name, "— WITH Platt scaling ===\n")
  r_with <- falsify_single_proxy(
    predictions = preds_lr,
    impermissible_labels = imp_labels,
    permissible_labels = perm_proxy,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices = eval_indices_rel
  )
  cat("  Result:", r_with$result, "  p-value:",
      format_pvalue(r_with$p_value, z = r_with$z_statistic),
      "  Mean diff:", sprintf("%.4f", r_with$delta_mean), "\n\n")

  cat("=== Logistic reg. vs", imp_name, "— WITHOUT Platt scaling ===\n")
  r_without <- falsify_single_no_platt(preds_lr, imp_labels, perm_proxy,
                                        eval_indices_rel)
  cat("  Result:", r_without$result, "  p-value:",
      format_pvalue(r_without$p_value),
      "  Mean diff:", sprintf("%.4f", r_without$delta_mean), "\n\n")

  results <- rbind(results, data.frame(
    Model = "Logistic reg.", Impermissible = imp_name,
    Calibration = "With Platt scaling",
    Mean_Diff = r_with$delta_mean, P_Value = r_with$p_value,
    Result = r_with$result))
  results <- rbind(results, data.frame(
    Model = "Logistic reg.", Impermissible = imp_name,
    Calibration = "Without Platt scaling",
    Mean_Diff = r_without$delta_mean, P_Value = r_without$p_value,
    Result = r_without$result))
}

# ============================================================================
# LSAC: LAFTR (race adversary) — Algorithm 2
# ============================================================================

pred_race <- read.csv("predictions_lsac_laftr.csv")$prediction

for (imp_info in list(list("Race", race_proxy), list("Gender", gender_proxy))) {
  imp_name <- imp_info[[1]]
  imp_labels <- imp_info[[2]]

  cat("=== LAFTR (adv=race) vs", imp_name, "— WITH Platt scaling ===\n")
  r_with <- run_alg2_with_platt(pred_race, imp_labels, permissible_matrix,
                                 calib_indices_rel, eval_indices_rel)
  cat("  Result:", r_with$result, "  p-value:",
      format_pvalue(r_with$p_value, z = r_with$z_statistic), "\n\n")

  cat("=== LAFTR (adv=race) vs", imp_name, "— WITHOUT Platt scaling ===\n")
  r_without <- falsify_multiple_no_platt(pred_race, imp_labels,
                                          permissible_matrix, eval_indices_rel)
  cat("  Result:", r_without$result, "  p-value:",
      format_pvalue(r_without$p_value, z = r_without$z_statistic),
      "  Mean diff:", sprintf("%.4f", r_without$delta_mean), "\n\n")

  results <- rbind(results, data.frame(
    Model = "LAFTR (adv=race)", Impermissible = imp_name,
    Calibration = "With Platt scaling",
    Mean_Diff = r_with$delta_mean, P_Value = r_with$p_value,
    Result = r_with$result))
  results <- rbind(results, data.frame(
    Model = "LAFTR (adv=race)", Impermissible = imp_name,
    Calibration = "Without Platt scaling",
    Mean_Diff = r_without$delta_mean, P_Value = r_without$p_value,
    Result = r_without$result))
}

# ============================================================================
# LSAC: LAFTR (gender adversary) — Algorithm 2
# ============================================================================

pred_gender <- read.csv("predictions_lsac_laftr_gender.csv")$prediction

for (imp_info in list(list("Gender", gender_proxy), list("Race", race_proxy))) {
  imp_name <- imp_info[[1]]
  imp_labels <- imp_info[[2]]

  cat("=== LAFTR (adv=gender) vs", imp_name, "— WITH Platt scaling ===\n")
  r_with <- run_alg2_with_platt(pred_gender, imp_labels, permissible_matrix,
                                 calib_indices_rel, eval_indices_rel)
  cat("  Result:", r_with$result, "  p-value:",
      format_pvalue(r_with$p_value, z = r_with$z_statistic), "\n\n")

  cat("=== LAFTR (adv=gender) vs", imp_name, "— WITHOUT Platt scaling ===\n")
  r_without <- falsify_multiple_no_platt(pred_gender, imp_labels,
                                          permissible_matrix, eval_indices_rel)
  cat("  Result:", r_without$result, "  p-value:",
      format_pvalue(r_without$p_value, z = r_without$z_statistic),
      "  Mean diff:", sprintf("%.4f", r_without$delta_mean), "\n\n")

  results <- rbind(results, data.frame(
    Model = "LAFTR (adv=gender)", Impermissible = imp_name,
    Calibration = "With Platt scaling",
    Mean_Diff = r_with$delta_mean, P_Value = r_with$p_value,
    Result = r_with$result))
  results <- rbind(results, data.frame(
    Model = "LAFTR (adv=gender)", Impermissible = imp_name,
    Calibration = "Without Platt scaling",
    Mean_Diff = r_without$delta_mean, P_Value = r_without$p_value,
    Result = r_without$result))
}

# ============================================================================
# COMPAS: Existing decile scores (Algorithm 1)
# ============================================================================

cat("\n========== COMPAS ==========\n")
repo_dir <- "compas-analysis"
data_file <- file.path(repo_dir, "compas-scores-two-years.csv")
if (!file.exists(data_file)) stop("COMPAS data not found at ", data_file)
compas_data <- read.csv(data_file, stringsAsFactors = FALSE)

compas_data$two_year_recid <- as.numeric(compas_data$two_year_recid)
compas_data$age_lt25 <- as.numeric(compas_data$age < 25)
compas_data$race_black <- as.numeric(
  grepl("African|Black", compas_data$race, ignore.case = TRUE))

compas_scores <- (compas_data$decile_score - 1) / 9
compas_scores <- pmax(pmin(compas_scores, 1), 0)

keep <- complete.cases(compas_data[, c("two_year_recid", "age_lt25",
                                        "race_black", "decile_score")])
compas_data <- compas_data[keep, ]
compas_scores <- compas_scores[keep]

set.seed(42)
n_c <- nrow(compas_data)
calib_size_c <- floor(n_c * 0.2)
calib_idx_c <- sample(n_c, calib_size_c)
eval_idx_c  <- setdiff(1:n_c, calib_idx_c)

compas_perm <- compas_data$two_year_recid

for (imp_info in list(list("Age < 25", compas_data$age_lt25),
                       list("Race", compas_data$race_black))) {
  imp_name <- imp_info[[1]]
  imp_labels <- imp_info[[2]]

  cat("=== COMPAS scores vs", imp_name, "— WITH Platt scaling ===\n")
  r_with <- falsify_single_proxy(
    predictions = compas_scores,
    impermissible_labels = imp_labels,
    permissible_labels = compas_perm,
    alpha = 0.05,
    calib_indices = calib_idx_c,
    eval_indices = eval_idx_c
  )
  cat("  Result:", r_with$result, "  p-value:",
      format_pvalue(r_with$p_value, z = r_with$z_statistic),
      "  Mean diff:", sprintf("%.4f", r_with$delta_mean), "\n\n")

  cat("=== COMPAS scores vs", imp_name, "— WITHOUT Platt scaling ===\n")
  r_without <- falsify_single_no_platt(compas_scores, imp_labels, compas_perm,
                                        eval_idx_c)
  cat("  Result:", r_without$result, "  p-value:",
      format_pvalue(r_without$p_value),
      "  Mean diff:", sprintf("%.4f", r_without$delta_mean), "\n\n")

  results <- rbind(results, data.frame(
    Model = "COMPAS scores", Impermissible = imp_name,
    Calibration = "With Platt scaling",
    Mean_Diff = r_with$delta_mean, P_Value = r_with$p_value,
    Result = r_with$result))
  results <- rbind(results, data.frame(
    Model = "COMPAS scores", Impermissible = imp_name,
    Calibration = "Without Platt scaling",
    Mean_Diff = r_without$delta_mean, P_Value = r_without$p_value,
    Result = r_without$result))
}

# ============================================================================
# Output
# ============================================================================

cat("\n========== Table A1: Effect of Platt Scaling ==========\n")
print(results)

write.csv(results, "ablation_laftr_platt_results.csv", row.names = FALSE)
cat("\nResults saved to ablation_laftr_platt_results.csv\n")
