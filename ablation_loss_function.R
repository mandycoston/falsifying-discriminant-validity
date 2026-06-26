# Ablation: Effect of Loss Function (Log Loss vs Brier Score)
# Reproduces Table A2 from the paper.
# Tests Algorithm 1 on LSAC (race, gender) and COMPAS (age, race)
# with both log loss and Brier score.

source("falsification_methods.R")
source("load_lsac_data.R")

set.seed(42)

# ============================================================================
# Brier score loss function
# ============================================================================

brier_score <- function(predictions, labels) {
  predictions <- pmax(pmin(predictions, 1 - 1e-15), 1e-15)
  (labels - predictions)^2
}

# ============================================================================
# Helper: run Algorithm 1 with a specified loss function
# ============================================================================

falsify_single_proxy_loss <- function(predictions, impermissible_labels,
                                       permissible_labels, calib_indices,
                                       eval_indices, loss_fn, alpha = 0.05) {
  # Platt scaling
  platt_imp <- platt_scaling(predictions, impermissible_labels, calib_indices)
  platt_perm <- platt_scaling(predictions, permissible_labels, calib_indices)

  calib_pred_imp <- platt_imp$calibrated[eval_indices]
  calib_pred_perm <- platt_perm$calibrated[eval_indices]

  loss_imp <- loss_fn(calib_pred_imp, impermissible_labels[eval_indices])
  loss_perm <- loss_fn(calib_pred_perm, permissible_labels[eval_indices])

  delta <- loss_imp - loss_perm
  test_result <- wilcox.test(delta, alternative = "greater", mu = 0)

  list(
    result = ifelse(test_result$p.value <= alpha, "DISCRIMINANT",
                    "INDISCRIMINANT"),
    p_value = test_result$p.value,
    delta_mean = mean(delta)
  )
}

# ============================================================================
# LSAC analysis
# ============================================================================

cat("Loading LSAC dataset...\n")
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
model <- glm(fygpa_binary ~ lsat + ugpa, data = train_data, family = binomial,
             control = list(maxit = 1000))
predictions_lsac <- predict(model, newdata = calib_eval_data, type = "response")

calib_rel <- 1:length(calib_indices)
eval_rel  <- (length(calib_indices) + 1):nrow(calib_eval_data)

race_proxy   <- as.numeric(tolower(as.character(calib_eval_data$race)) %in%
                             c("white", "1"))
gender_proxy <- as.numeric(calib_eval_data$male)
perm_proxy   <- calib_eval_data$gpa_binary  # single permissible proxy for Alg 1

results <- data.frame(
  Dataset = character(), Impermissible = character(), Loss_Function = character(),
  Mean_Diff = numeric(), P_Value = numeric(), Result = character(),
  stringsAsFactors = FALSE
)

for (loss_name in c("Log loss", "Brier score")) {
  loss_fn <- if (loss_name == "Log loss") log_loss else brier_score
  for (imp_info in list(list("Race", race_proxy), list("Gender", gender_proxy))) {
    imp_name <- imp_info[[1]]
    imp_labels <- imp_info[[2]]
    cat("LSAC |", imp_name, "|", loss_name, "... ")
    r <- falsify_single_proxy_loss(predictions_lsac, imp_labels, perm_proxy,
                                    calib_rel, eval_rel, loss_fn)
    cat(r$result, " p=", format_pvalue(r$p_value), "\n")
    results <- rbind(results, data.frame(
      Dataset = "LSAC", Impermissible = imp_name, Loss_Function = loss_name,
      Mean_Diff = round(r$delta_mean, 2), P_Value = r$p_value,
      Result = r$result, stringsAsFactors = FALSE))
  }
}

# ============================================================================
# COMPAS analysis
# ============================================================================

cat("\nLoading COMPAS dataset...\n")
repo_dir <- "compas-analysis"
data_file <- file.path(repo_dir, "compas-scores-two-years.csv")
if (!file.exists(data_file)) stop("COMPAS data not found at ", data_file)
compas_data <- read.csv(data_file, stringsAsFactors = FALSE)

compas_data$two_year_recid <- as.numeric(compas_data$two_year_recid)
compas_data$age_lt25 <- as.numeric(compas_data$age < 25)
compas_data$race_black <- as.numeric(
  grepl("African|Black", compas_data$race, ignore.case = TRUE))

# Use decile_score, convert to [0,1]
compas_scores <- (compas_data$decile_score - 1) / 9
compas_scores <- pmax(pmin(compas_scores, 1), 0)

keep <- complete.cases(compas_data[, c("two_year_recid", "age_lt25", "race_black")]) &
        !is.na(compas_scores)
compas_data <- compas_data[keep, ]
compas_scores <- compas_scores[keep]

set.seed(42)
n_c <- nrow(compas_data)
calib_size_c <- floor(n_c * 0.2)
calib_idx_c <- sample(n_c, calib_size_c)
eval_idx_c  <- setdiff(1:n_c, calib_idx_c)

for (loss_name in c("Log loss", "Brier score")) {
  loss_fn <- if (loss_name == "Log loss") log_loss else brier_score
  for (imp_info in list(list("Age < 25", compas_data$age_lt25),
                         list("Race", compas_data$race_black))) {
    imp_name <- imp_info[[1]]
    imp_labels <- imp_info[[2]]
    cat("COMPAS |", imp_name, "|", loss_name, "... ")
    r <- falsify_single_proxy_loss(compas_scores, imp_labels,
                                    compas_data$two_year_recid,
                                    calib_idx_c, eval_idx_c, loss_fn)
    cat(r$result, " p=", format_pvalue(r$p_value), "\n")
    results <- rbind(results, data.frame(
      Dataset = "COMPAS", Impermissible = imp_name, Loss_Function = loss_name,
      Mean_Diff = round(r$delta_mean, 2), P_Value = r$p_value,
      Result = r$result, stringsAsFactors = FALSE))
  }
}

# ============================================================================
# Output
# ============================================================================

cat("\n========== Table A2: Effect of Loss Function ==========\n")
print(results)
write.csv(results, "ablation_loss_function_results.csv", row.names = FALSE)
cat("\nResults saved to ablation_loss_function_results.csv\n")
