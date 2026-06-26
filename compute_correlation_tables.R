# Compute Correlation Tables (Tables 1, A3, and COMPAS correlation table)
#
# Table 1:  Pearson correlations for 3 models x 5 proxies (LSAC)
# Table A3: Spearman and Kendall correlations for 3 models x 5 proxies (LSAC)
# COMPAS:   Pearson, Spearman, Kendall for COMPAS decile scores x 3 proxies

source("falsification_methods.R")
source("load_lsac_data.R")

set.seed(42)

# ============================================================================
# LSAC: Load data and split (same as main analysis)
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
eval_data  <- lsac_data[eval_indices, ]

# ============================================================================
# LSAC: Prepare proxy labels on the evaluation set
# ============================================================================

race_eval   <- as.numeric(tolower(as.character(eval_data$race)) %in%
                            c("white", "1"))
gender_eval <- as.numeric(eval_data$male)

proxy_labels <- data.frame(
  `1styearGPA` = eval_data$fygpa_binary,
  GPA          = eval_data$gpa_binary,
  PassBar      = eval_data$pass_bar,
  Race         = race_eval,
  Gender       = gender_eval,
  check.names  = FALSE
)

# ============================================================================
# LSAC: Get predictions from each model on eval set
# ============================================================================

# Model 1: Logistic regression
model_lr <- glm(fygpa_binary ~ lsat + ugpa, data = train_data,
                family = binomial, control = list(maxit = 1000))
preds_lr <- predict(model_lr, newdata = eval_data, type = "response")

# Model 2: LAFTR (race adversary) — load from CSV
calib_eval_data <- lsac_data[c(calib_indices, eval_indices), ]
eval_rel <- (length(calib_indices) + 1):nrow(calib_eval_data)

pred_file_race <- "predictions_lsac_laftr.csv"
pred_file_gender <- "predictions_lsac_laftr_gender.csv"

if (file.exists(pred_file_race)) {
  preds_laftr_race <- read.csv(pred_file_race)$prediction[eval_rel]
} else {
  cat("WARNING:", pred_file_race, "not found. Skipping LAFTR (adv=race).\n")
  preds_laftr_race <- NULL
}

# Model 3: LAFTR (gender adversary) — load from CSV
if (file.exists(pred_file_gender)) {
  preds_laftr_gender <- read.csv(pred_file_gender)$prediction[eval_rel]
} else {
  cat("WARNING:", pred_file_gender, "not found. Skipping LAFTR (adv=gender).\n")
  preds_laftr_gender <- NULL
}

# ============================================================================
# LSAC: Compute correlations (Pearson, Spearman, Kendall)
# ============================================================================

compute_corr_row <- function(preds, proxy_labels, method) {
  sapply(proxy_labels, function(y) cor(preds, y, method = method,
                                        use = "complete.obs"))
}

model_names <- "Logistic regression"
pred_list <- list(preds_lr)
if (!is.null(preds_laftr_race)) {
  model_names <- c(model_names, "LAFTR (adv=race)")
  pred_list <- c(pred_list, list(preds_laftr_race))
}
if (!is.null(preds_laftr_gender)) {
  model_names <- c(model_names, "LAFTR (adv=gender)")
  pred_list <- c(pred_list, list(preds_laftr_gender))
}

cat("\n========== Table 1: Pearson Correlations (LSAC) ==========\n")
pearson_table <- do.call(rbind, lapply(pred_list, compute_corr_row,
                                        proxy_labels, "pearson"))
rownames(pearson_table) <- model_names
print(round(pearson_table, 4))

cat("\n========== Table A3: Spearman Correlations (LSAC) ==========\n")
spearman_table <- do.call(rbind, lapply(pred_list, compute_corr_row,
                                         proxy_labels, "spearman"))
rownames(spearman_table) <- model_names
print(round(spearman_table, 4))

cat("\n========== Table A3: Kendall Correlations (LSAC) ==========\n")
kendall_table <- do.call(rbind, lapply(pred_list, compute_corr_row,
                                        proxy_labels, "kendall"))
rownames(kendall_table) <- model_names
print(round(kendall_table, 4))

# Save LSAC correlation tables
lsac_corr <- rbind(
  data.frame(Measure = "Pearson", Model = model_names, pearson_table,
             check.names = FALSE, stringsAsFactors = FALSE),
  data.frame(Measure = "Spearman", Model = model_names, spearman_table,
             check.names = FALSE, stringsAsFactors = FALSE),
  data.frame(Measure = "Kendall", Model = model_names, kendall_table,
             check.names = FALSE, stringsAsFactors = FALSE)
)
write.csv(lsac_corr, "lsac_correlation_table_all.csv", row.names = FALSE)
cat("\nSaved to lsac_correlation_table_all.csv\n")

# ============================================================================
# COMPAS: Correlation table (Pearson, Spearman, Kendall)
# ============================================================================

cat("\n========== COMPAS Correlation Table ==========\n")

repo_dir <- "compas-analysis"
data_file <- file.path(repo_dir, "compas-scores-two-years.csv")
if (file.exists(data_file)) {
  compas_data <- read.csv(data_file, stringsAsFactors = FALSE)
  compas_data$two_year_recid <- as.numeric(compas_data$two_year_recid)
  compas_data$age_lt25 <- as.numeric(compas_data$age < 25)
  compas_data$race_black <- as.numeric(
    grepl("African|Black", compas_data$race, ignore.case = TRUE))

  keep <- complete.cases(compas_data[, c("two_year_recid", "age_lt25",
                                          "race_black", "decile_score")])
  compas_data <- compas_data[keep, ]

  # Use all data for correlation (no split needed — this is descriptive)
  # But to match the paper, use the eval set
  set.seed(42)
  n_c <- nrow(compas_data)
  calib_size_c <- floor(n_c * 0.2)
  calib_idx <- sample(n_c, calib_size_c)
  eval_idx  <- setdiff(1:n_c, calib_idx)
  compas_eval <- compas_data[eval_idx, ]

  compas_proxies <- data.frame(
    `Re-arrest`   = compas_eval$two_year_recid,
    `Age < 25`    = compas_eval$age_lt25,
    `Race (Black)` = compas_eval$race_black,
    check.names = FALSE
  )

  compas_corr_results <- data.frame(
    Measure = character(), `Re-arrest` = numeric(), `Age < 25` = numeric(),
    `Race (Black)` = numeric(), check.names = FALSE, stringsAsFactors = FALSE
  )

  for (method in c("Pearson", "Spearman", "Kendall")) {
    r <- sapply(compas_proxies, function(y)
      cor(compas_eval$decile_score, y, method = tolower(method),
          use = "complete.obs"))
    compas_corr_results <- rbind(compas_corr_results,
      data.frame(Measure = method, t(r), check.names = FALSE,
                 stringsAsFactors = FALSE))
  }

  print(compas_corr_results)
  write.csv(compas_corr_results, "compas_correlation_table.csv",
            row.names = FALSE)
  cat("Saved to compas_correlation_table.csv\n")
} else {
  cat("COMPAS data not found at", data_file, "— skipping.\n")
}
