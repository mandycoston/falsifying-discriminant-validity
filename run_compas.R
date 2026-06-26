# ============================================================================
# COMPAS Dataset Analysis: Falsification Tests
# ============================================================================
# 
# This script performs falsification tests on the COMPAS dataset.
# Modeled outcome: two_year_recid (two-year recidivism)
# Impermissible proxy: race
# Permissible proxy: two_year_recid (same as modeled outcome)
# 
# Only runs Algorithm 1 (single proxy test)
# ============================================================================

# Load required libraries
library(tidyverse)

# Source the falsification methods
source("falsification_methods.R")

# ============================================================================
# Load COMPAS Dataset from OpenML
# ============================================================================

cat("Loading COMPAS dataset from OpenML...\n")

# Check if OpenML is installed, if not, provide instructions
if (!require(OpenML, quietly = TRUE)) {
  cat("OpenML package not found. Please install it with:\n")
  cat("  install.packages('OpenML')\n")
  cat("  library(OpenML)\n")
  stop("OpenML package required")
}

# Download COMPAS dataset
cat("Downloading COMPAS dataset (data.id = 42193)...\n")
ds <- getOMLDataSet(data.id = 42193)  # compas-two-years
compas_data <- ds$data

cat("Dataset loaded successfully.\n")
cat("  Number of rows:", nrow(compas_data), "\n")
cat("  Number of columns:", ncol(compas_data), "\n")
cat("  Column names:", paste(colnames(compas_data), collapse = ", "), "\n\n")

# ============================================================================
# Data Preparation
# ============================================================================

cat("Preparing data for analysis...\n")

# Check required columns
if (!"two_year_recid" %in% colnames(compas_data)) {
  stop("ERROR: Required column 'two_year_recid' not found in COMPAS dataset")
}

# Check for race columns (may be one-hot encoded)
race_cols <- grep("^race_", colnames(compas_data), value = TRUE)
if (length(race_cols) == 0) {
  # Try to find a single race column
  if (!"race" %in% colnames(compas_data)) {
    stop("ERROR: No race columns found in COMPAS dataset")
  }
}

# Inspect the data
cat("Inspecting key variables:\n")
cat("  two_year_recid:\n")
cat("    Class:", class(compas_data$two_year_recid), "\n")
cat("    Unique values:", paste(unique(compas_data$two_year_recid[!is.na(compas_data$two_year_recid)]), collapse = ", "), "\n")
cat("    Summary:\n")
print(summary(compas_data$two_year_recid))
cat("\n")

# Inspect race columns
if (length(race_cols) > 0) {
  cat("  Race columns (one-hot encoded):\n")
  for (col in race_cols) {
    cat("    ", col, ":", sum(compas_data[[col]] == 1, na.rm = TRUE), "samples\n")
  }
} else {
  cat("  race:\n")
  cat("    Class:", class(compas_data$race), "\n")
  cat("    Unique values:", paste(unique(compas_data$race[!is.na(compas_data$race)]), collapse = ", "), "\n")
  cat("    Summary:\n")
  print(table(compas_data$race, useNA = "ifany"))
}
cat("\n")

# Convert two_year_recid to binary if needed
if (!is.numeric(compas_data$two_year_recid) || 
    !all(compas_data$two_year_recid %in% c(0, 1, NA))) {
  cat("Converting two_year_recid to binary...\n")
  # Try to convert to numeric
  compas_data$two_year_recid <- as.numeric(as.character(compas_data$two_year_recid))
  # If still not binary, create binary version
  if (!all(compas_data$two_year_recid %in% c(0, 1, NA), na.rm = TRUE)) {
    # Use median split or other method
    median_val <- median(compas_data$two_year_recid, na.rm = TRUE)
    compas_data$two_year_recid <- as.numeric(compas_data$two_year_recid > median_val)
    cat("  Created binary version using median split (median =", median_val, ")\n")
  }
}

# Ensure two_year_recid is numeric binary
compas_data$two_year_recid <- as.numeric(compas_data$two_year_recid)
cat("  Final two_year_recid: proportion = 1:", mean(compas_data$two_year_recid == 1, na.rm = TRUE), "\n\n")

# Prepare impermissible proxy: age < 25 only
cat("Creating impermissible proxy: age_cat_Lessthan25\n")

# Check for age category column
if (!"age_cat_Lessthan25" %in% colnames(compas_data)) {
  stop("ERROR: age_cat_Lessthan25 column not found in COMPAS dataset")
}

# Create impermissible proxy: age < 25 only
compas_data$impermissible_proxy <- as.numeric(compas_data$age_cat_Lessthan25 == 1)

cat("  Proportion age < 25 (impermissible proxy):", 
    mean(compas_data$impermissible_proxy, na.rm = TRUE), "\n\n")

# Remove rows with missing values in key variables
complete_cases <- complete.cases(compas_data[, c("two_year_recid", "impermissible_proxy")])
compas_data <- compas_data[complete_cases, ]
cat("After removing missing values:\n")
cat("  Number of rows:", nrow(compas_data), "\n\n")

# ============================================================================
# Select Features for Model Training
# ============================================================================

cat("Selecting features for model training...\n")

# Use the explicit feature set from the paper (Appendix):
# "sex, age, juvenile felony count, juvenile misdemeanor count,
#  juvenile other count, prior offenses count, and charge degree indicators"
# Excludes race-related variables and age category indicators.
paper_features <- c("sex", "age", "juv_fel_count", "juv_misd_count",
                     "juv_other_count", "priors_count",
                     "c_charge_degree_F", "c_charge_degree_M")
feature_cols <- intersect(paper_features, colnames(compas_data))

if (length(feature_cols) < length(paper_features)) {
  missing <- setdiff(paper_features, colnames(compas_data))
  cat("WARNING: Missing expected features:", paste(missing, collapse = ", "), "\n")
  cat("  Available columns:", paste(colnames(compas_data), collapse = ", "), "\n")
}

cat("Selected", length(feature_cols), "features:\n")
cat("  ", paste(feature_cols, collapse = ", "), "\n\n")

# ============================================================================
# Data Splitting
# ============================================================================

cat("Splitting data into training, calibration, and evaluation sets...\n")

set.seed(42)
n <- nrow(compas_data)
train_prop <- 0.6
calib_prop <- 0.2
eval_prop <- 0.2

# Create indices
all_indices <- 1:n
train_indices <- sample(all_indices, size = floor(n * train_prop))
remaining_indices <- setdiff(all_indices, train_indices)
calib_indices <- sample(remaining_indices, size = floor(n * calib_prop))
eval_indices <- setdiff(remaining_indices, calib_indices)

cat("Data split:\n")
cat("  Training set:", length(train_indices), "samples (", 
    sprintf("%.1f%%", 100 * length(train_indices) / n), ")\n")
cat("  Calibration set:", length(calib_indices), "samples (", 
    sprintf("%.1f%%", 100 * length(calib_indices) / n), ")\n")
cat("  Evaluation set:", length(eval_indices), "samples (", 
    sprintf("%.1f%%", 100 * length(eval_indices) / n), ")\n\n")

# ============================================================================
# Baseline Analysis: Using All Features
# ============================================================================

k <- length(feature_cols)
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("BASELINE ANALYSIS: USING ALL FEATURES\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")
cat("Total number of features:", k, "\n")
cat("Features:", paste(feature_cols, collapse = ", "), "\n\n")

# Run baseline analysis with all features
cat(paste0(rep("-", 80), collapse = ""), "\n")
cat("BASELINE: Using all", k, "features\n")
cat(paste0(rep("-", 80), collapse = ""), "\n\n")

# Train model with all features
train_data <- compas_data[train_indices, ]
formula_str <- paste("two_year_recid ~", paste(feature_cols, collapse = " + "))
formula_obj <- as.formula(formula_str)

model <- glm(formula_obj, data = train_data, family = binomial, 
             control = list(maxit = 1000))

# Get predictions on calibration and evaluation sets 
calib_eval_data <- compas_data[c(calib_indices, eval_indices), ]
predictions_raw <- predict(model, newdata = calib_eval_data, type = "response")

# Get true labels for calib+eval set
y_true_calib_eval <- compas_data$two_year_recid[c(calib_indices, eval_indices)]

# Compute log loss and AUC on calib+eval set
log_loss_calib_eval <- mean(log_loss(predictions_raw, y_true_calib_eval))

# Compute AUC manually
sorted_indices <- order(predictions_raw, decreasing = TRUE)
sorted_labels <- y_true_calib_eval[sorted_indices]
n_pos <- sum(y_true_calib_eval == 1)
n_neg <- sum(y_true_calib_eval == 0)

if (n_pos > 0 && n_neg > 0) {
  tp <- cumsum(sorted_labels == 1)
  fp <- cumsum(sorted_labels == 0)
  tpr <- tp / n_pos
  fpr <- fp / n_neg
  auc_calib_eval <- sum((fpr[-1] - fpr[-length(fpr)]) * (tpr[-1] + tpr[-length(tpr)]) / 2)
} else {
  auc_calib_eval <- NA
}

cat("Model performance on calib+eval set:\n")
cat("  Log loss:", sprintf("%.4f", log_loss_calib_eval), "\n")
if (!is.na(auc_calib_eval)) {
  cat("  AUC:", sprintf("%.4f", auc_calib_eval), "\n")
} else {
  cat("  AUC: N/A\n")
}
cat("\n")

# Prepare indices for falsification functions
calib_indices_rel <- 1:length(calib_indices)
eval_indices_rel <- (length(calib_indices) + 1):(length(calib_indices) + length(eval_indices))

# Prepare proxies
impermissible_proxy <- calib_eval_data$impermissible_proxy
permissible_proxy <- calib_eval_data$two_year_recid

# Run falsification test
result_baseline <- tryCatch({
  falsify_single_proxy(
    predictions = predictions_raw,
    impermissible_labels = impermissible_proxy,
    permissible_labels = permissible_proxy,
    calib_indices = calib_indices_rel,
    eval_indices = eval_indices_rel
  )
}, error = function(e) {
  cat("ERROR in falsification test:", e$message, "\n")
  return(list(
    result = "ERROR",
    p_value = NA,
    delta_mean = NA,
    test_used = "error"
  ))
})

cat("FALSIFICATION TEST RESULTS (BASELINE - ALL FEATURES):\n")
if (!is.null(result_baseline) && length(result_baseline) > 0) {
  test_stat_baseline <- if (!is.null(result_baseline$delta_mean) && length(result_baseline$delta_mean) > 0 && !is.na(result_baseline$delta_mean)) {
    result_baseline$delta_mean
  } else { NA }
  
  p_val_baseline <- if (!is.null(result_baseline$p_value) && length(result_baseline$p_value) > 0 && !is.na(result_baseline$p_value)) {
    result_baseline$p_value
  } else { NA }
  
  res_baseline <- if (!is.null(result_baseline$result) && length(result_baseline$result) > 0) {
    as.character(result_baseline$result)
  } else { "UNKNOWN" }
  
  cat("  Test statistic (mean difference):", 
      if (!is.na(test_stat_baseline)) sprintf("%.4f", test_stat_baseline) else "N/A", "\n")
  cat("  P-value:", 
      if (!is.na(p_val_baseline)) sprintf("%.4f", p_val_baseline) else "N/A", "\n")
  cat("  Result:", res_baseline, "\n\n")
} else {
  cat("  ERROR: Falsification test returned empty result\n\n")
  test_stat_baseline <- NA
  p_val_baseline <- NA
  res_baseline <- "ERROR"
}

# ============================================================================
# Leave-One-Out Sensitivity Analysis
# ============================================================================
# For each of k predictors, drop one and rerun the analysis

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("LEAVE-ONE-OUT SENSITIVITY ANALYSIS\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")
cat("Total number of features:", k, "\n")
cat("Will run", k, "analyses, each dropping one feature.\n\n")

# Store results for all analyses (including baseline)
all_results <- data.frame(
  dropped_feature = character(k + 1),
  p_value = numeric(k + 1),
  test_statistic = numeric(k + 1),
  result = character(k + 1),
  stringsAsFactors = FALSE)

# Store baseline result
all_results$dropped_feature[1] <- "NONE (all features)"
all_results$p_value[1] <- if (!is.na(p_val_baseline)) p_val_baseline else NA_real_
all_results$test_statistic[1] <- if (!is.na(test_stat_baseline)) test_stat_baseline else NA_real_
all_results$result[1] <- res_baseline

# Loop through each feature, dropping one at a time
# Start from index 2 since index 1 is the baseline
for (i in 1:k) {
  dropped_feature <- feature_cols[i]
  features_to_use <- setdiff(feature_cols, dropped_feature)
  
  cat(paste0(rep("-", 80), collapse = ""), "\n")
  cat("ANALYSIS", i + 1, "OF", k + 1, ": Dropping feature '", dropped_feature, "'\n")
  cat(paste0(rep("-", 80), collapse = ""), "\n\n")
  cat("Features used (", length(features_to_use), "): ", 
      paste(features_to_use, collapse = ", "), "\n\n")
  
  # ============================================================================
  # Train Predictive Model on Training Set
  # ============================================================================
  
  cat("Training predictive model...\n")
  cat("  Modeled outcome: two_year_recid (two-year recidivism)\n")
  
  train_data <- compas_data[train_indices, ]
  
  # Create formula with selected features (excluding dropped feature)
  if (length(features_to_use) > 0) {
    formula_str <- paste("two_year_recid ~", paste(features_to_use, collapse = " + "))
    formula_obj <- as.formula(formula_str)
  } else {
    # Fallback: use a simple model if no features available
    cat("WARNING: No features available. Using intercept-only model.\n")
    formula_obj <- two_year_recid ~ 1
  }
  
  model <- glm(formula_obj, data = train_data, family = binomial, 
               control = list(maxit = 1000))

# Get predictions on calibration and evaluation sets 
calib_eval_data <- compas_data[c(calib_indices, eval_indices), ]
predictions_raw <- predict(model, newdata = calib_eval_data, type = "response")

# Get true labels for calib+eval set
y_true_calib_eval <- calib_eval_data$two_year_recid

# Compute log loss and AUC on calib+eval set
log_loss_calib_eval <- mean(log_loss(predictions_raw, y_true_calib_eval))

# Compute AUC manually (area under ROC curve)
sorted_indices <- order(predictions_raw, decreasing = TRUE)
sorted_labels <- y_true_calib_eval[sorted_indices]
n_pos <- sum(y_true_calib_eval == 1)
n_neg <- sum(y_true_calib_eval == 0)

if (n_pos > 0 && n_neg > 0) {
  tp <- cumsum(sorted_labels == 1)
  fp <- cumsum(sorted_labels == 0)
  tpr <- tp / n_pos
  fpr <- fp / n_neg
  auc_calib_eval <- sum((fpr[-1] - fpr[-length(fpr)]) * (tpr[-1] + tpr[-length(tpr)]) / 2)
} else {
  auc_calib_eval <- NA
}

  cat("Model performance on calib+eval set:\n")
  cat("  Log loss:", sprintf("%.4f", log_loss_calib_eval), "\n")
  if (!is.na(auc_calib_eval)) {
    cat("  AUC:", sprintf("%.4f", auc_calib_eval), "\n")
  } else {
    cat("  AUC: N/A\n")
  }
  cat("\n")
  
  # ============================================================================
  # Prepare Indices for Falsification Functions
  # ============================================================================
  
  # Relative indices within calib_eval_data
  calib_indices_rel <- 1:length(calib_indices)
  eval_indices_rel <- (length(calib_indices) + 1):(length(calib_indices) + length(eval_indices))
  
  # ============================================================================
  # Prepare Proxies
  # ============================================================================
  
  # Impermissible proxy: intersection of race=African-American AND age < 25
  impermissible_proxy <- calib_eval_data$impermissible_proxy
  impermissible_labels_eval <- impermissible_proxy[eval_indices_rel]
  
  # Permissible proxy: two_year_recid (same as modeled outcome)
  permissible_proxy <- calib_eval_data$two_year_recid
  permissible_labels_eval <- permissible_proxy[eval_indices_rel]
  
  # ============================================================================
  # Test Algorithm 1: Single Permissible Proxy
  # ============================================================================
  
  result_single <- tryCatch({
    falsify_single_proxy(
      predictions = predictions_raw,
      impermissible_labels = impermissible_proxy,
      permissible_labels = permissible_proxy,
      calib_indices = calib_indices_rel,
      eval_indices = eval_indices_rel
    )
  }, error = function(e) {
    cat("ERROR in falsification test:", e$message, "\n")
    return(list(
      result = "ERROR",
      p_value = NA,
      test_statistic = NA,
      test_used = "error"
    ))
  })
  
  cat("FALSIFICATION TEST RESULTS:\n")
  
  # Check if result_single has the expected structure
  if (is.null(result_single) || length(result_single) == 0) {
    cat("  ERROR: Falsification test returned empty result\n")
    all_results$dropped_feature[i] <- dropped_feature
    all_results$p_value[i] <- NA
    all_results$test_statistic[i] <- NA
    all_results$result[i] <- "ERROR"
    cat("\n")
    next
  }
  
  # Safely extract values with defaults
  # Note: falsify_single_proxy returns delta_mean (mean of loss differences), not test_statistic
  test_stat <- if (!is.null(result_single$delta_mean) && length(result_single$delta_mean) > 0 && !is.na(result_single$delta_mean)) {
    result_single$delta_mean
  } else {
    NA
  }
  
  p_val <- if (!is.null(result_single$p_value) && length(result_single$p_value) > 0 && !is.na(result_single$p_value)) {
    result_single$p_value
  } else {
    NA
  }
  
  res <- if (!is.null(result_single$result) && length(result_single$result) > 0) {
    as.character(result_single$result)
  } else {
    "UNKNOWN"
  }
  
  cat("  Test statistic (mean difference):", 
      if (!is.na(test_stat)) sprintf("%.4f", test_stat) else "N/A", "\n")
  cat("  P-value:", 
      if (!is.na(p_val)) sprintf("%.4f", p_val) else "N/A", "\n")
  cat("  Result:", res, "\n\n")
  
  # Store results (index i+1 since index 1 is baseline)
  all_results$dropped_feature[i + 1] <- dropped_feature
  all_results$p_value[i + 1] <- if (!is.na(p_val)) p_val else NA_real_
  all_results$test_statistic[i + 1] <- if (!is.na(test_stat)) test_stat else NA_real_
  all_results$result[i + 1] <- res
  
  cat("\n")
}

# ============================================================================
# Summary of All Analyses
# ============================================================================

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("SUMMARY OF ALL LEAVE-ONE-OUT ANALYSES\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

cat("Results for each dropped feature:\n\n")
print(all_results)

cat("\nSummary statistics:\n")
cat("  Number of analyses:", nrow(all_results), " (1 baseline +", k, "leave-one-out)\n")
cat("  P-value range: [", sprintf("%.4f", min(all_results$p_value, na.rm = TRUE)), ", ", 
    sprintf("%.4f", max(all_results$p_value, na.rm = TRUE)), "]\n", sep = "")
cat("  Mean p-value:", sprintf("%.4f", mean(all_results$p_value, na.rm = TRUE)), "\n")
cat("  Median p-value:", sprintf("%.4f", median(all_results$p_value, na.rm = TRUE)), "\n")
cat("  Baseline (all features) p-value:", 
    sprintf("%.4f", all_results$p_value[1]), "\n")
cat("  Number of 'DISCRIMINANT' results:", sum(all_results$result == "DISCRIMINANT", na.rm = TRUE), "\n")
cat("  Number of 'INDISCRIMINATE (inconclusive)' results:", 
    sum(all_results$result == "INDISCRIMINATE (inconclusive)", na.rm = TRUE), "\n\n")

# Save results to CSV
output_file <- "compas_leave_one_out_results_age_only.csv"
write.csv(all_results, file = output_file, row.names = FALSE)
cat("Results saved to:", output_file, "\n\n")

# ============================================================================
# Final Summary
# ============================================================================

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("ANALYSIS COMPLETE\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

cat("Dataset: COMPAS (compas-two-years)\n")
cat("Modeled outcome: two_year_recid\n")
cat("Impermissible proxy: age_cat_Lessthan25 (age < 25 only)\n")
cat("Permissible proxy: two_year_recid (same as modeled outcome)\n")
cat("Test performed: Algorithm 1 (Single Permissible Proxy)\n")
cat("Sensitivity analysis: Leave-one-out (dropped each of", k, "features)\n\n")

