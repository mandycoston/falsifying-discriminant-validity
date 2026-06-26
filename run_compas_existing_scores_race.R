# ============================================================================
# COMPAS Dataset Analysis: Falsification Tests on Existing COMPAS Scores
# ============================================================================
# 
# This script performs falsification tests on the COMPAS dataset using
# the existing COMPAS risk scores (rather than training a new model).
# 
# Modeled outcome: two_year_recid (two-year recidivism)
# Impermissible proxy: race (African-American)
# Permissible proxy: two_year_recid (same as modeled outcome)
# 
# Uses Algorithm 1 (single proxy test)
# ============================================================================

# Load required libraries
library(tidyverse)

# Source the falsification methods
source("falsification_methods.R")

# ============================================================================
# Load COMPAS Dataset from ProPublica GitHub Repository
# ============================================================================

cat("Loading COMPAS dataset from ProPublica GitHub repository...\n")

# Set repository path (relative to script location or absolute)
repo_dir <- "compas-analysis"
data_file <- file.path(repo_dir, "compas-scores-two-years.csv")

# Check if repository exists, if not clone it
if (!dir.exists(repo_dir)) {
  cat("Repository not found. Cloning from GitHub...\n")
  system(paste("git clone https://github.com/propublica/compas-analysis.git", repo_dir))
  if (!dir.exists(repo_dir)) {
    stop("ERROR: Failed to clone repository. Please run manually:\n  git clone https://github.com/propublica/compas-analysis.git")
  }
  cat("Repository cloned successfully.\n\n")
} else {
  cat("Repository found at:", repo_dir, "\n")
}

# Check if data file exists
if (!file.exists(data_file)) {
  # Try alternative file names
  alt_files <- c(
    file.path(repo_dir, "compas-scores.csv"),
    file.path(repo_dir, "compas-scores-raw.csv"),
    file.path(repo_dir, "compas.csv")
  )
  
  found_file <- NULL
  for (alt_file in alt_files) {
    if (file.exists(alt_file)) {
      found_file <- alt_file
      break
    }
  }
  
  if (is.null(found_file)) {
    cat("ERROR: Data file not found. Looking in:", repo_dir, "\n")
    cat("Available files:\n")
    print(list.files(repo_dir, pattern = "\\.csv$", full.names = TRUE))
    stop("Please check the repository structure and update the file path.")
  } else {
    data_file <- found_file
    cat("Using alternative file:", data_file, "\n")
  }
}

# Load the dataset
cat("Loading data from:", data_file, "\n")
compas_data <- read.csv(data_file, stringsAsFactors = FALSE)

cat("Dataset loaded successfully.\n")
cat("  Number of rows:", nrow(compas_data), "\n")
cat("  Number of columns:", ncol(compas_data), "\n")
cat("  Column names:", paste(colnames(compas_data), collapse = ", "), "\n\n")

# ============================================================================
# Identify COMPAS Score Columns
# ============================================================================

cat("Identifying COMPAS score columns...\n")

# Common COMPAS score column names to check
possible_score_cols <- c(
  "decile_score",           # Decile score (1-10)
  "score_text",             # Text score (Low/Medium/High)
  "score",                  # Generic score
  "compas_score",           # Generic COMPAS score
  "risk_score",             # Risk score
  "decile_score.1",         # Alternative naming
  "decile_score.2"          # Alternative naming
)

# Find which score columns exist
score_cols_found <- intersect(possible_score_cols, colnames(compas_data))

if (length(score_cols_found) == 0) {
  cat("WARNING: No standard COMPAS score columns found.\n")
  cat("Available columns:\n")
  print(colnames(compas_data))
  cat("\nPlease check the dataset and update the script with the correct column name.\n")
  stop("COMPAS score column not found")
}

cat("Found COMPAS score columns:", paste(score_cols_found, collapse = ", "), "\n")

# Use decile_score as primary (most common), fallback to first found
if ("decile_score" %in% score_cols_found) {
  score_col <- "decile_score"
} else {
  score_col <- score_cols_found[1]
  cat("Using", score_col, "as the COMPAS score column.\n")
}

cat("Using COMPAS score column:", score_col, "\n\n")

# Inspect the score column
cat("Inspecting COMPAS score column:\n")
cat("  Class:", class(compas_data[[score_col]]), "\n")
cat("  Unique values:", paste(sort(unique(compas_data[[score_col]][!is.na(compas_data[[score_col]])])), collapse = ", "), "\n")
cat("  Summary:\n")
print(summary(compas_data[[score_col]]))
cat("\n")

# ============================================================================
# Data Preparation
# ============================================================================

cat("Preparing data for analysis...\n")

# Check required columns - try different possible names
recid_col <- NULL
recid_possible <- c("two_year_recid", "two_year_recidivism", "is_recid", "recid")
for (col in recid_possible) {
  if (col %in% colnames(compas_data)) {
    recid_col <- col
    break
  }
}

if (is.null(recid_col)) {
  cat("ERROR: Recidivism column not found. Looking for one of:", 
      paste(recid_possible, collapse = ", "), "\n")
  cat("Available columns:", paste(colnames(compas_data), collapse = ", "), "\n")
  stop("Required recidivism column not found in COMPAS dataset")
}

# Rename to standard name for consistency
compas_data$two_year_recid <- compas_data[[recid_col]]
cat("Using recidivism column:", recid_col, "\n")

# Check for race column - try different possible names
race_col <- NULL
race_possible <- c("race", "Race", "RACE", "ethnicity", "Ethnicity")
for (col in race_possible) {
  if (col %in% colnames(compas_data)) {
    race_col <- col
    break
  }
}

# Also check for one-hot encoded race columns
race_onehot_cols <- grep("^race_", colnames(compas_data), value = TRUE, ignore.case = TRUE)
if (length(race_onehot_cols) > 0) {
  cat("Found one-hot encoded race columns:", paste(race_onehot_cols, collapse = ", "), "\n")
  # Check for African-American column
  aa_cols <- grep("African|Black|Afro", race_onehot_cols, value = TRUE, ignore.case = TRUE)
  if (length(aa_cols) > 0) {
    race_col <- aa_cols[1]
    cat("Using one-hot encoded race column:", race_col, "\n")
  }
}

if (is.null(race_col) && length(race_onehot_cols) == 0) {
  cat("ERROR: Race column not found. Looking for one of:", 
      paste(race_possible, collapse = ", "), "\n")
  cat("Available columns:", paste(colnames(compas_data), collapse = ", "), "\n")
  stop("Required race column not found in COMPAS dataset")
}

# Create impermissible proxy: African-American (or Black)
cat("Creating impermissible proxy from race column...\n")
if (race_col %in% colnames(compas_data)) {
  race_values <- unique(compas_data[[race_col]][!is.na(compas_data[[race_col]])])
  cat("  Unique race values:", paste(race_values, collapse = ", "), "\n")
  
  # Check if it's already binary (one-hot encoded)
  if (is.numeric(compas_data[[race_col]]) && 
      all(compas_data[[race_col]] %in% c(0, 1, NA))) {
    compas_data$impermissible_proxy <- as.numeric(compas_data[[race_col]] == 1)
    cat("  Using binary race column directly\n")
  } else {
    # Convert race to binary: 1 if African-American/Black, 0 otherwise
    race_str <- as.character(compas_data[[race_col]])
    compas_data$impermissible_proxy <- as.numeric(
      grepl("African|Black|Afro", race_str, ignore.case = TRUE)
    )
    cat("  Created binary indicator for African-American/Black\n")
  }
} else {
  stop("ERROR: Race column not found after processing")
}

cat("  Proportion African-American/Black (impermissible proxy):", 
    mean(compas_data$impermissible_proxy, na.rm = TRUE), "\n\n")

# Convert COMPAS scores to probabilities if needed
# COMPAS decile scores are typically 1-10, we'll convert to 0-1 scale
compas_scores_raw <- compas_data[[score_col]]

# Check if scores need conversion
if (is.character(compas_scores_raw) || is.factor(compas_scores_raw)) {
  # If text scores (Low/Medium/High), convert to numeric
  cat("Converting text scores to numeric...\n")
  score_mapping <- c("Low" = 0.2, "Medium" = 0.5, "High" = 0.8)
  compas_scores_raw <- as.numeric(score_mapping[as.character(compas_scores_raw)])
  cat("  Mapped text scores to probabilities\n")
} else if (is.numeric(compas_scores_raw)) {
  # If numeric, check range
  score_range <- range(compas_scores_raw, na.rm = TRUE)
  cat("Score range:", score_range[1], "to", score_range[2], "\n")
  
  # If scores are 1-10 (decile), convert to 0-1 scale
  if (score_range[1] >= 1 && score_range[2] <= 10) {
    cat("Converting decile scores (1-10) to probabilities (0-1)...\n")
    compas_scores_raw <- (compas_scores_raw - 1) / 9  # Map 1->0, 10->1
    cat("  Converted decile scores to probabilities\n")
  } else if (score_range[1] >= 0 && score_range[2] <= 1) {
    cat("Scores already in 0-1 range, using as-is.\n")
  } else {
    cat("WARNING: Scores are not in expected range. Normalizing to 0-1...\n")
    compas_scores_raw <- (compas_scores_raw - score_range[1]) / (score_range[2] - score_range[1])
  }
} else {
  stop("ERROR: Unable to process COMPAS score column type")
}

# Ensure scores are in valid probability range [0, 1]
compas_scores_raw <- pmax(pmin(compas_scores_raw, 1), 0)

cat("Final score statistics:\n")
cat("  Mean:", sprintf("%.4f", mean(compas_scores_raw, na.rm = TRUE)), "\n")
cat("  Median:", sprintf("%.4f", median(compas_scores_raw, na.rm = TRUE)), "\n")
cat("  Range: [", sprintf("%.4f", min(compas_scores_raw, na.rm = TRUE)), ", ", 
    sprintf("%.4f", max(compas_scores_raw, na.rm = TRUE)), "]\n\n", sep = "")

# Convert two_year_recid to binary if needed
if (!is.numeric(compas_data$two_year_recid) || 
    !all(compas_data$two_year_recid %in% c(0, 1, NA))) {
  cat("Converting two_year_recid to binary...\n")
  compas_data$two_year_recid <- as.numeric(as.character(compas_data$two_year_recid))
  if (!all(compas_data$two_year_recid %in% c(0, 1, NA), na.rm = TRUE)) {
    median_val <- median(compas_data$two_year_recid, na.rm = TRUE)
    compas_data$two_year_recid <- as.numeric(compas_data$two_year_recid > median_val)
    cat("  Created binary version using median split (median =", median_val, ")\n")
  }
}

# Ensure two_year_recid is numeric binary
compas_data$two_year_recid <- as.numeric(compas_data$two_year_recid)
cat("  Final two_year_recid: proportion = 1:", 
    mean(compas_data$two_year_recid == 1, na.rm = TRUE), "\n\n")

# Remove rows with missing values in key variables
complete_cases <- complete.cases(
  compas_data[, c("two_year_recid", "impermissible_proxy")],
  compas_scores_raw
)
compas_data <- compas_data[complete_cases, ]
compas_scores_raw <- compas_scores_raw[complete_cases]

cat("After removing missing values:\n")
cat("  Number of rows:", nrow(compas_data), "\n\n")

# ============================================================================
# Data Splitting
# ============================================================================

cat("Splitting data into calibration and evaluation sets...\n")

set.seed(42)
n <- nrow(compas_data)
calib_prop <- 0.2
eval_prop <- 0.8

# Create indices
all_indices <- 1:n
calib_size <- floor(n * calib_prop)
calib_indices <- sample(all_indices, size = calib_size)
eval_indices <- setdiff(all_indices, calib_indices)

cat("Data split:\n")
cat("  Calibration set:", length(calib_indices), "samples (", 
    sprintf("%.1f%%", 100 * length(calib_indices) / n), ")\n")
cat("  Evaluation set:", length(eval_indices), "samples (", 
    sprintf("%.1f%%", 100 * length(eval_indices) / n), ")\n\n")

# ============================================================================
# Prepare Data for Falsification Test
# ============================================================================

cat("Preparing data for falsification test...\n")

# Get COMPAS scores for calibration and evaluation sets
predictions_all <- compas_scores_raw

# Prepare proxies
impermissible_proxy <- compas_data$impermissible_proxy
permissible_proxy <- compas_data$two_year_recid

# Relative indices within the full dataset
calib_indices_rel <- calib_indices
eval_indices_rel <- eval_indices

cat("Data prepared:\n")
cat("  Total predictions:", length(predictions_all), "\n")
cat("  Impermissible proxy (African-American/Black): proportion = 1:", 
    mean(impermissible_proxy), "\n")
cat("  Permissible proxy (two_year_recid): proportion = 1:", 
    mean(permissible_proxy), "\n\n")

# ============================================================================
# Compute Baseline Performance Metrics
# ============================================================================

cat("Computing baseline performance metrics...\n")

# Performance on evaluation set
eval_predictions <- predictions_all[eval_indices_rel]
eval_labels <- permissible_proxy[eval_indices_rel]

# Compute log loss
log_loss_eval <- mean(log_loss(eval_predictions, eval_labels))

# Compute AUC manually
sorted_indices <- order(eval_predictions, decreasing = TRUE)
sorted_labels <- eval_labels[sorted_indices]
n_pos <- sum(eval_labels == 1)
n_neg <- sum(eval_labels == 0)

if (n_pos > 0 && n_neg > 0) {
  tp <- cumsum(sorted_labels == 1)
  fp <- cumsum(sorted_labels == 0)
  tpr <- tp / n_pos
  fpr <- fp / n_neg
  auc_eval <- sum((fpr[-1] - fpr[-length(fpr)]) * (tpr[-1] + tpr[-length(tpr)]) / 2)
} else {
  auc_eval <- NA
}

cat("COMPAS score performance on evaluation set:\n")
cat("  Log loss:", sprintf("%.4f", log_loss_eval), "\n")
if (!is.na(auc_eval)) {
  cat("  AUC:", sprintf("%.4f", auc_eval), "\n")
} else {
  cat("  AUC: N/A\n")
}
cat("\n")

# ============================================================================
# Run Falsification Test: Algorithm 1 (Single Permissible Proxy)
# ============================================================================

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("FALSIFICATION TEST: ALGORITHM 1 (SINGLE PERMISSIBLE PROXY)\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

cat("Testing whether COMPAS scores predict two_year_recid better than race (African-American/Black)...\n\n")

result_single <- tryCatch({
  falsify_single_proxy(
    predictions = predictions_all,
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

cat("FALSIFICATION TEST RESULTS:\n")

if (is.null(result_single) || length(result_single) == 0) {
  cat("  ERROR: Falsification test returned empty result\n\n")
  test_stat <- NA
  p_val <- NA
  res <- "ERROR"
} else {
  # Safely extract values
  test_stat <- if (!is.null(result_single$delta_mean) && 
                   length(result_single$delta_mean) > 0 && 
                   !is.na(result_single$delta_mean)) {
    result_single$delta_mean
  } else {
    NA
  }
  
  p_val <- if (!is.null(result_single$p_value) && 
               length(result_single$p_value) > 0 && 
               !is.na(result_single$p_value)) {
    result_single$p_value
  } else {
    NA
  }
  
  res <- if (!is.null(result_single$result) && 
             length(result_single$result) > 0) {
    as.character(result_single$result)
  } else {
    "UNKNOWN"
  }
  
  test_used <- if (!is.null(result_single$test_used) && 
                   length(result_single$test_used) > 0) {
    as.character(result_single$test_used)
  } else {
    "unknown"
  }
  
  cat("  Test statistic (mean loss difference):", 
      if (!is.na(test_stat)) sprintf("%.4f", test_stat) else "N/A", "\n")
  cat("  P-value:", 
      if (!is.na(p_val)) sprintf("%.6f", p_val) else "N/A", "\n")
  cat("  Test used:", test_used, "\n")
  cat("  Result:", res, "\n\n")
}

# ============================================================================
# Save Results
# ============================================================================

results_summary <- data.frame(
  analysis_type = "COMPAS_existing_scores_race",
  score_column = score_col,
  impermissible_proxy = "race_AfricanAmerican",
  permissible_proxy = "two_year_recid",
  n_total = n,
  n_calibration = length(calib_indices),
  n_evaluation = length(eval_indices),
  log_loss = log_loss_eval,
  auc = if (!is.na(auc_eval)) auc_eval else NA,
  test_statistic = if (!is.na(test_stat)) test_stat else NA,
  p_value = if (!is.na(p_val)) p_val else NA,
  test_used = if (exists("test_used")) test_used else "unknown",
  result = res,
  stringsAsFactors = FALSE
)

output_file <- "compas_existing_scores_results_race.csv"
write.csv(results_summary, file = output_file, row.names = FALSE)
cat("Results saved to:", output_file, "\n\n")

# ============================================================================
# Final Summary
# ============================================================================

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("ANALYSIS COMPLETE\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

cat("Dataset: COMPAS (compas-two-years)\n")
cat("COMPAS score column used:", score_col, "\n")
cat("Modeled outcome: two_year_recid\n")
cat("Impermissible proxy: race (African-American/Black)\n")
cat("Permissible proxy: two_year_recid (same as modeled outcome)\n")
cat("Test performed: Algorithm 1 (Single Permissible Proxy)\n")
cat("Analysis type: Using existing COMPAS scores (no model training)\n\n")

cat("Key Findings:\n")
cat("  COMPAS scores predict recidivism with AUC:", 
    if (!is.na(auc_eval)) sprintf("%.4f", auc_eval) else "N/A", "\n")
cat("  Falsification test result:", res, "\n")
if (!is.na(p_val)) {
  cat("  P-value:", sprintf("%.6f", p_val), "\n")
  if (p_val < 0.05) {
    cat("  Interpretation: COMPAS scores predict recidivism significantly better than race alone.\n")
  } else {
    cat("  Interpretation: Cannot establish that COMPAS scores predict recidivism better than race alone.\n")
  }
}
cat("\n")
