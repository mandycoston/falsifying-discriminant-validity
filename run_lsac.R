# Run Falsification Methods on LSAC Dataset
# This script runs the falsification methods on LSAC data

# Source the main functions
source("falsification_methods.R")
source("load_lsac_data.R")

# ============================================================================
# Configuration Parameters
# ============================================================================

# Set to FALSE to suppress detailed output during sensitivity analysis
# (only summary will be shown)
verbose_sensitivity <- TRUE

# Define which impermissible proxies to analyze
# Options: "race", "family_income", "gender"
impermissible_proxies_to_run <- c("race")

# Set seed for reproducibility
set.seed(42)

# ============================================================================
# Load and Prepare Data
# ============================================================================

cat("Loading LSAC dataset...\n")
lsac_data <- load_lsac_data()

# Display data structure
cat("\nData dimensions:", dim(lsac_data), "\n")
cat("Variables:", paste(names(lsac_data), collapse = ", "), "\n\n")

# Check for required columns
required_cols <- c("lsat", "gpa", "fygpa_binary", "gpa_binary", "pass_bar", "race", "male")
missing_cols <- setdiff(required_cols, colnames(lsac_data))

if (length(missing_cols) > 0) {
  cat("WARNING: Missing required columns:", paste(missing_cols, collapse = ", "), "\n")
  cat("Available columns:", paste(colnames(lsac_data), collapse = ", "), "\n")
  cat("Please check the dataset structure or update the column names.\n\n")
  
  # Try to use inspect function if available
  if (exists("inspect_lsac_data")) {
    inspect_lsac_data(lsac_data)
  }
  
  stop("Cannot proceed without required columns. Please fix the data loading.")
}

# ============================================================================
# Split Data into Training, Calibration, and Evaluation Sets
# ============================================================================

cat("Splitting data into training, calibration, and evaluation sets...\n")
n <- nrow(lsac_data)
train_prop <- 0.5   # 50% for training
calib_prop <- 0.25  # 25% for calibration
eval_prop <- 0.25   # 25% for evaluation

# Create indices for each fold
all_indices <- sample(n)  # Randomize order
train_size <- floor(n * train_prop)
calib_size <- floor(n * calib_prop)
eval_size <- n - train_size - calib_size  # Remaining goes to evaluation

train_indices <- all_indices[1:train_size]
calib_indices <- all_indices[(train_size + 1):(train_size + calib_size)]
eval_indices <- all_indices[(train_size + calib_size + 1):n]

cat("Data split:\n")
cat("  Training set:", length(train_indices), "samples\n")
cat("  Calibration set:", length(calib_indices), "samples\n")
cat("Evaluation set:", length(eval_indices),  "samples\n")
cat("\n")

# ============================================================================
# Train Predictive Model on Training Set
# ============================================================================

cat("Training predictive model on training set...\n")
cat("  Modeled outcome: first-year GPA (above/below median)\n")

# Train model to predict first-year GPA (fygpa_binary)
train_data <- lsac_data[train_indices, ]

model <- glm(fygpa_binary ~ lsat + ugpa, data = train_data, family = binomial, 
             control = list(maxit = 1000))

# Get predictions on calibration and evaluation sets 
calib_eval_data <- lsac_data[c(calib_indices, eval_indices), ]

predictions_raw <- predict(model, newdata = calib_eval_data, type = "response")

# Get true labels for calib+eval set (fygpa_binary)
y_true_calib_eval <- calib_eval_data$fygpa_binary

# Compute log loss and AUC on calib+eval set
log_loss_calib_eval <- mean(log_loss(predictions_raw, y_true_calib_eval))

# Compute AUC manually (area under ROC curve)
# Sort by predictions descending
sorted_indices <- order(predictions_raw, decreasing = TRUE)
sorted_labels <- y_true_calib_eval[sorted_indices]
n_pos <- sum(y_true_calib_eval == 1)
n_neg <- sum(y_true_calib_eval == 0)

if (n_pos > 0 && n_neg > 0) {
  # Compute true positive rate and false positive rate
  tp <- cumsum(sorted_labels == 1)
  fp <- cumsum(sorted_labels == 0)
  tpr <- tp / n_pos
  fpr <- fp / n_neg
  
  # Compute AUC using trapezoidal rule
  auc_calib_eval <- sum((fpr[-1] - fpr[-length(fpr)]) * (tpr[-1] + tpr[-length(tpr)]) / 2)
} else {
  auc_calib_eval <- NA
}

cat("Model trained. Mean prediction on calib+eval:", mean(predictions_raw), "\n")
cat("Model performance on calib+eval set:\n")
cat("  Log loss:", sprintf("%.4f", log_loss_calib_eval), "\n")
if (!is.na(auc_calib_eval)) {
  cat("  AUC:", sprintf("%.4f", auc_calib_eval), "\n")
} else {
  cat("  AUC: N/A (insufficient positive or negative samples)\n")
}
cat("\n")

# ============================================================================
# Train Random Forest Model
# ============================================================================

library(randomForest)

cat("Training random forest model on training set...\n")
rf_model <- randomForest(
  x = train_data[, c("lsat", "ugpa")],
  y = as.factor(train_data$fygpa_binary),
  ntree = 500,
  seed = 42
)

predictions_rf_raw <- predict(rf_model, newdata = calib_eval_data, type = "prob")[, "1"]
predictions_rf_raw <- pmax(pmin(predictions_rf_raw, 1 - 1e-15), 1e-15)

write.csv(data.frame(prediction = predictions_rf_raw),
          "predictions_lsac_rf_from_R.csv", row.names = FALSE)
cat("Random forest predictions saved to predictions_lsac_rf_from_R.csv\n\n")

# ============================================================================
# Helper Function: Run Full Analysis for an Impermissible Proxy
# ============================================================================

#' Run complete falsification analysis for a given impermissible proxy
#' 
#' @param proxy_name Name of the impermissible proxy ("race", "family_income", "gender")
#' @param proxy_label Display label (e.g., "Race", "Family Income", "Gender")
#' @param calib_eval_data Data for calibration and evaluation sets
#' @param predictions_raw Raw predictions from the model
#' @param calib_indices_rel Relative indices for calibration set
#' @param eval_indices_rel Relative indices for evaluation set
run_analysis_for_proxy <- function(proxy_name, proxy_label, calib_eval_data, 
                                   predictions_raw, calib_indices_rel, eval_indices_rel) {
  
  cat("\n")
  cat(paste0(rep("#", 80), collapse = ""), "\n")
  cat("ANALYSIS FOR IMPERMISSIBLE PROXY:", toupper(proxy_label), "\n")
  cat(paste0(rep("#", 80), collapse = ""), "\n\n")
  
  # Create binary impermissible proxy based on proxy_name
  # Note: First-year GPA is the modeled outcome. Race is the impermissible proxy.
  if (proxy_name == "race") {
    # Race: White = 1, non-White = 0
    race_vals <- as.character(calib_eval_data$race)
    race_vals_lower <- tolower(race_vals)
    impermissible_proxy <- as.numeric(
      race_vals_lower == "white" | 
      race_vals == "White" | 
      race_vals == "WHITE" |
      race_vals == "1" |
      grepl("white", race_vals_lower, ignore.case = TRUE)
    )
    cat("Impermissible proxy (Race: White):\n")
    cat("  Unique race values:", paste(unique(calib_eval_data$race), collapse = ", "), "\n")
    cat("  Proportion White:", mean(impermissible_proxy), "\n")
    if (mean(impermissible_proxy) == 0) {
      warning("WARNING: No White samples found! Check race column encoding.")
    }
  } else if (proxy_name == "family_income") {
    # Family Income: fam_inc is categorical, convert to binary (above median category = 1)
    if ("fam_inc" %in% colnames(calib_eval_data)) {
      income_data <- calib_eval_data$fam_inc
      cat("Impermissible proxy (Family Income):\n")
      cat("  Income variable: fam_inc\n")
      cat("  Variable class:", class(income_data), "\n")
      cat("  Unique values:", paste(unique(income_data[!is.na(income_data)]), collapse = ", "), "\n")
      
      # Convert categorical to numeric if needed (assuming ordered categories)
      if (!is.numeric(income_data)) {
        # Try to convert to numeric (works if categories are numeric strings)
        income_numeric <- suppressWarnings(as.numeric(as.character(income_data)))
        if (all(!is.na(income_numeric) | is.na(income_data))) {
          # Successfully converted to numeric
          income_data <- income_numeric
          cat("  Converted to numeric values\n")
        } else {
          # Treat as factor/character - use median split based on factor levels
          income_factor <- as.factor(income_data)
          income_levels <- levels(income_factor)
          n_levels <- length(income_levels)
          cat("  Treating as categorical with", n_levels, "levels\n")
          
          # Convert to numeric based on factor order, then split at median
          income_numeric <- as.numeric(income_factor)
          income_median <- median(income_numeric, na.rm = TRUE)
          impermissible_proxy <- as.numeric(income_numeric > income_median)
          cat("  Median category index:", income_median, "\n")
          cat("  Proportion above median category:", mean(impermissible_proxy, na.rm = TRUE), "\n")
        }
      }
      
      # If we have numeric data, split at median
      if (is.numeric(income_data) && !exists("impermissible_proxy")) {
        income_median <- median(income_data, na.rm = TRUE)
        impermissible_proxy <- as.numeric(income_data > income_median)
        cat("  Median income value:", income_median, "\n")
        cat("  Proportion above median:", mean(impermissible_proxy, na.rm = TRUE), "\n")
      }
      
      cat("\n")
    } else {
      cat("ERROR: fam_inc variable not found. Available columns:", 
          paste(colnames(calib_eval_data), collapse = ", "), "\n\n")
      return(NULL)
    }
  } else if (proxy_name == "gender") {
    # Gender: Male = 1, Female = 0
    if ("male" %in% colnames(calib_eval_data)) {
      impermissible_proxy <- as.numeric(calib_eval_data$male)
      cat("Impermissible proxy (Gender: Male):\n")
      cat("  Proportion Male:", mean(impermissible_proxy, na.rm = TRUE), "\n")
    } else {
      cat("ERROR: Male/gender variable not found. Skipping this proxy.\n\n")
      return(NULL)
    }
  } else {
    cat("ERROR: Unknown proxy name:", proxy_name, "\n\n")
    return(NULL)
  }
  cat("\n")
  
  # Permissible proxy: GPA (binary: above/below median)
  permissible_proxy <- calib_eval_data$gpa_binary
  cat("Permissible proxy (GPA, binary):\n")
  cat("  Proportion above median:", mean(permissible_proxy, na.rm = TRUE), "\n\n")
  
  # ============================================================================
  # Uncalibrated Performance Metrics on Evaluation Set
  # ============================================================================
  
  cat(paste0(rep("=", 60), collapse = ""), "\n")
  cat("UNCALIBRATED MODEL PERFORMANCE (Evaluation Set)\n")
  cat(paste0(rep("=", 60), collapse = ""), "\n\n")
  
  # Create indices relative to calib_eval_data (calibration is first, evaluation is second)
  n_calib_eval <- length(predictions_raw)
  calib_indices_rel <- 1:length(calib_indices)
  eval_indices_rel <- (length(calib_indices) + 1):n_calib_eval
  
  predictions_eval <- predictions_raw[eval_indices_rel]
  
  # Helper function to compute AUC
  compute_auc <- function(predictions, labels) {
    n_pos <- sum(labels == 1)
    n_neg <- sum(labels == 0)
    if (n_pos == 0 || n_neg == 0) return(NA)
    
    sorted_indices <- order(predictions, decreasing = TRUE)
    sorted_labels <- labels[sorted_indices]
    tp <- cumsum(sorted_labels == 1)
    fp <- cumsum(sorted_labels == 0)
    tpr <- tp / n_pos
    fpr <- fp / n_neg
    auc <- sum((fpr[-1] - fpr[-length(fpr)]) * (tpr[-1] + tpr[-length(tpr)]) / 2)
    return(auc)
  }
  
  # Helper function to compute AU PR (Area Under Precision-Recall curve)
  compute_aupr <- function(predictions, labels) {
    n_pos <- sum(labels == 1)
    if (n_pos == 0) return(NA)
    
    sorted_indices <- order(predictions, decreasing = TRUE)
    sorted_labels <- labels[sorted_indices]
    tp <- cumsum(sorted_labels == 1)
    fp <- cumsum(sorted_labels == 0)
    precision <- tp / (tp + fp)
    recall <- tp / n_pos
    
    # Remove duplicates and compute AU PR using trapezoidal rule
    # Keep only points where recall changes
    unique_recall <- unique(recall)
    if (length(unique_recall) < 2) return(NA)
    
    # For each unique recall value, get the maximum precision
    precision_at_recall <- sapply(unique_recall, function(r) {
      max(precision[recall == r], na.rm = TRUE)
    })
    
    # Sort by recall
    sorted_idx <- order(unique_recall)
    unique_recall <- unique_recall[sorted_idx]
    precision_at_recall <- precision_at_recall[sorted_idx]
    
    # Compute AU PR using trapezoidal rule
    aupr <- sum((unique_recall[-1] - unique_recall[-length(unique_recall)]) * 
                (precision_at_recall[-1] + precision_at_recall[-length(precision_at_recall)]) / 2)
    return(aupr)
  }
  
  # Helper function to compute MSE
  compute_mse <- function(predictions, labels) {
    mean((predictions - labels)^2)
  }
  
  # Helper function to compute PPV (Positive Predictive Value) at top k%
  # PPV = TP / (TP + FP) = P(Y = 1 | Y_hat in top k%)
  compute_ppv_topk <- function(predictions, labels, top_k_percent) {
    n <- length(predictions)
    k <- max(1, floor(n * top_k_percent / 100))
    
    # Get top k indices (those predicted as positive/high)
    top_k_indices <- order(predictions, decreasing = TRUE)[1:k]
    
    # Compute PPV: among top k predictions, what proportion are actually positive?
    top_k_labels <- labels[top_k_indices]
    ppv <- mean(top_k_labels == 1, na.rm = TRUE)
    return(ppv)
  }
  
  # Helper function to compute all metrics for a given prediction-label pair
  compute_all_metrics <- function(predictions, labels, proxy_name) {
    auc_val <- compute_auc(predictions, labels)
    aupr_val <- compute_aupr(predictions, labels)
    mse_val <- compute_mse(predictions, labels)
    ppv_top2 <- compute_ppv_topk(predictions, labels, 2)
    ppv_top10 <- compute_ppv_topk(predictions, labels, 10)
    ppv_top50 <- compute_ppv_topk(predictions, labels, 50)
    ppv_top75 <- compute_ppv_topk(predictions, labels, 75)
    
    return(data.frame(
      Proxy = proxy_name,
      AUC = ifelse(is.na(auc_val), NA, auc_val),
      AUPR = ifelse(is.na(aupr_val), NA, aupr_val),
      MSE = mse_val,
      PPV_Top2 = ppv_top2,
      PPV_Top10 = ppv_top10,
      PPV_Top50 = ppv_top50,
      PPV_Top75 = ppv_top75,
      stringsAsFactors = FALSE
    ))
  }
  
  # Modeled outcome (first-year GPA)
  y_true_eval <- calib_eval_data$fygpa_binary[eval_indices_rel]
  log_loss_modeled <- mean(log_loss(predictions_eval, y_true_eval))
  auc_modeled <- compute_auc(predictions_eval, y_true_eval)
  mse_modeled <- compute_mse(predictions_eval, y_true_eval)
  cat("Modeled outcome (first-year GPA):\n")
  cat("  Log loss:", sprintf("%.4f", log_loss_modeled), "\n")
  if (!is.na(auc_modeled)) {
    cat("  AUC:", sprintf("%.4f", auc_modeled), "\n")
  } else {
    cat("  AUC: N/A\n")
  }
  cat("  MSE:", sprintf("%.4f", mse_modeled), "\n\n")
  
  # Impermissible proxy
  impermissible_labels_eval <- impermissible_proxy[eval_indices_rel]
  log_loss_impermissible_uncal <- mean(log_loss(predictions_eval, impermissible_labels_eval))
  auc_impermissible_uncal <- compute_auc(predictions_eval, impermissible_labels_eval)
  mse_impermissible_uncal <- compute_mse(predictions_eval, impermissible_labels_eval)
  cat("Impermissible proxy (", proxy_label, "):\n", sep = "")
  cat("  Log loss:", sprintf("%.4f", log_loss_impermissible_uncal), "\n")
  if (!is.na(auc_impermissible_uncal)) {
    cat("  AUC:", sprintf("%.4f", auc_impermissible_uncal), "\n")
  } else {
    cat("  AUC: N/A\n")
  }
  cat("  MSE:", sprintf("%.4f", mse_impermissible_uncal), "\n\n")
  
  # Permissible proxy (single)
  permissible_labels_eval <- permissible_proxy[eval_indices_rel]
  log_loss_permissible_uncal <- mean(log_loss(predictions_eval, permissible_labels_eval))
  auc_permissible_uncal <- compute_auc(predictions_eval, permissible_labels_eval)
  mse_permissible_uncal <- compute_mse(predictions_eval, permissible_labels_eval)
  cat("Permissible proxy (GPA, binary):\n")
  cat("  Log loss:", sprintf("%.4f", log_loss_permissible_uncal), "\n")
  if (!is.na(auc_permissible_uncal)) {
    cat("  AUC:", sprintf("%.4f", auc_permissible_uncal), "\n")
  } else {
    cat("  AUC: N/A\n")
  }
  cat("  MSE:", sprintf("%.4f", mse_permissible_uncal), "\n\n")
  
  # ============================================================================
  # Fairness Metrics (using impermissible proxy as sensitive attribute)
  # ============================================================================
  
  cat(paste0(rep("=", 60), collapse = ""), "\n")
  cat("FAIRNESS METRICS (Sensitive Attribute: ", proxy_label, ")\n", sep = "")
  cat(paste0(rep("=", 60), collapse = ""), "\n\n")
  
  # Helper function to compute fairness metrics
  compute_fairness_metrics <- function(predictions, labels, sensitive_attribute, threshold = 0.5) {
    # Convert predictions to binary using threshold
    predictions_binary <- as.numeric(predictions >= threshold)
    
    # Get groups
    group_0 <- sensitive_attribute == 0
    group_1 <- sensitive_attribute == 1
    
    # Demographic Parity (Statistical Parity)
    # P(Y_hat = 1 | S = 0) vs P(Y_hat = 1 | S = 1)
    dp_group_0 <- mean(predictions_binary[group_0], na.rm = TRUE)
    dp_group_1 <- mean(predictions_binary[group_1], na.rm = TRUE)
    demographic_parity_diff <- abs(dp_group_0 - dp_group_1)
    
    # Equalized Odds
    # P(Y_hat = 1 | Y = y, S = 0) = P(Y_hat = 1 | Y = y, S = 1) for y in {0, 1}
    # True Positive Rate (TPR) difference
    tpr_group_0 <- sum(predictions_binary[group_0] == 1 & labels[group_0] == 1, na.rm = TRUE) / 
                    max(sum(labels[group_0] == 1, na.rm = TRUE), 1)
    tpr_group_1 <- sum(predictions_binary[group_1] == 1 & labels[group_1] == 1, na.rm = TRUE) / 
                    max(sum(labels[group_1] == 1, na.rm = TRUE), 1)
    tpr_diff <- abs(tpr_group_0 - tpr_group_1)
    
    # False Positive Rate (FPR) difference
    fpr_group_0 <- sum(predictions_binary[group_0] == 1 & labels[group_0] == 0, na.rm = TRUE) / 
                    max(sum(labels[group_0] == 0, na.rm = TRUE), 1)
    fpr_group_1 <- sum(predictions_binary[group_1] == 1 & labels[group_1] == 0, na.rm = TRUE) / 
                    max(sum(labels[group_1] == 0, na.rm = TRUE), 1)
    fpr_diff <- abs(fpr_group_0 - fpr_group_1)
    
    equalized_odds_diff <- max(tpr_diff, fpr_diff)
    
    # Equal Opportunity (TPR equality)
    equal_opportunity_diff <- tpr_diff
    
    # Calibration (Positive Predictive Value equality)
    # P(Y = 1 | Y_hat = 1, S = 0) vs P(Y = 1 | Y_hat = 1, S = 1)
    ppv_group_0 <- sum(labels[group_0] == 1 & predictions_binary[group_0] == 1, na.rm = TRUE) / 
                   max(sum(predictions_binary[group_0] == 1, na.rm = TRUE), 1)
    ppv_group_1 <- sum(labels[group_1] == 1 & predictions_binary[group_1] == 1, na.rm = TRUE) / 
                   max(sum(predictions_binary[group_1] == 1, na.rm = TRUE), 1)
    calibration_diff <- abs(ppv_group_0 - ppv_group_1)
    
    return(list(
      demographic_parity = list(
        group_0 = dp_group_0,
        group_1 = dp_group_1,
        difference = demographic_parity_diff
      ),
      equalized_odds = list(
        tpr_group_0 = tpr_group_0,
        tpr_group_1 = tpr_group_1,
        tpr_diff = tpr_diff,
        fpr_group_0 = fpr_group_0,
        fpr_group_1 = fpr_group_1,
        fpr_diff = fpr_diff,
        max_diff = equalized_odds_diff
      ),
      equal_opportunity = list(
        tpr_group_0 = tpr_group_0,
        tpr_group_1 = tpr_group_1,
        difference = equal_opportunity_diff
      ),
      calibration = list(
        ppv_group_0 = ppv_group_0,
        ppv_group_1 = ppv_group_1,
        difference = calibration_diff
      )
    ))
  }
  
  # Fairness metrics for modeled outcome
  fairness_modeled <- compute_fairness_metrics(
    predictions = predictions_eval,
    labels = y_true_eval,
    sensitive_attribute = impermissible_labels_eval
  )
  
  cat("Fairness metrics for modeled outcome (first-year GPA):\n")
  cat("  Demographic Parity:\n")
  cat("    Group 0 (", proxy_label, " = 0):", sprintf("%.4f", fairness_modeled$demographic_parity$group_0), "\n", sep = "")
  cat("    Group 1 (", proxy_label, " = 1):", sprintf("%.4f", fairness_modeled$demographic_parity$group_1), "\n", sep = "")
  cat("    Difference:", sprintf("%.4f", fairness_modeled$demographic_parity$difference), "\n")
  cat("  Equalized Odds:\n")
  cat("    TPR Group 0:", sprintf("%.4f", fairness_modeled$equalized_odds$tpr_group_0), "\n")
  cat("    TPR Group 1:", sprintf("%.4f", fairness_modeled$equalized_odds$tpr_group_1), "\n")
  cat("    TPR Difference:", sprintf("%.4f", fairness_modeled$equalized_odds$tpr_diff), "\n")
  cat("    FPR Group 0:", sprintf("%.4f", fairness_modeled$equalized_odds$fpr_group_0), "\n")
  cat("    FPR Group 1:", sprintf("%.4f", fairness_modeled$equalized_odds$fpr_group_1), "\n")
  cat("    FPR Difference:", sprintf("%.4f", fairness_modeled$equalized_odds$fpr_diff), "\n")
  cat("    Max Difference:", sprintf("%.4f", fairness_modeled$equalized_odds$max_diff), "\n")
  cat("  Equal Opportunity (TPR equality):\n")
  cat("    TPR Group 0:", sprintf("%.4f", fairness_modeled$equal_opportunity$tpr_group_0), "\n")
  cat("    TPR Group 1:", sprintf("%.4f", fairness_modeled$equal_opportunity$tpr_group_1), "\n")
  cat("    Difference:", sprintf("%.4f", fairness_modeled$equal_opportunity$difference), "\n")
  cat("  Calibration (PPV equality):\n")
  cat("    PPV Group 0:", sprintf("%.4f", fairness_modeled$calibration$ppv_group_0), "\n")
  cat("    PPV Group 1:", sprintf("%.4f", fairness_modeled$calibration$ppv_group_1), "\n")
  cat("    Difference:", sprintf("%.4f", fairness_modeled$calibration$difference), "\n\n")
  
  # ============================================================================
  # Test Algorithm 1: Single Permissible Proxy
  # ============================================================================
  
  cat(paste0(rep("=", 60), collapse = ""), "\n")
  cat("ALGORITHM 1: FALSIFICATION WITH SINGLE PERMISSIBLE PROXY\n")
  cat(paste0(rep("=", 60), collapse = ""), "\n\n")
  cat("Modeled outcome: first-year GPA (above/below median)\n")
  cat("Single permissible proxy: GPA (binary, above/below median)\n")
  cat("Impermissible proxy:", proxy_label, "\n\n")
  
  result_single <- falsify_single_proxy(
    predictions = predictions_raw,
    impermissible_labels = impermissible_proxy,
    permissible_labels = permissible_proxy,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices = eval_indices_rel
  )
  
  cat("RESULTS:\n")
  cat("  Result:", result_single$result, "\n")
  cat("  P-value:", format_pvalue(result_single$p_value, z = result_single$z_statistic), "\n")
  cat("  Test used:", result_single$test_used, "\n")
  cat("  Mean Delta (loss_impermissible - loss_permissible):", 
      sprintf("%.4f", result_single$delta_mean), "\n")
  cat("  Median Delta:", sprintf("%.4f", result_single$delta_median), "\n")
  cat("  Data appears normal:", result_single$is_normal, "\n")
  cat("  Has outliers:", result_single$has_outliers, "\n\n")
  
  # Interpretation
  if (result_single$result == "DISCRIMINANT") {
    cat("INTERPRETATION: The algorithm appears to predict the permissible proxy\n")
    cat("(GPA, binary) better than the impermissible proxy (", proxy_label, "), suggesting it may\n", sep = "")
    cat("not be discriminating based on", proxy_label, ".\n\n")
  } else {
    cat("INTERPRETATION: The test is inconclusive. The algorithm may be\n")
    cat("indiscriminate, or there may be insufficient power to detect a difference.\n\n")
  }
  
  # ============================================================================
  # Test Algorithm 2: Multiple Permissible Proxies
  # ============================================================================
  
  cat(paste0(rep("=", 60), collapse = ""), "\n")
  cat("ALGORITHM 2: FALSIFICATION WITH MULTIPLE PERMISSIBLE PROXIES\n")
  cat(paste0(rep("=", 60), collapse = ""), "\n\n")
  
  # Create multiple permissible proxies (including fygpa for the multiple proxy test)
  permissible_matrix <- cbind(
    fygpa = calib_eval_data$fygpa_binary,
    gpa = calib_eval_data$gpa_binary,
    pass_bar = calib_eval_data$pass_bar
  )
  
  cat("Using", ncol(permissible_matrix), "permissible proxies:\n")
  cat("  1. FYGPA (binary, above/below median)\n")
  cat("  2. GPA (binary, above/below median)\n")
  cat("  3. Bar passage on first attempt (binary, passed/failed)\n\n")
  
  # Compute calibrated predictions and performance metrics
  cat("Computing calibrated predictions and performance metrics...\n")
  platt_impermissible <- platt_scaling(predictions_raw, impermissible_proxy, calib_indices_rel)
  calib_pred_impermissible_eval <- platt_impermissible$calibrated[eval_indices_rel]
  
  platt_permissible_list <- list()
  calib_pred_permissible_eval_list <- list()
  for (j in 1:ncol(permissible_matrix)) {
    platt_permissible_list[[j]] <- platt_scaling(predictions_raw, permissible_matrix[, j], calib_indices_rel)
    calib_pred_permissible_eval_list[[j]] <- platt_permissible_list[[j]]$calibrated[eval_indices_rel]
  }
  
  
  # ============================================================================
  # Calibrated Performance Metrics on Evaluation Set
  # ============================================================================
  
  cat(paste0(rep("=", 60), collapse = ""), "\n")
  cat("CALIBRATED MODEL PERFORMANCE (Evaluation Set)\n")
  cat(paste0(rep("=", 60), collapse = ""), "\n\n")
  
  # Define proxy names (matching permissible_matrix columns)
  proxy_names <- c("fygpa", "GPA", "Pass Bar")
  
  # Report performance for impermissible proxy
  impermissible_labels_eval <- impermissible_proxy[eval_indices_rel]
  log_loss_impermissible <- mean(log_loss(calib_pred_impermissible_eval, impermissible_labels_eval))
  auc_impermissible <- compute_auc(calib_pred_impermissible_eval, impermissible_labels_eval)
  cat("  Impermissible proxy (", proxy_label, "):\n", sep = "")
  cat("    Log loss:", sprintf("%.4f", log_loss_impermissible), "\n")
  if (!is.na(auc_impermissible)) {
    cat("    AUC:", sprintf("%.4f", auc_impermissible), "\n")
  } else {
    cat("    AUC: N/A\n")
  }
  
  # Report performance for each permissible proxy
  for (j in 1:ncol(permissible_matrix)) {
    permissible_labels_eval <- permissible_matrix[eval_indices_rel, j]
    log_loss_permissible <- mean(log_loss(calib_pred_permissible_eval_list[[j]], permissible_labels_eval))
    auc_permissible <- compute_auc(calib_pred_permissible_eval_list[[j]], permissible_labels_eval)
    cat("  Permissible proxy", j, "(", proxy_names[j], "):\n")
    cat("    Log loss:", sprintf("%.4f", log_loss_permissible), "\n")
    if (!is.na(auc_permissible)) {
      cat("    AUC:", sprintf("%.4f", auc_permissible), "\n")
    } else {
      cat("    AUC: N/A\n")
    }
  }
  cat("\n")
  
  result_multiple <- falsify_multiple_proxy(
    predictions = predictions_raw,
    impermissible_labels = impermissible_proxy,
    permissible_labels_matrix = permissible_matrix,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices = eval_indices_rel
  )
  
  cat("RESULTS:\n")
  cat("  Result:", result_multiple$result, "\n")
  cat("  Method used:", result_multiple$method_used, "\n")
  cat("  Conditional rank test p-value:", format_pvalue(result_multiple$conditional_rank_pvalue, z = result_multiple$z_statistic), "\n")
  cat("  Test statistic (mean of impermissible ranks):", sprintf("%.4f", result_multiple$test_statistic), "\n")
  cat("  Expected value under H0:", sprintf("%.4f", result_multiple$expected_value), "\n")
  if (!is.na(result_multiple$z_statistic)) {
    cat("  Z-statistic:", sprintf("%.4f", result_multiple$z_statistic), "\n")
  }
  if (!is.null(result_multiple$n_permutations)) {
    cat("  Number of permutations:", result_multiple$n_permutations, "\n")
  }
  
  if (!is.null(result_multiple$avg_ranks)) {
    cat("\n  Average ranks (higher = worse performance):\n")
    for (i in 1:length(result_multiple$avg_ranks)) {
      cat(sprintf("    %s: %.3f\n", names(result_multiple$avg_ranks)[i], 
                  result_multiple$avg_ranks[i]))
    }
  }
  cat("\n")
  
  # Interpretation
  if (result_multiple$result == "DISCRIMINANT") {
    cat("INTERPRETATION: The algorithm appears to predict all permissible proxies\n")
    cat("better than the impermissible proxy (", proxy_label, "), suggesting it may not be\n", sep = "")
    cat("discriminating based on", proxy_label, ".\n\n")
  } else {
    cat("INTERPRETATION: The test is inconclusive. Either the algorithm is\n")
    cat("indiscriminate, or there is insufficient power to detect differences.\n\n")
  }
  
  # ============================================================================
  # Experiment: Algorithm 2 with Single Permissible Proxy (gpa_binary only)
  # ============================================================================
  
  cat(paste0(rep("=", 60), collapse = ""), "\n")
  cat("EXPERIMENT: ALGORITHM 2 WITH SINGLE PERMISSIBLE PROXY (gpa_binary only)\n")
  cat(paste0(rep("=", 60), collapse = ""), "\n\n")
  
  permissible_matrix_single <- cbind(gpa = calib_eval_data$gpa_binary)
  
  cat("Using 1 permissible proxy:\n")
  cat("  1. GPA (binary, above/below median)\n\n")
  
  result_multiple_single <- falsify_multiple_proxy(
    predictions = predictions_raw,
    impermissible_labels = impermissible_proxy,
    permissible_labels_matrix = permissible_matrix_single,
    alpha = 0.05,
    calib_indices = calib_indices_rel,
    eval_indices = eval_indices_rel
  )
  
  cat("RESULTS:\n")
  cat("  Result:", result_multiple_single$result, "\n")
  cat("  Method used:", result_multiple_single$method_used, "\n")
  cat("  Conditional rank test p-value:", format_pvalue(result_multiple_single$conditional_rank_pvalue, z = result_multiple_single$z_statistic), "\n")
  cat("  Test statistic (mean of impermissible ranks):", sprintf("%.4f", result_multiple_single$test_statistic), "\n")
  cat("  Expected value under H0:", sprintf("%.4f", result_multiple_single$expected_value), "\n")
  if (!is.na(result_multiple_single$z_statistic)) {
    cat("  Z-statistic:", sprintf("%.4f", result_multiple_single$z_statistic), "\n")
  }
  if (!is.null(result_multiple_single$n_permutations)) {
    cat("  Number of permutations:", result_multiple_single$n_permutations, "\n")
  }
  
  if (!is.null(result_multiple_single$avg_ranks)) {
    cat("\n  Average ranks (higher = worse performance):\n")
    for (i in 1:length(result_multiple_single$avg_ranks)) {
      cat(sprintf("    %s: %.3f\n", names(result_multiple_single$avg_ranks)[i], 
                  result_multiple_single$avg_ranks[i]))
    }
  }
  cat("\n")
  
  # Interpretation
  if (result_multiple_single$result == "DISCRIMINANT") {
    cat("INTERPRETATION: The algorithm appears to predict the permissible proxy\n")
    cat("(GPA, binary) better than the impermissible proxy (", proxy_label, "), suggesting it may\n", sep = "")
    cat("not be discriminating based on", proxy_label, ".\n\n")
  } else {
    cat("INTERPRETATION: The test is inconclusive. Either the algorithm is\n")
    cat("indiscriminate, or there is insufficient power to detect differences.\n\n")
  }
  
  # Comparison with Algorithm 1
  cat("COMPARISON WITH ALGORITHM 1:\n")
  cat("  Algorithm 1 result:", result_single$result, "\n")
  cat("  Algorithm 1 p-value:", format_pvalue(result_single$p_value, z = result_single$z_statistic), "\n")
  cat("  Algorithm 2 (M=1) result:", result_multiple_single$result, "\n")
  cat("  Algorithm 2 (M=1) p-value:", format_pvalue(result_multiple_single$conditional_rank_pvalue, z = result_multiple_single$z_statistic), "\n\n")
  
  # ============================================================================
  # Generate Falsification Test Results Table for Paper
  # ============================================================================
  
  # Create table with p-values for single and multiple proxy tests
  falsification_table <- data.frame(
    Test = c("Single Proxy", "Multiple Proxy"),
    P_Value = c(result_single$p_value, result_multiple$conditional_rank_pvalue),
    stringsAsFactors = FALSE
  )
  
  cat("FALSIFICATION TEST RESULTS TABLE:\n")
  print(falsification_table)
  cat("\n")
  
  # Save to CSV
  falsification_filename <- paste0("falsification_results_", proxy_name, ".csv")
  write.csv(falsification_table, file = falsification_filename, row.names = FALSE)
  cat("Falsification results table saved to:", falsification_filename, "\n\n")
  
  # Generate LaTeX table
  cat("LaTeX table for falsification results:\n")
  cat("\\begin{table}[h]\n")
  cat("\\centering\n")
  cat("\\caption{Falsification Test Results - ", proxy_label, " (Modeled Outcome: First-Year GPA)}\n", sep = "")
  cat("\\begin{tabular}{lc}\n")
  cat("\\toprule\n")
  cat("\\textbf{Test} & \\textbf{P-Value} \\\\\n")
  cat("\\midrule\n")
  cat("Single Proxy & ", format_pvalue(result_single$p_value, z = result_single$z_statistic), " \\\\\n")
  cat("Multiple Proxy & ", format_pvalue(result_multiple$conditional_rank_pvalue, z = result_multiple$z_statistic), " \\\\\n")
  cat("\\bottomrule\n")
  cat("\\end{tabular}\n")
  cat("\\end{table}\n\n")
  
  # ============================================================================
  # Generate Performance Table for Paper
  # ============================================================================
  
  cat(paste0(rep("=", 60), collapse = ""), "\n")
  cat("PERFORMANCE TABLE FOR PAPER\n")
  cat(paste0(rep("=", 60), collapse = ""), "\n\n")
  
  # Uncalibrated metrics table
  uncal_table <- data.frame()
  
  # Modeled outcome (top row)
  uncal_table <- rbind(uncal_table, compute_all_metrics(
    predictions_eval, y_true_eval, "1styearGPA"
  ))
  
  # Permissible proxies (middle rows)
  for (j in 1:ncol(permissible_matrix)) {
    permissible_labels_eval <- permissible_matrix[eval_indices_rel, j]
    uncal_table <- rbind(uncal_table, compute_all_metrics(
      predictions_eval, permissible_labels_eval,
      proxy_names[j]
    ))
  }
  
  # Impermissible proxy (bottom row)
  uncal_table <- rbind(uncal_table, compute_all_metrics(
    predictions_eval, impermissible_labels_eval, 
    paste0("Impermissible (", proxy_label, ")")
  ))
  
  # Calibrated metrics table
  calib_table <- data.frame()
  
  # Modeled outcome (calibrated, top row) - first-year GPA
  platt_modeled <- platt_scaling(predictions_raw, calib_eval_data$fygpa_binary, calib_indices_rel)
  calib_pred_modeled_eval <- platt_modeled$calibrated[eval_indices_rel]
  calib_table <- rbind(calib_table, compute_all_metrics(
    calib_pred_modeled_eval, y_true_eval, "1styearGPA"
  ))
  
  # Permissible proxies (calibrated, middle rows)
  for (j in 1:ncol(permissible_matrix)) {
    permissible_labels_eval <- permissible_matrix[eval_indices_rel, j]
    calib_table <- rbind(calib_table, compute_all_metrics(
      calib_pred_permissible_eval_list[[j]], permissible_labels_eval,
      proxy_names[j]
    ))
  }
  
  # Impermissible proxy (calibrated, bottom row)
  calib_table <- rbind(calib_table, compute_all_metrics(
    calib_pred_impermissible_eval, impermissible_labels_eval,
    paste0("Impermissible (", proxy_label, ")")
  ))
  
  # Format tables with proper column names
  colnames(uncal_table) <- c("Proxy", "AUC", "AU PR", "MSE", "PPV Top 2%", "PPV Top 10%", "PPV Top 50%", "PPV Top 75%")
  colnames(calib_table) <- c("Proxy", "AUC", "AU PR", "MSE", "PPV Top 2%", "PPV Top 10%", "PPV Top 50%", "PPV Top 75%")
  
  # Format numbers for display
  format_table <- function(df) {
    df_formatted <- df
    df_formatted$AUC <- sprintf("%.4f", df_formatted$AUC)
    df_formatted$`AU PR` <- sprintf("%.4f", df_formatted$`AU PR`)
    df_formatted$MSE <- sprintf("%.4f", df_formatted$MSE)
    df_formatted$`PPV Top 2%` <- sprintf("%.4f", df_formatted$`PPV Top 2%`)
    df_formatted$`PPV Top 10%` <- sprintf("%.4f", df_formatted$`PPV Top 10%`)
    df_formatted$`PPV Top 50%` <- sprintf("%.4f", df_formatted$`PPV Top 50%`)
    df_formatted$`PPV Top 75%` <- sprintf("%.4f", df_formatted$`PPV Top 75%`)
    return(df_formatted)
  }
  
  cat("UNCALIBRATED METRICS:\n")
  print(format_table(uncal_table))
  cat("\n")
  
  cat("CALIBRATED METRICS:\n")
  print(format_table(calib_table))
  cat("\n")
  
  # Create combined table with LSAC header for paper
  # Add dataset column
  uncal_table_paper <- cbind(Dataset = "LSAC", uncal_table)
  calib_table_paper <- cbind(Dataset = "LSAC", calib_table)
  
  # Save to CSV files
  uncal_filename <- paste0("results_uncalibrated_", proxy_name, ".csv")
  calib_filename <- paste0("results_calibrated_", proxy_name, ".csv")
  
  write.csv(uncal_table_paper, file = uncal_filename, row.names = FALSE)
  write.csv(calib_table_paper, file = calib_filename, row.names = FALSE)
  
  cat("Tables saved to:\n")
  cat("  ", uncal_filename, "\n")
  cat("  ", calib_filename, "\n\n")
  
  # Also create a LaTeX-formatted table
  create_latex_table <- function(df, title) {
    cat("\nLaTeX table for", title, ":\n")
    cat("\\begin{table}[h]\n")
    cat("\\centering\n")
    cat("\\caption{", title, " (Modeled Outcome: First-Year GPA)}\n", sep = "")
    cat("\\begin{tabular}{l", paste(rep("c", ncol(df) - 1), collapse = ""), "}\n", sep = "")
    cat("\\toprule\n")
    cat(paste(colnames(df), collapse = " & "), "\\\\\n")
    cat("\\midrule\n")
    for (i in 1:nrow(df)) {
      row_vals <- sapply(df[i, ], function(x) {
        if (is.numeric(x)) {
          if (is.na(x)) return("N/A")
          return(sprintf("%.4f", x))
        }
        return(as.character(x))
      })
      cat(paste(row_vals, collapse = " & "), "\\\\\n")
    }
    cat("\\bottomrule\n")
    cat("\\end{tabular}\n")
    cat("\\end{table}\n\n")
  }
  
  create_latex_table(uncal_table_paper, paste("Uncalibrated Metrics -", proxy_label))
  create_latex_table(calib_table_paper, paste("Calibrated Metrics -", proxy_label))
  
  return(list(
    proxy_name = proxy_name,
    proxy_label = proxy_label,
    result_single = result_single,
    result_multiple = result_multiple,
    result_multiple_single = result_multiple_single
  ))
}

# ============================================================================
# Run Analysis for Each Impermissible Proxy
# ============================================================================

# Create indices relative to calib_eval_data (calibration is first, evaluation is second)
n_calib_eval <- length(predictions_raw)
calib_indices_rel <- 1:length(calib_indices)
eval_indices_rel <- (length(calib_indices) + 1):n_calib_eval

# Define proxy configurations based on what's requested
proxy_configs <- list()
if ("race" %in% impermissible_proxies_to_run) {
  proxy_configs[[length(proxy_configs) + 1]] <- list(name = "race", label = "Race")
}
if ("family_income" %in% impermissible_proxies_to_run) {
  proxy_configs[[length(proxy_configs) + 1]] <- list(name = "family_income", label = "Family Income")
}
if ("gender" %in% impermissible_proxies_to_run) {
  proxy_configs[[length(proxy_configs) + 1]] <- list(name = "gender", label = "Gender")
}

# Store results for all proxies
all_results <- list()

# Run analysis for each impermissible proxy
for (proxy_config in proxy_configs) {
  result <- run_analysis_for_proxy(
    proxy_name = proxy_config$name,
    proxy_label = proxy_config$label,
    calib_eval_data = calib_eval_data,
    predictions_raw = predictions_raw,
    calib_indices_rel = calib_indices_rel,
    eval_indices_rel = eval_indices_rel
  )
  
  if (!is.null(result)) {
    all_results[[proxy_config$name]] <- result
  }
}

# ============================================================================
# Additional Analysis: Sensitivity Analysis (Optional)
# ============================================================================

# Note: Sensitivity analysis can be run separately for each proxy if needed
# For now, we'll skip it to keep output manageable when running multiple proxies

cat("\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("ANALYSIS COMPLETE\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

cat("Summary of analyses run:\n")
for (proxy_name in names(all_results)) {
  result <- all_results[[proxy_name]]
  cat("  ", result$proxy_label, ":\n", sep = "")
  cat("    Algorithm 1 result:", result$result_single$result, "\n")
  cat("    Algorithm 2 result:", result$result_multiple$result, "\n")
  cat("    Algorithm 2 (M=1) result:", result$result_multiple_single$result, "\n")
}
cat("\n")

