# Implementation of Falsification Methods for Predictive Algorithms
# Tested on LSAC (Law School Admission Council) Dataset

library(tidyverse)
library(glmnet)
library(caret)
library(boot) # For bootstrap if needed

# ============================================================================
# Helper Functions
# ============================================================================

#' Format p-value for display
#'
#' Shows p-values in scientific notation when very small (< 1e-4),
#' otherwise uses fixed decimal format. For extremely small p-values
#' (beyond floating point precision), computes approximate value from
#' z-statistic if provided.
#'
#' @param p P-value to format
#' @param z Optional z-statistic (used to compute more precise p-value when p is tiny)
#' @param digits Number of significant digits for scientific notation (default 2)
#' @return Formatted string
format_pvalue <- function(p, z = NULL, digits = 2) {
  if (is.na(p)) {
    return("NA")
  } else if (p < 1e-300 && !is.null(z) && !is.na(z)) {
    # For extremely small p-values, compute from z using log scale
    # log10(p) = log10(1 - pnorm(z)) ≈ pnorm(z, lower.tail=FALSE, log.p=TRUE) / log(10)
    log10_p <- pnorm(z, lower.tail = FALSE, log.p = TRUE) / log(10)
    # Format as 10^x
    return(sprintf("10^(%.1f)", log10_p))
  } else if (p < 1e-300) {
    return("< 1e-300")
  } else if (p < 1e-4) {
    # Use scientific notation for very small p-values
    return(sprintf("%.*e", digits, p))
  } else {
    return(sprintf("%.6f", p))
  }
}

#' Apply Platt scaling to calibrate predictions for binary outcomes
#'
#' @param predictions Vector of raw predictions
#' @param labels Vector of binary labels (0/1)
#' @param calibration_indices Indices for calibration set
#' @return Calibrated predictions
platt_scaling <- function(predictions, labels, calibration_indices) {
  # Use calibration set
  calib_pred <- predictions[calibration_indices]
  calib_labels <- labels[calibration_indices]

  # Fit Platt scaling (logistic regression):
  # P(y=1 | f(x)) = 1 / (1 + exp(-(A*f(x) + B)))
  #
  # Note: glm() may warn under (near-)separation; we still prefer the fitted
  # calibration because returning the identity mapping can inflate losses for
  # well-separated proxies and make the falsification tests look (incorrectly)
  # INDISCRIMINANT.
  model <- glm(
    calib_labels ~ calib_pred,
    family = binomial(link = "logit"),
    control = list(maxit = 1000)
  )

  # Extract coefficients
  A <- coef(model)[2]
  B <- coef(model)[1]

  # Apply calibration to all predictions
  calibrated <- 1 / (1 + exp(-(A * predictions + B)))

  # Clip predictions to avoid log(0) downstream
  calibrated <- pmax(pmin(calibrated, 1 - 1e-15), 1e-15)

  return(list(calibrated = calibrated, A = A, B = B, method = "platt"))
}

#' Compute log-loss (binary cross-entropy loss)
#'
#' @param predictions Vector of predicted probabilities
#' @param labels Vector of true labels (0/1)
#' @return Vector of losses
log_loss <- function(predictions, labels) {
  # Clip predictions to avoid log(0)
  predictions <- pmax(pmin(predictions, 1 - 1e-15), 1e-15)
  # Log-loss: -y*log(p) - (1-y)*log(1-p)
  loss <- -labels * log(predictions) - (1 - labels) * log(1 - predictions)
  return(loss)
}

#' Compute log-loss for binary outcomes
#'
#' @param predictions Vector of predicted probabilities
#' @param labels Vector of true binary labels (0/1)
#' @return Vector of losses
compute_loss <- function(predictions, labels) {
  # Use log-loss for binary outcomes
  return(log_loss(predictions, labels))
}

# ============================================================================
# Algorithm 1: Falsification with Single Permissible Proxy
# ============================================================================

#' Falsification procedure with single permissible proxy
#'
#' @param predictions Vector of raw algorithm predictions
#' @param impermissible_labels Vector of impermissible proxy labels (tilde Y)
#' @param permissible_labels Vector of permissible proxy labels (Y_1)
#' @param alpha Significance level (default 0.05)
#' @param calibration_prop Proportion of data to use for calibration (default 0.2, ignored if calib_indices provided)
#' @param min_eval_size Minimum size for evaluation set (default 3)
#' @param calib_indices Optional: indices for calibration set (if NULL, will split internally)
#' @param eval_indices Optional: indices for evaluation set (if NULL, will split internally)
#' @return List with result ("INDISCRIMINANT" or "DISCRIMINANT"), p-value, test used
falsify_single_proxy <- function(predictions, impermissible_labels, permissible_labels,
                                 alpha = 0.05, calibration_prop = 0.2, min_eval_size = 3,
                                 calib_indices = NULL, eval_indices = NULL) {
  n <- length(predictions)

  # Use provided indices or split internally
  if (is.null(calib_indices) || is.null(eval_indices)) {
    # Ensure we have enough data for evaluation
    # Adjust calibration_prop if needed
    max_calib_prop <- (n - min_eval_size) / n
    calibration_prop <- min(calibration_prop, max_calib_prop)

    # Split data into calibration and evaluation sets
    calib_size <- floor(n * calibration_prop)
    # Ensure evaluation set has at least min_eval_size samples
    if (n - calib_size < min_eval_size) {
      calib_size <- n - min_eval_size
    }
    calib_indices <- sample(n, calib_size)
    eval_indices <- setdiff(1:n, calib_indices)
  }



  # Apply Platt scaling separately for each outcome
  platt_impermissible <- platt_scaling(predictions, impermissible_labels, calib_indices)
  platt_permissible <- platt_scaling(predictions, permissible_labels, calib_indices)

  # Get calibrated predictions for evaluation set
  calib_pred_impermissible <- platt_impermissible$calibrated[eval_indices]
  calib_pred_permissible <- platt_permissible$calibrated[eval_indices]

  # Compute losses (binary cross-entropy for binary outcomes)
  loss_impermissible <- compute_loss(calib_pred_impermissible, impermissible_labels[eval_indices])
  loss_permissible <- compute_loss(calib_pred_permissible, permissible_labels[eval_indices])

  # Compute loss differences: Delta = loss_impermissible - loss_permissible
  delta <- loss_impermissible - loss_permissible

  # Check evaluation set size
  n_eval <- length(delta)
  if (n_eval < 50) {
    warning("Evaluation set has only ", n_eval, " samples. Results may be unreliable.")
  }

  # Check normality (Shapiro-Wilk test) - only if sample size is appropriate
  # Shapiro-Wilk requires sample size between 3 and 5000
  is_normal <- FALSE

  if (n_eval >= 3 && n_eval <= 5000) {
    shapiro_test <- shapiro.test(delta)
    is_normal <- shapiro_test$p.value > 0.05
  } else if (n_eval > 5000) {
    # For large samples, use a subset for normality test
    shapiro_test <- shapiro.test(sample(delta, 5000))
    is_normal <- shapiro_test$p.value > 0.05
  } else {
    # For very small samples (< 3), assume non-normal and use Wilcoxon
    is_normal <- FALSE
  }

  # Check for outliers (using IQR method)
  Q1 <- quantile(delta, 0.25, na.rm = TRUE)
  Q3 <- quantile(delta, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  outliers <- sum(delta < (Q1 - 1.5 * IQR) | delta > (Q3 + 1.5 * IQR), na.rm = TRUE)
  has_outliers <- outliers > 0

  # Perform appropriate test
  if (is_normal && !has_outliers) {
    # Paired t-test: H0: E[Delta] <= 0 vs H1: E[Delta] > 0
    test_result <- t.test(delta, alternative = "greater", mu = 0)
    test_name <- "paired t-test"
    p_value <- test_result$p.value
    # t-statistic can be used as z-equivalent for large samples
    z_equivalent <- test_result$statistic
  } else {
    # Wilcoxon signed-rank test: H0: median(Delta) <= 0 vs H1: median(Delta) > 0
    test_result <- wilcox.test(delta, alternative = "greater", mu = 0, paired = FALSE)
    test_name <- "Wilcoxon signed-rank test"
    p_value <- test_result$p.value
    # Compute z-equivalent from Wilcoxon statistic using normal approximation
    # W ~ Normal(mu, sigma) where mu = n(n+1)/4, sigma = sqrt(n(n+1)(2n+1)/24)
    W <- test_result$statistic
    n <- n_eval
    mu_W <- n * (n + 1) / 4
    sigma_W <- sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z_equivalent <- (W - mu_W) / sigma_W
  }

  # Make decision
  if (p_value <= alpha) {
    result <- "DISCRIMINANT"
  } else {
    result <- "INDISCRIMINANT (inconclusive)"
  }

  return(list(
    result = result,
    p_value = p_value,
    z_statistic = as.numeric(z_equivalent),
    test_used = test_name,
    delta_mean = mean(delta),
    delta_median = median(delta),
    is_normal = is_normal,
    has_outliers = has_outliers
  ))
}

# ============================================================================
# Algorithm 2: Falsification with Multiple Permissible Proxies
# ============================================================================

#' Falsification procedure with multiple permissible proxies
#'
#' @param predictions Vector of raw algorithm predictions
#' @param impermissible_labels Vector of impermissible proxy labels (tilde Y)
#' @param permissible_labels_matrix Matrix where each column is a permissible proxy (Y_1, ..., Y_M)
#' @param alpha Significance level (default 0.05)
#' @param calibration_prop Proportion of data to use for calibration (default 0.2, ignored if calib_indices provided)
#' @param min_eval_size Minimum size for evaluation set (default 3)
#' @param calib_indices Optional: indices for calibration set (if NULL, will split internally)
#' @param eval_indices Optional: indices for evaluation set (if NULL, will split internally)
#' @param method Method to use: "auto" (default, chooses based on sample size and ties), "normal" (normal approximation), or "permutation" (permutation test)
#' @param n_permutations Number of permutations for permutation test (default 10000)
#' @return List with result, p-values, test statistics
falsify_multiple_proxy <- function(predictions, impermissible_labels, permissible_labels_matrix,
                                   alpha = 0.05, calibration_prop = 0.2, min_eval_size = 3,
                                   calib_indices = NULL, eval_indices = NULL,
                                   method = "auto", n_permutations = 10000) {
  n <- length(predictions)
  M <- ncol(permissible_labels_matrix)

  # Use provided indices or split internally
  if (is.null(calib_indices) || is.null(eval_indices)) {
    # Ensure we have enough data for evaluation
    # Adjust calibration_prop if needed
    max_calib_prop <- (n - min_eval_size) / n
    calibration_prop <- min(calibration_prop, max_calib_prop)

    # Split data into calibration and evaluation sets
    calib_size <- floor(n * calibration_prop)
    # Ensure evaluation set has at least min_eval_size samples
    if (n - calib_size < min_eval_size) {
      calib_size <- n - min_eval_size
    }
    calib_indices <- sample(n, calib_size)
    eval_indices <- setdiff(1:n, calib_indices)
  }


  # Apply Platt scaling for impermissible proxy
  platt_impermissible <- platt_scaling(predictions, impermissible_labels, calib_indices)
  calib_pred_impermissible <- platt_impermissible$calibrated[eval_indices]

  # Apply Platt scaling for each permissible proxy
  platt_permissible_list <- list()
  calib_pred_permissible_matrix <- matrix(NA, nrow = length(eval_indices), ncol = M)

  for (j in 1:M) {
    platt_permissible_list[[j]] <- platt_scaling(predictions, permissible_labels_matrix[, j], calib_indices)
    calib_pred_permissible_matrix[, j] <- platt_permissible_list[[j]]$calibrated[eval_indices]
  }

  # Compute losses for evaluation set (automatically handles binary)
  loss_impermissible <- compute_loss(calib_pred_impermissible, impermissible_labels[eval_indices])
  loss_permissible_matrix <- matrix(NA, nrow = length(eval_indices), ncol = M)

  for (j in 1:M) {
    loss_permissible_matrix[, j] <- compute_loss(
      calib_pred_permissible_matrix[, j],
      permissible_labels_matrix[eval_indices, j]
    )
  }

  # Prepare data: each row is a sample, each column is an outcome
  loss_matrix <- cbind(loss_impermissible, loss_permissible_matrix)
  colnames(loss_matrix) <- c("impermissible", paste0("permissible_", 1:M))
  n_eval <- nrow(loss_matrix)

  # Conditional rank test: compute row-wise ranks of impermissible loss
  # For each sample, rank the impermissible loss among all M+1 losses
  # Use method="average" to handle ties consistently
  ranks_matrix <- t(apply(loss_matrix, 1, function(x) rank(x, ties.method = "average")))
  colnames(ranks_matrix) <- colnames(loss_matrix)

  # Extract ranks of impermissible proxy for each sample
  impermissible_ranks <- ranks_matrix[, "impermissible"]

  # Test statistic: mean of row-wise ranks of impermissible proxy
  test_statistic <- mean(impermissible_ranks)

  # Under H0: ranks are uniform over {1, 2, ..., M+1}
  # Expected value of a single rank = (1 + 2 + ... + (M+1)) / (M+1) = (M+2)/2
  # Expected value of mean rank = (M+2)/2
  expected_value <- (M + 2) / 2

  # Check for rank ties: check if any row has duplicate loss values
  has_ties <- any(apply(loss_matrix, 1, function(x) length(unique(x)) < length(x)))

  # Determine which method to use
  if (method == "auto") {
    # Use normal approximation if sample size > 30 AND no ties
    # Otherwise use permutation test
    if (n_eval > 30 && !has_ties) {
      method_used <- "normal"
    } else {
      method_used <- "permutation"
      if (n_eval <= 30) {
        cat("Note: Using permutation test because sample size (", n_eval, ") <= 30\n", sep = "")
      }
      if (has_ties) {
        cat("Note: Using permutation test because rank ties were detected\n", sep = "")
      }
    }
  } else {
    method_used <- method
  }

  # Perform the test
  if (method_used == "normal") {
    # Normal approximation method
    # Variance for normal approximation
    # For uniform distribution on {1, 2, ..., M+1}:
    # Var(R) = ((M+1)^2 - 1) / 12 = (M^2 + 2M) / 12
    # For mean of n independent ranks: Var(mean) = Var(R) / n
    variance_per_rank <- ((M + 1)^2 - 1) / 12
    variance_of_mean <- variance_per_rank / n_eval
    std_error <- sqrt(variance_of_mean)

    # Normal approximation: Z = (T - E[T]) / SE
    z_statistic <- (test_statistic - expected_value) / std_error

    # One-sided test: H1: impermissible has higher rank (worse performance)
    # H0: E[mean rank] = (M+2)/2 vs H1: E[mean rank] > (M+2)/2
    # We test if test_statistic > expected_value (upper tail)
    p_value <- 1 - pnorm(z_statistic)

    cat("Conditional rank test (normal approximation):\n")
    cat("  Test statistic (mean of impermissible ranks):", sprintf("%.6f", test_statistic), "\n")
    cat("  Expected value under H0:", sprintf("%.6f", expected_value), "\n")
    cat("  Z-statistic:", sprintf("%.6f", z_statistic), "\n")
    cat("  P-value (one-sided, upper):", format_pvalue(p_value, z = z_statistic), "\n\n")

    return(list(
      result = ifelse(p_value <= alpha, "DISCRIMINANT", "INDISCRIMINANT (inconclusive)"),
      p_value = p_value,
      conditional_rank_pvalue = p_value, # Backward compatibility
      test_statistic = test_statistic,
      expected_value = expected_value,
      z_statistic = z_statistic,
      method_used = "normal approximation",
      avg_ranks = colMeans(ranks_matrix),
      has_ties = has_ties,
      n_eval = n_eval
    ))
  } else {
    # Permutation test method
    cat("Conditional rank test (permutation test):\n")
    cat("  Test statistic (mean of impermissible ranks):", sprintf("%.6f", test_statistic), "\n")
    cat("  Expected value under H0:", sprintf("%.6f", expected_value), "\n")
    cat("  Running", n_permutations, "permutations...\n")

    # Store observed test statistic
    test_statistic_obs <- test_statistic

    # Permutation test: randomly permute losses within each sample
    permuted_statistics <- numeric(n_permutations)

    for (b in 1:n_permutations) {
      # For each sample, randomly permute the M+1 losses
      permuted_loss_matrix <- t(apply(loss_matrix, 1, function(x) sample(x)))

      # Recompute ranks for permuted losses
      permuted_ranks_matrix <- t(apply(permuted_loss_matrix, 1, function(x) rank(x, ties.method = "average")))

      # Extract rank of the loss that ended up in the impermissible position (first column)
      # After permutation, we look at what rank the loss in the first column position gets
      permuted_impermissible_ranks <- permuted_ranks_matrix[, 1]

      # Compute mean rank for this permutation
      permuted_statistics[b] <- mean(permuted_impermissible_ranks)
    }

    # Compute one-sided empirical p-value
    # H1: E[mean rank] > (M+2)/2, so we count how many permuted statistics >= observed
    n_ge <- sum(permuted_statistics >= test_statistic_obs)
    p_value <- (1 + n_ge) / (n_permutations + 1)

    cat("  P-value (one-sided, upper):", format_pvalue(p_value), "\n")
    cat("  (", n_ge, " out of ", n_permutations, " permutations had mean rank >= observed)\n\n", sep = "")

    return(list(
      result = ifelse(p_value <= alpha, "DISCRIMINANT", "INDISCRIMINANT (inconclusive)"),
      p_value = p_value,
      conditional_rank_pvalue = p_value, # Backward compatibility
      test_statistic = test_statistic_obs,
      expected_value = expected_value,
      z_statistic = NA, # Not applicable for permutation test
      method_used = "permutation test",
      n_permutations = n_permutations,
      avg_ranks = colMeans(ranks_matrix),
      has_ties = has_ties,
      n_eval = n_eval
    ))
  }
}
