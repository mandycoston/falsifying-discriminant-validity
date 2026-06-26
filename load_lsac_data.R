# ============================================================================
# Load and Prepare LSAC Dataset
# ============================================================================

#' Load and prepare LSAC dataset
#' 
#' This function loads the Law School Admission Council (LSAC) dataset
#' from GitHub. If the dataset cannot be loaded, the function will stop
#' with an error.
#' 
#' @return Data frame with LSAC variables including:
#'   - race: Race/ethnicity (White, Black, Hispanic, Asian, Other)
#'   - male: Male indicator (1 = male, 0 = female)
#'   - lsat: LSAT score (continuous)
#'   - gpa: Undergraduate GPA (continuous)
#'   - fygpa: First-year law school GPA (continuous)
#'   - pass_bar: Bar passage (binary, 1 = passed, 0 = failed)
#'   - fygpa_binary: Binary version of FYGPA (above/below median)
#'   - gpa_binary: Binary version of GPA (above/below median)
#' 
#' @examples
#' lsac_data <- load_lsac_data()
#' head(lsac_data)
#' summary(lsac_data)
load_lsac_data <- function() {
  lsac_url <- "https://raw.githubusercontent.com/damtharvey/law-school-dataset/master/law_dataset.csv"
  
  # Load from GitHub - stop with error if it fails
  lsac_data <- tryCatch({
    read.csv(lsac_url, stringsAsFactors = FALSE)
  }, error = function(e) {
    stop("Failed to load LSAC data from GitHub. Error: ", e$message, 
         "\nPlease check your internet connection and the URL: ", lsac_url)
  })
  
  # Rename specific columns from the GitHub dataset
  if ("zfygpa" %in% colnames(lsac_data)) {
    lsac_data$fygpa <- lsac_data$zfygpa
    lsac_data$zfygpa <- NULL  # Remove the old column
  }
  
  if ("zgpa" %in% colnames(lsac_data)) {
    lsac_data$gpa <- lsac_data$zgpa
    lsac_data$zgpa <- NULL  # Remove the old column
  }
  
  # Standardize column names (handle different naming conventions)
  # Common variations in LSAC datasets
  col_names <- tolower(colnames(lsac_data))
  
  # Map common column name variations to standard names
  if ("race" %in% col_names) {
    # Already correct
  } else if ("race1" %in% col_names) {
    lsac_data$race <- lsac_data[[which(col_names == "race1")]]
  } else if (any(grepl("race", col_names))) {
    race_col <- which(grepl("race", col_names))[1]
    lsac_data$race <- lsac_data[[race_col]]
  }
  
  # Handle gender/male column
  if ("male" %in% col_names) {
    # Already correct - male column exists (1 = male, 0 = female)
    lsac_data$male <- as.numeric(lsac_data[[which(col_names == "male")]])
  } else if ("gender" %in% col_names) {
    # Convert gender to male (1 = male, 0 = female)
    gender_vals <- tolower(lsac_data[[which(col_names == "gender")]])
    lsac_data$male <- as.numeric(gender_vals == "male" | gender_vals == "m")
  } else if ("sex" %in% col_names) {
    # Convert sex to male (1 = male, 0 = female)
    sex_vals <- tolower(lsac_data[[which(col_names == "sex")]])
    lsac_data$male <- as.numeric(sex_vals == "male" | sex_vals == "m")
  } else if (any(grepl("gender", col_names))) {
    gender_col <- which(grepl("gender", col_names))[1]
    gender_vals <- tolower(lsac_data[[gender_col]])
    lsac_data$male <- as.numeric(gender_vals == "male" | gender_vals == "m")
  } else if (any(grepl("male", col_names))) {
    male_col <- which(grepl("male", col_names))[1]
    lsac_data$male <- as.numeric(lsac_data[[male_col]])
  }
  
  if ("lsat" %in% col_names) {
    # Already correct
  } else if ("lsat_score" %in% col_names) {
    lsac_data$lsat <- lsac_data[[which(col_names == "lsat_score")]]
  } else if (any(grepl("lsat", col_names))) {
    lsat_col <- which(grepl("lsat", col_names))[1]
    lsac_data$lsat <- lsac_data[[lsat_col]]
  }
  
  if ("gpa" %in% col_names) {
    # Already correct
  } else if ("ugpa" %in% col_names) {
    lsac_data$gpa <- lsac_data[[which(col_names == "ugpa")]]
  } else if (any(grepl("gpa", col_names))) {
    gpa_col <- which(grepl("gpa", col_names))[1]
    lsac_data$gpa <- lsac_data[[gpa_col]]
  }
  
  if ("fygpa" %in% col_names) {
    # Already correct
  } else if (any(grepl("fygpa", col_names))) {
    fygpa_col <- which(grepl("fygpa", col_names))[1]
    lsac_data$fygpa <- lsac_data[[fygpa_col]]
  } else if (any(grepl("first.*year.*gpa", col_names))) {
    fygpa_col <- which(grepl("first.*year.*gpa", col_names))[1]
    lsac_data$fygpa <- lsac_data[[fygpa_col]]
  }
  
  # Handle bar passage - look for pass_bar variable (already binary)
  if ("pass_bar" %in% col_names) {
    # pass_bar is already binary, just ensure it's numeric
    lsac_data$pass_bar <- as.numeric(lsac_data[[which(col_names == "pass_bar")]])
  } else if (any(grepl("pass.*bar", col_names))) {
    # Try to find pass_bar with different case or naming
    pass_bar_col <- which(grepl("pass.*bar", col_names))[1]
    lsac_data$pass_bar <- as.numeric(lsac_data[[pass_bar_col]])
  }
  
  # Create binary versions of all proxy outcomes (above/below median)
  # fygpa_binary
  if (!"fygpa_binary" %in% colnames(lsac_data) && "fygpa" %in% colnames(lsac_data)) {
    lsac_data$fygpa_binary <- as.numeric(lsac_data$fygpa > median(lsac_data$fygpa, na.rm = TRUE))
  } else if ("fygpa_binary" %in% colnames(lsac_data)) {
    # Ensure it's numeric if it already exists
    lsac_data$fygpa_binary <- as.numeric(lsac_data$fygpa_binary)
  }
  
  # gpa_binary
  if (!"gpa_binary" %in% colnames(lsac_data) && "gpa" %in% colnames(lsac_data)) {
    lsac_data$gpa_binary <- as.numeric(lsac_data$gpa > median(lsac_data$gpa, na.rm = TRUE))
  } else if ("gpa_binary" %in% colnames(lsac_data)) {
    # Ensure it's numeric if it already exists
    lsac_data$gpa_binary <- as.numeric(lsac_data$gpa_binary)
  }
  
  
  # Display column names for debugging
  cat("Loaded LSAC dataset with", nrow(lsac_data), "rows and", ncol(lsac_data), "columns\n")
  cat("Available columns:", paste(colnames(lsac_data), collapse = ", "), "\n")
  
  return(lsac_data)
}

# ============================================================================
# Helper function to prepare LSAC data for analysis
# ============================================================================

#' Prepare LSAC data for falsification analysis
#' 
#' This function prepares the LSAC data by creating features and labels
#' for use in the falsification procedures.
#' 
#' @param lsac_data Data frame from load_lsac_data()
#' @param outcome_var Name of the outcome variable to predict (default: "fygpa_binary")
#' @param feature_vars Vector of feature variable names (default: c("lsat", "gpa"))
#' @return List containing:
#'   - X: Feature matrix
#'   - y: Outcome vector
#'   - data: Original data frame
#' 
#' @examples
#' lsac_data <- load_lsac_data()
#' prepared_data <- prepare_lsac_for_analysis(lsac_data)
#' X <- prepared_data$X
#' y <- prepared_data$y
prepare_lsac_for_analysis <- function(lsac_data, 
                                      outcome_var = "fygpa_binary",
                                      feature_vars = c("lsat", "gpa")) {
  # Extract features
  X <- as.matrix(lsac_data[, feature_vars, drop = FALSE])
  
  # Extract outcome
  y <- lsac_data[[outcome_var]]
  
  return(list(
    X = X,
    y = y,
    data = lsac_data
  ))
}

# ============================================================================
# Helper function to inspect LSAC data structure
# ============================================================================

#' Inspect LSAC dataset structure
#' 
#' This function helps debug data loading issues by showing what columns
#' are available and their types.
#' 
#' @param lsac_data Data frame from load_lsac_data()
#' @return Prints summary information about the dataset
inspect_lsac_data <- function(lsac_data) {
  cat("Dataset structure:\n")
  cat("  Rows:", nrow(lsac_data), "\n")
  cat("  Columns:", ncol(lsac_data), "\n\n")
  cat("Column names:\n")
  print(colnames(lsac_data))
  cat("\nColumn types:\n")
  print(sapply(lsac_data, class))
  cat("\nFirst few rows:\n")
  print(head(lsac_data))
  cat("\nSummary statistics:\n")
  print(summary(lsac_data))
}

