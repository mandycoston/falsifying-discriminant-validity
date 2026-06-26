source("falsification_methods.R")
source("load_lsac_data.R")

set.seed(42)

lsac_data <- load_lsac_data()
n <- nrow(lsac_data)
train_size <- floor(n * 0.5)
calib_size <- floor(n * 0.25)

all_indices <- sample(n)
train_indices <- all_indices[1:train_size]
calib_indices <- all_indices[(train_size + 1):(train_size + calib_size)]
eval_indices <- all_indices[(train_size + calib_size + 1):n]

train_data <- lsac_data[train_indices, ]
calib_eval_data <- lsac_data[c(calib_indices, eval_indices), ]

model <- glm(fygpa_binary ~ lsat + ugpa, data = train_data, family = binomial,
             control = list(maxit = 1000))
predictions_raw <- predict(model, newdata = calib_eval_data, type = "response")

calib_indices_rel <- 1:length(calib_indices)
eval_indices_rel <- (length(calib_indices) + 1):nrow(calib_eval_data)

permissible_matrix <- cbind(
  fygpa = calib_eval_data$fygpa_binary,
  gpa   = calib_eval_data$gpa_binary,
  pass_bar = calib_eval_data$pass_bar
)

get_rank_distribution <- function(predictions_raw, impermissible_proxy,
                                   permissible_matrix, calib_indices_rel, eval_indices_rel) {
  M <- ncol(permissible_matrix)
  platt_imp <- platt_scaling(predictions_raw, impermissible_proxy, calib_indices_rel)
  calib_pred_imp <- platt_imp$calibrated[eval_indices_rel]

  calib_pred_perm <- matrix(NA, nrow = length(eval_indices_rel), ncol = M)
  for (j in 1:M) {
    ps <- platt_scaling(predictions_raw, permissible_matrix[, j], calib_indices_rel)
    calib_pred_perm[, j] <- ps$calibrated[eval_indices_rel]
  }

  loss_imp <- compute_loss(calib_pred_imp, impermissible_proxy[eval_indices_rel])
  loss_perm <- matrix(NA, nrow = length(eval_indices_rel), ncol = M)
  for (j in 1:M) {
    loss_perm[, j] <- compute_loss(calib_pred_perm[, j],
                                    permissible_matrix[eval_indices_rel, j])
  }

  loss_matrix <- cbind(loss_imp, loss_perm)
  ranks_matrix <- t(apply(loss_matrix, 1, function(x) rank(x, ties.method = "average")))
  imp_ranks <- ranks_matrix[, 1]
  return(imp_ranks)
}

race_vals <- tolower(as.character(calib_eval_data$race))
impermissible_race <- as.numeric(race_vals == "white" | race_vals == "1" |
                                  grepl("white", race_vals, ignore.case = TRUE))
impermissible_gender <- as.numeric(calib_eval_data$male)

ranks_gender <- get_rank_distribution(predictions_raw, impermissible_gender,
                                       permissible_matrix, calib_indices_rel, eval_indices_rel)
ranks_race   <- get_rank_distribution(predictions_raw, impermissible_race,
                                       permissible_matrix, calib_indices_rel, eval_indices_rel)

M <- 3
max_rank <- M + 1

png("../figures/rank_distribution_gender_race.png", width = 1200, height = 500, res = 150)
par(mfrow = c(1, 2), mar = c(5, 5, 3, 1), cex.axis = 1.3, cex.lab = 1.4, cex.main = 1.5)

gender_tab <- table(factor(ranks_gender, levels = 1:max_rank)) / length(ranks_gender)
bp1 <- barplot(gender_tab, col = "steelblue",
        main = "Gender (Impermissible)",
        xlab = "Rank of Gender Loss", ylab = "Proportion of Observations",
        ylim = c(0, max(gender_tab, 0.25) * 1.25),
        names.arg = 1:max_rank)
abline(h = 1/max_rank, lty = 2, lwd = 2)
text(bp1, gender_tab + 0.04, labels = sprintf("%.1f%%", gender_tab * 100), cex = 1.1)

race_tab <- table(factor(ranks_race, levels = 1:max_rank)) / length(ranks_race)
bp2 <- barplot(race_tab, col = "indianred",
        main = "Race (Impermissible)",
        xlab = "Rank of Race Loss", ylab = "Proportion of Observations",
        ylim = c(0, max(race_tab) + 0.12),
        names.arg = 1:max_rank)
abline(h = 1/max_rank, lty = 2, lwd = 2)
text(bp2, race_tab + 0.05, labels = sprintf("%.1f%%", race_tab * 100), cex = 1.1)

dev.off()
cat("Figure saved to ../figures/rank_distribution_gender_race.png\n")
