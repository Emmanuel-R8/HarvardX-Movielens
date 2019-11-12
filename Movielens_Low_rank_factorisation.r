# The algorithm is implemented from scratch and relies on nothing but the `Tidyverse` libraries.

library(tidyverse)

# The quality of the training and predictions is measured by the _root mean squared error_
# (RMSE), for which we define a few helper functions (the global variables are defined
# later):
rmse_training <- function(){
  prediction_Z <- rowSums(LF_Model$P[tri_train$userN,] *
                            LF_Model$Q[tri_train$movieN,])
  prediction <- prediction_Z * r_sd + r_m
  sqrt( sum((tri_train$rating - prediction)^2 / nSamples) )
}

rmse_validation <- function(){
  prediction_Z <- rowSums(LF_Model$P[tri_test$userN,] *
                            LF_Model$Q[tri_test$movieN,])
  prediction <- prediction_Z * r_sd + r_m
  sqrt( sum((tri_test$rating - prediction)^2) / nTest )
}

sum_square <- function(v){
  return (sqrt(sum(v^2) / nrow(v)))
}

# The key function updates the model coefficients. Its inputs are:
#
# + a list that contains the $P$ an $Q$ matrices, the training RMSE of those matrices, and
# a logical value indicating whether this RMSE is worse than what it was before the update
# (i.e. did the update diverge).
#
# + a `batch_size` that defines the number of samples to be drawn from the training set. A
# normal gradient descent would use the full training set; by default we only use 10,000
# samples out of 10 million (one tenth of a percent).
#
# + The cost regularisation `lambda` and gradient descent learning parameter `alpha`.
#
# + A number of `times` to run the descent before recalculating the RMSE and exiting the
# function (calculating the RMSE is computationally expensive).
#
#
# The training set used is less rich than the original set. As discussed, it only uses the
# rating (more exactly on the z_score of the rating). Genres, timestamps,... are
# discarded.


# Iterate gradient descent
stochastic_grad_descent <- function(model, times = 1,
                                    batch_size = 10000, lambda = 0.1, alpha = 0.01,
                                    verbose = TRUE) {

  # Run the descent `times` times.
  for(i in 1:times) {

    # Extract a sample of size `batch_size` from the training set.
    spl <- sample(1:nSamples, size = batch_size, replace = FALSE)
    spl_training_values <- tri_train[spl,]


    # Take a subset of `P` and `Q` matching the users and
    # movies in the training sample.
    spl_P <- model$P[spl_training_values$userN,]
    spl_Q <- model$Q[spl_training_values$movieN,]

    # rowSums returns the cross-product for a given user and movie.
    # err is the term inside brackets in the partial derivatives
    # calculation above.
    err <- spl_training_values$rating_z - rowSums(spl_P * spl_Q)

    # Partial derivatives wrt p and q
    delta_P <- -err * spl_Q + lambda * spl_P
    delta_Q <- -err * spl_P + lambda * spl_Q

    model$P[spl_training_values$userN,]  <- spl_P - alpha * delta_P
    model$Q[spl_training_values$movieN,] <- spl_Q - alpha * delta_Q

  }

  # RMSE against the training set
  error <- sqrt(sum(
    (tri_train$rating_z - rowSums(model$P[tri_train$userN,] *
                                    model$Q[tri_train$movieN,]))^2)
    / nSamples )

  # Compares to RMSE before update
  model$WORSE_RMSE <- (model$RMSE < error)
  model$RMSE <- error

  # Print some information to keep track of success
  if (verbose) {
    cat("  # features=", ncol(model$P),
        "  J=",  nSamples * error ^2 +
          lambda/2 * (sum(model$P^2) + sum(model$Q^2)),
        "  Z-scores RMSE=", model$RMSE,
        "\n")
    flush.console()
  }

  return(model)
}


# Now that the functions are defined, we prepare the data sets.
#
# + First load the original data if not already available.

# Load the datasets which were saved on disk after using the course source code.
if(!exists("edx"))        edx <- readRDS("datasets/edx.rds")
if(!exists("validation")) validation <- readRDS("datasets/validation.rds")

# If the datasets are not yet available, the following should be run:
#
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
# dl <- tempfile()
# download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
#
# ratings <- fread(text = gsub("::", "\t", readLines("datasets/ml-10M100K/ratings.dat")),
#                  col.names = c("userId", "movieId", "rating", "timestamp"))
# saveRDS(ratings, "datasets/ratings.rds")
#
# movies <- str_split_fixed(readLines("datasets/ml-10M100K/movies.dat"), "\\::", 3)
# colnames(movies) <- c("movieId", "title", "genres")
# movies <- as.data.frame(movies) %>%
#   mutate(movieId = as.numeric(levels(movieId))[movieId],
#          title = as.character(title),
#          genres = as.character(genres))
# saveRDS(movies, "datasets/movies.rds")
#
# movielens <- left_join(ratings, movies, by = "movieId")
# saveRDS(movielens, "datasets/movielens.rds")
#
#
# # Validation set will be 10% of MovieLens data
#
# set.seed(1, sample.kind="Rounding")
# test_index <- createDataPartition(y = movielens$rating, times = 1,
#                                   p = 0.1, list = FALSE)
# edx <- movielens[-test_index,]
# temp <- movielens[test_index,]
#
# # Make sure userId and movieId in validation set are also in edx set
# validation <- temp %>%
#   semi_join(edx, by = "movieId") %>%
#   semi_join(edx, by = "userId")
#
# # Add rows removed from validation set back into edx set
#
# removed <- anti_join(temp, validation)
# edx <- rbind(edx, removed)
#
# saveRDS(edx, "datasets/edx.rds")
# saveRDS(validation, "datasets/validation.rds")
#



# + Calculate the z-score of all ratings.

# Creates a movie index from 1 to nMovies
r_m <- mean(edx$rating)
r_sd <- sd(edx$rating)

training_set <- edx %>%
  select(userId, movieId, rating) %>%
  mutate(rating_z = (rating - r_m) / r_sd)

test_set <- validation %>%
  select(userId, movieId, rating) %>%
  mutate(rating_z = (rating - r_m) / r_sd)


# + We do not know if there are any gaps in the userId's and movieId's in the datasets.
# They cannot be used as the row numbers of the $P$ and $Q$ matrices. Therefore we count
# how many distinct users and movies there are and create an index to link a movieId
# (resp. userId) to its $Q$ (resp. $P$) -matrix row number.
movieIndex <-
  training_set %>%
  distinct(movieId) %>%
  arrange(movieId) %>%
  mutate(movieN = row_number())

userIndex <-
  training_set %>%
  distinct(userId) %>%
  arrange(userId) %>%
  mutate(userN = row_number())


# + For each movie and user, we calculate its mean rating z-score.
movieMean <-
  training_set %>%
  group_by(movieId) %>%
  summarise(m = mean(rating_z))

userMean <-
  training_set %>%
  group_by(userId) %>%
  summarise(m = mean(rating_z))



# + We can now create the training and validation sets contining the movie index (instead
# of the movieId), user index and ratings (original and z-score).

# Training triplets with z_score
tri_train <- training_set %>%
  left_join(userIndex, by = "userId") %>%
  left_join(movieIndex, by = "movieId") %>%
  select(-userId, -movieId)

tri_test <- test_set %>%
  select(userId, movieId, rating) %>%
  left_join(userIndex, by = "userId") %>%
  left_join(movieIndex, by = "movieId") %>%
  select(-userId, -movieId) %>%
  mutate(rating_z = (rating - r_m)/r_sd,
         error = 0)


nSamples <- nrow(tri_train)
nTest <- nrow(tri_test)

nUsers <- tri_train %>% select(userN) %>%  n_distinct()
nMovies <- tri_train %>% select(movieN) %>%  n_distinct()


# + The $P$ and $Q$ matrices are defined with 3 latent factors to start with.


# number of initial latent factors
nLF <- 3

LF_Model <- list( P = matrix(0, nrow = nUsers, ncol = nLF),
                  Q = matrix(0, nrow = nMovies, ncol = nLF),
                  RMSE = 1000.0,
                  WORSE_RMSE = FALSE)


# The algorithm is implemented from scratch and relies on nothing but the `Tidyverse`
# libraries. + To speed up the training, the matrices are initialised so that the cross
# product is the sum of the movie average z-rating ($m_{movieN}$) and user z-rating
# ($u_{userN}$).

# Features matrices are initialised with:
# Users: 1st column is 1, 2nd is the mean rating (centered), rest is noise
# Movies: 1st column is the mean rating (centered), 2nd is 1, rest is noise
#
# That way, the matrix multiplication will start by giving reasonable value

LF_Model$P[,1] <- matrix(1, nrow = nUsers, ncol = 1)
LF_Model$P[,2] <- as.matrix(userIndex %>%
                              left_join(userMean, by ="userId") %>% select(m))

LF_Model$Q[,1] <- as.matrix(movieIndex %>%
                              left_join(movieMean, by ="movieId") %>% select(m))
LF_Model$Q[,2] <- matrix(1, nrow = nMovies, ncol = 1)


# + Random noise is added to all model parameters, otherwise the gradient descent has
# nowhere to start (zeros wipe everything in the matrix multiplications).

# Add random noise
set.seed(42, sample.kind = "Rounding")
LF_Model$P <- LF_Model$P + matrix(rnorm(nUsers * nLF,
                                        mean = 0,
                                        sd = 0.01),
                                  nrow = nUsers,
                                  ncol = nLF)

LF_Model$Q <- LF_Model$Q + matrix(rnorm(nMovies * nLF,
                                        mean = 0,
                                        sd = 0.01),
                                  nrow = nMovies,
                                  ncol = nLF)

# + We also have a list that keeps track of all the training steps and values.

rm(list_results)
list_results <- tibble("alpha" = numeric(),
                       "lambda" = numeric(),
                       "nFeatures" = numeric(),
                       "rmse_training_z_score" = numeric(),
                       "rmse_training" = numeric(),
                       "rmse_validation" = numeric())


# The main training loop runs as follows:
#
# + We start with 3 features.
#
# + The model is updated in batches of 100 updates. This is done up to 250 times. At each
# time, if the model starts diverging, the learning parameter ($\alpha$) is reduced.
#
# + Once the 250 times have passed, or if $\alpha$ has become incredibly small, or if the
# RMSE doesn't really improve anymoe (by less than 1 millionth), we add another features
# and start again.
Ma
initial_alpha <- 0.1
for(n in 1:100){

  # Current number of features
  number_features <- ncol(LF_Model$P)

  # lambda = 0.01 for 25 features, i.e. for about 2,000,000 parameters.
  # We keep lambda proportional to the number of features
  lambda <- 0.1 * (nUsers + nMovies) * number_features / 2000000

  alpha <- initial_alpha

  cat("CURRENT FEATURES: ", number_features,
      "---- Pre-training validation RMSE = ", rmse_validation(), "\n")

  list_results <- list_results %>% add_row(alpha = alpha,
                                           lambda = lambda,
                                           nFeatures = number_features,
                                           rmse_training_z_score = LF_Model$RMSE,
                                           rmse_training = rmse_training(),
                                           rmse_validation = rmse_validation())

  for (i in 1:250){
    pre_RMSE <- LF_Model$RMSE
    LF_Model <- stochastic_grad_descent(model = LF_Model,
                                        times = 100,
                                        batch_size = 1000 * number_features,
                                        alpha = alpha,
                                        lambda = lambda)

    list_results <- list_results %>% add_row(alpha = alpha,
                                             lambda = lambda,
                                             nFeatures = number_features,
                                             rmse_training_z_score = LF_Model$RMSE,
                                             rmse_training = rmse_training(),
                                             rmse_validation = rmse_validation())

    if (LF_Model$WORSE_RMSE) {
      alpha <- alpha / 2
      cat("Decreasing gradient parameter to: ", alpha, "\n")
    }

    if (initial_alpha / alpha > 1000 |
        abs( (LF_Model$RMSE - pre_RMSE) / pre_RMSE) < 1e-6) {
      break()
      }
  }


  # RMSE against validation set:
  rmse_validation_post <- rmse_validation()
  cat("CURRENT FEATURES: ", number_features,
      "---- POST-training validation RMSE = ", rmse_validation_post, "\n")

  # if (number_features == 12){
  #   break()
  # }


  # Add k features
  k_features <- 1
  LF_Model$P <- cbind(LF_Model$P,
                         matrix(rnorm(nrow(LF_Model$P) * k_features,
                                      mean = 0,
                                      sd = sd(LF_Model$P)/100),
                                nrow = nrow(LF_Model$P),
                                ncol = k_features))

  LF_Model$Q <- cbind(LF_Model$Q,
                      matrix(rnorm(nrow(LF_Model$Q) * k_features,
                                   mean = 0,
                                   sd = sd(LF_Model$Q)/100),
                             nrow = nrow(LF_Model$Q),
                             ncol = k_features))

}

saveRDS(list_results, "datasets/LRMF_results.rds")
