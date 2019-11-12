# From the book
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# START
# 
library(tidyverse)

edx <- readRDS("datasets/edx.rds")
validation <- readRDS("datasets/validation.rds")

# Create a database of all the ratings, but only the ratings
movie_ratings <- rbind(edx, validation)
movie_ratings <- movie_ratings %>% select(userId, movieId, rating)
saveRDS(movie_ratings, "datasets/ratings_full.rds")

rm(edx, validation)

#############################################

library(caret)

set.seed(42, sample.kind = "Rounding")

result <- data.frame()
n_runs <- 100

start_time = Sys.time()
for (i in 1:n_runs)  {
  # Create 100k dataset
  rating_sample <- 
    movie_ratings %>% 
    sample_n(size = 100000, replace = FALSE)
  
  test_index <- createDataPartition(y = rating_sample$rating, 
                                    times = 1, 
                                    p = 0.2, 
                                    list = FALSE)
  
  train_set <- rating_sample[-test_index,]
  test_set <- rating_sample[test_index,]
  
  # Awful. Never look at the test set to make your life easier!
  test_set <- test_set %>%
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")
  
  
  lambdas <- seq(0, 10, 0.25)
  
  mu <- mean(train_set$rating)
  just_the_sum <- train_set %>% 
    group_by(movieId) %>% 
    summarize(s = sum(rating - mu), n_i = n())
  
  rmses <- sapply(lambdas, function(l){
    
    predicted_ratings <- test_set %>% 
      left_join(just_the_sum, by='movieId') %>% 
      mutate(b_i = s/(n_i+l)) %>%
      mutate(pred = mu + b_i) %>%
      pull(pred)
    
    return(RMSE(predicted_ratings, test_set$rating))
  })
  
  best_rmse <- min(rmses)
  best_lambda <- lambdas[which.min(rmses)]
  
  result <- result %>% rbind(data.frame(mu = mu, rmse = rmses[1], rmse_reg = best_rmse, lambda = best_lambda))
}

end_time = Sys.time()
cat("time per run:", (end_time - start_time)/n_runs)


# plot improvement of regularisation in % against starting RMSE
plot(result$rmse, 100 * (result$rmse_reg - result$rmse) / result$rmse)









