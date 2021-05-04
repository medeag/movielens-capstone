##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")
cols<-colnames(movielens)
#extract year from timestamp since it may influence rating
movielens<-movielens %>% mutate(year=format(as.POSIXct(timestamp,origin="1970-01-01"), "%Y"))
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#split edx data into train and test set
test_index_edx <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index_edx,]
tmp_test_set <- edx[test_index_edx,]

# Ensure userId and movieId in test_set are also in train set
test_set <- tmp_test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed_test <- anti_join(tmp_test_set, test_set)
train_set <- rbind(train_set, removed_test)

#Naive rmse model
mu <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse

rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

#Predict using only movie effect
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

i_rmse <-RMSE(predicted_ratings, test_set$rating)
rmse_results <- rmse_results %>% add_row(method = "Movie", RMSE = i_rmse)

#Predict using only user effect
user_avgs <- train_set %>% 
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu))

predicted_ratings <- test_set %>% 
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_u) %>%
  pull(pred)

u_rmse <-RMSE(predicted_ratings, test_set$rating)
rmse_results <- rmse_results %>% add_row(method = "User", RMSE = u_rmse)

#Predict using only year effect
year_avgs <- train_set %>% 
  group_by(year) %>%
  summarize(b_y = mean(rating - mu))

predicted_ratings <- test_set %>% 
  left_join(year_avgs, by='year') %>%
  mutate(pred = mu + b_y) %>%
  pull(pred)

y_rmse <-RMSE(predicted_ratings, test_set$rating)
rmse_results <- rmse_results %>% add_row(method = "Year", RMSE = y_rmse)

#Predict using user & movie effect
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

i_u_rmse <-RMSE(predicted_ratings, test_set$rating)
rmse_results <- rmse_results %>% add_row(method = "Movie/User", RMSE = i_u_rmse)

#Predict using user & movie & year effect
year_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  mutate(pred = mu + b_i + b_u + b_y ) %>%
  pull(pred)

i_u_y_rmse <-RMSE(predicted_ratings, test_set$rating)
rmse_results <- rmse_results %>% add_row(method = "Movie/User/Year", RMSE = i_u_y_rmse)

#Predict using user & movie &  genres effect
genres_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u ))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

i_u_g_rmse<-RMSE(predicted_ratings, test_set$rating)
rmse_results <- rmse_results %>% add_row(method = "Movie/User/Genres", RMSE = i_u_g_rmse)


#Predict using user & movie & year & genres effect
genres_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_y))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
  pull(pred)

i_u_y_g_rmse<-RMSE(predicted_ratings, test_set$rating)
rmse_results <- rmse_results %>% add_row(method = "Movie/User/Year/Genres", RMSE = i_u_y_g_rmse)

rmse_results

#--- Regularization ---

lambdas <- seq(0, 5, 0.25)
rmses <- sapply(lambdas, function(lambda) {
  # Calculate the average by movie
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  # Calculate the average by user
  
  b_u <- train_set %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))
  
  # Calculate the average by year
  b_y <- train_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(year) %>%
    summarize(b_y = mean(rating - mu - b_i  - b_u))
  
  # Calculate the average by genres
  b_g <- train_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_y, by='year') %>%
    group_by(genres) %>%
    summarize(b_g = mean(rating - mu - b_i - b_u - b_y))
  
  # Compute the predicted ratings on test_set dataset
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_y, by='year') %>%
    left_join(b_g, by='genres') %>%
    mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
    pull(pred)
  
  # Predict the RMSE on the test set
  
  return (RMSE(predicted_ratings,test_set$rating))
})

# Get the lambda value that minimize the RMSE

min_lambda <- lambdas[which.min(rmses)]
i_u_y_g_cv_rmse<-min(rmses)
rmse_results <- rmse_results %>% add_row(method = "Movie/User/Year/Genres cross-validation", RMSE = i_u_y_g_cv_rmse)


#--- validation ---

# Calculate the average by movie
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + min_lambda))

# Calculate the average by user

b_u <- edx %>%
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + min_lambda))

# Calculate the average by year
b_y <- edx %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i  - b_u))

# Calculate the average by genres
b_g <- edx %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_y, by='year') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_y))

# Compute the predicted ratings on validation dataset

predicted_ratings <- validation %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_y, by='year') %>%
  left_join(b_g, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
  pull(pred)

# Predict the RMSE on the validation set

finalRMSE<- RMSE(predicted_ratings,validation$rating)

finalRMSE


