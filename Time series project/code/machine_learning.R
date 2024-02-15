
library(car)    # Test di autocorrelazione
library(lmtest) # Test di eteroschedasticità

############### tirare fuori R2

rm(list = ls())  # cleaning the working space

set.seed(1234)
setwd("C:/Users/fsilv/Desktop/agrimonia")
load("./response_variable.RData")
load("./covariate_variables.RData")

## diving into training and validation
y_train = y[seq(round(length(y)*0.8))]
y_validation = y[-seq(round(length(y)*0.8))]


data_train <- matrix(NA, nrow = round(length(y)*0.8), ncol = length(Schivenoglia.imputed) - 1)
data_validation <- matrix(NA, nrow = length(y) - round(length(y)*0.8), ncol = length(Schivenoglia.imputed) - 1)

colnames(data_train) <- names(Schivenoglia.imputed)[seq(2, length(Schivenoglia.imputed))]
colnames(data_validation) <- names(Schivenoglia.imputed)[seq(2, length(Schivenoglia.imputed))]


for (i in 2:length(Schivenoglia.imputed)) {
  data_train[,i - 1] <- as.numeric(Schivenoglia.imputed[[i]][seq(round(length(y)*0.8))])
  data_validation[,i - 1] <- as.numeric(Schivenoglia.imputed[[i]][-seq(round(length(y)*0.8))])
}

x_train <- data.matrix(data.frame(data_train))
x_validation <- data.matrix(data.frame(data_validation))


# Statistics 
calculate_metrics <- function(model, y_train_pred, y_train, y_validation_pred, y_validation) {
  
  rmse_train <- sqrt(mean((y_train_pred - y_train)^2))
  rmse_validation <- sqrt(mean((y_validation_pred - y_validation)^2))
  
  # Return the metrics
  metrics <- list(
    rmse_train = rmse_train,
    rmse_validation = rmse_validation
  )
  return(metrics)
}

################################################################################
############################ XGBOOST
################################################################################

if(!require('xgboost')) install.packages('xgboost')
if(!require('zoo')) install.packages('zoo')
if(!require('lmtest')) install.packages('lmtest')
if(!require('MLmetrics')) install.packages('MLmetrics')
if(!require('DiagrammeR')) install.packages('DiagrammeR')

library(xgboost)
library(zoo)
library(lmtest) 
library(MLmetrics)
library(DiagrammeR)

xgb_train <- xgb.DMatrix(data = x_train, label = y_train)
xgb_test <- xgb.DMatrix(data = x_validation, label = y_validation)
xgb_params <- list(
  booster = "gbtree",
  eta = 0.01,
  gamma = 1,
  max_depth = 8,
  subsample = 0.75,
  colsample_bytree = 1,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)


watchlist <- list(train=xgb_train,  test=xgb_test)
xgb_model <- xgb.train(xgb_params, data = xgb_train, watchlist, nrounds = 1000, verbose = 1)

# Tuning performances
opt_models <- list()
opt_models$min_val_rmse <- list()
opt_models$min_ite <- list()
opt_models$train_rmse_ite <- list()
opt_models$validation_ts_rmse <- list()

for (depth in 1:6) {
  xgb_params <- list(
    booster = "gbtree",
    eta = 0.01,
    gamma = 1,
    max_depth = depth,
    subsample = 0.75,
    colsample_bytree = 1,
    objective = "reg:squarederror",
    eval_metric = "rmse"
  )
  
  # Train the XGBoost model
  xgb_model_i <- xgb.train(xgb_params, data = xgb_train, watchlist, nrounds = 1000)
  eval_data <- xgb_model_i$evaluation_log[,3]
  etrain_data <- xgb_model_i$evaluation_log[,2]
  
  for (i in 1:max(index(eval_data))) {
    if (eval_data[i] == min(eval_data)) {
      x_min_ite = i;
    }
  }
  
  # Store the results in opt_models
  opt_models$min_ite[[paste0("max_depth_", depth)]] <- x_min_ite
  opt_models$min_val_rmse[[paste0("max_depth_", depth)]] <- min(eval_data)
  opt_models$train_rmse_ite[[paste0("max_depth_", depth)]] <- etrain_data[i]
  opt_models$validation_ts_rmse[[paste0("max_depth_", depth)]] <- eval_data
}


par(mfrow = c(3,2))

for (i in 1:6) {
  plot(as.matrix(opt_models$validation_ts_rmse[[i]]), 
       type = "o",                    
       main =paste("Tuning: ", i, "max_depth"),     
       xlab = "iteration",          
       ylab = "Validation RMSE"           
  )
  abline(v = opt_models$min_ite[i], col = "blue", lty = "dashed", lwd = 2)
  legend("topright", legend = "Minimum", col = "blue", lty = 1, cex = 1.5)
  axis(1, at = opt_models$min_ite[i], labels = , col.axis = "blue", font = 2) # Aggiungi valore sull'asse x
  axis(2, at = opt_models$min_val_rmse[i], labels = round(opt_models$min_val_rmse[[i]],2), col.axis = "blue", font = 2) # Aggiungi valore sull'asse y
}

min <- opt_models$min_val_rmse[[1]]
max_iterations <- 1
for (i in 1:6) {
  if(min > opt_models$min_val_rmse[[i]]){
    min <- opt_models$min_val_rmse[[i]]
    max_iterations <- as.numeric(as.character(unlist(i)))
  }
}



# Final model 
xgb_params <- list(
  booster = "gbtree",
  eta = 0.01,
  gamma = 1,
  max_depth = max_iterations,
  subsample = 0.75,
  colsample_bytree = 1,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
xgb_model <- xgb.train(xgb_params, data = xgb_train, watchlist, nrounds = as.numeric(as.character(unlist(opt_models$min_ite[max_iterations]))) + 10, verbose = 0)



importance_matrix <- xgb.importance(
  feature_names = colnames(xgb_train), 
  model = xgb_model
)

importance_matrix
par(mfrow = c(1,1))
xgb.plot.importance(importance_matrix[seq(1,10)])


y_validation_pred <- predict(xgb_model, xgb_test)

# plotting prediction vs observed data
ts_validation_pred = zoo(y_validation_pred, seq(from = as.Date("2020-10-19"), to = as.Date("2021-12-31"), by = 1)) 
ts_validation_obser = zoo(y_validation, seq(from = as.Date("2020-10-19"), to = as.Date("2021-12-31"), by = 1)) 

par(mfrow = c(2,1))
plot(ts_validation_pred, 
     type = "l",                    
     main = "Predicted AQ_nh3 values", 
     xlab = "Date", 
     ylab = "AQ_nh3",
     ylim = c(0, 40)
)


abline(h = mean(ts_validation_pred), col = "blue", lwd = 2, lty = 2)
text(x = as.Date("2021-01-01"), y = mean(ts_validation_pred) + 6, labels = paste("Mean:", round(mean(ts_validation_pred), 1)), col = "blue", pos = 3, font = 2)

plot(ts_validation_obser, 
     type = "l",                          
     main = "Observed AQ_nh3 values",
     xlab = "Date",
     ylab = "AQ_nh3",
     ylim = c(0, 40)
)

abline(h = mean(ts_validation_obser), col = "blue", lwd = 2, lty = 2)
text(x = as.Date("2021-01-01"), y = mean(ts_validation_pred) + 6, labels = paste("Mean:", round(mean(ts_validation_pred), 1)), col = "blue", pos = 3, font = 2)


# Compute uncertainty of predictions
num_rounds <- 10
predictions <- matrix(NA, nrow = length(y_validation_pred), ncol = num_rounds)


for (i in 1:num_rounds) {
  xgb_model_i <- xgb.train(xgb_params, data = xgb_train, watchlist, nrounds = x_min_ite)
  predictions[, i] <- predict(xgb_model_i, xgb_test)
}
prediction_uncertainty <- apply(predictions, 1, sd)


# Residual distribution
res_xgb = y_validation_pred - y_validation
rmse_xgb_validation = sqrt(mean(res_xgb^2))
par(mfrow = c(1,4))
hist(res_xgb,40,
     xlab = "Value",
     main = "Empirical distribution of residuals") 

plot(res_xgb, pch = "o", col = "blue" ,
     ylab = "Residual", main = paste0("Residual plot - mean:",round(mean(res_xgb),digits = 4),
                                      "- var:", round(var(res_xgb),digits = 4)))
abline(c(0,0),c(0,length(res_xgb)), col= "red", lwd = 2)

boxplot(res_xgb, ylab = "Residuals", main = "Outliers")$out

qqnorm(res_xgb, main='Residuals')
qqline(res_xgb)

plot(scale(y_validation), res_xgb, xlab = "y_validation", ylab = "Residuals XGBoost", main = "Scatter plot of residuals")


# Print RMSE and prediction uncertainty
y_train_pred <- predict(xgb_model, x_train)
y_validation_pred <- predict(xgb_model, x_validation)

xgb_metrics <- calculate_metrics(xgb_model, y_train_pred, y_train, y_validation_pred, y_validation)
print(xgb_metrics)
print(paste("Prediction Uncertainty (Standard Deviation):", mean(prediction_uncertainty)))

# test
shapiro.test(res_xgb)                             # H0: normally distributed
bptest(lm(res_xgb ~ y_validation))                # H0: omoschedasticity
Box.test(res_xgb, lag = 7, type = "Ljung-Box")    # H0: no correlation
Box.test(res_xgb, lag = 30, type = "Ljung-Box")
Box.test(res_xgb, lag = 365, type = "Ljung-Box")

# R2_Score(y_pred = y_train_pred, y_true = y_train)
# R2_Score(y_pred = y_validation_pred, y_true = y_validation)

################################################################################
############################### Prophet
################################################################################

if(!require('prophet')) install.packages('prophet')

library(prophet)
library(lubridate)

date_sequence_train <- seq(as.Date("2016-01-01"), as.Date("2020-10-19"), by = "day")
date_sequence_validation <- seq(as.Date("2020-10-19"), as.Date("2021-12-30"), by = "day")

df <- data.frame(ds = date_sequence_train)
df$y <- y[seq(round(length(y)*0.8))]


prophet_model <- prophet(daily.seasonality = FALSE)
prophet_model <- add_country_holidays(prophet_model, country_name = 'IT')
prophet_model <- fit.prophet(prophet_model, df = df)

future <- data.frame(ds = date_sequence_validation)
tail(future)

forecast <- predict(prophet_model, future)
tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])

plot(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
plot(prophet_model, forecast)

prophet_plot_components(prophet_model, forecast)

# Residual distribution
res_prophet = y_validation - forecast$yhat
rmse_prophet_validation = sqrt(mean(res_prophet^2))
par(mfrow = c(1,5))
hist(res_prophet,40,
     xlab = "Value",
     main = "Empirical distribution of residuals") 

plot(res_prophet, pch = "o", col = "blue" ,
     ylab = "Residual", main = paste0("Residual plot - mean:",round(mean(res_prophet),digits = 4),
                                      "- var:", round(var(res_prophet),digits = 4)))
abline(c(0,0),c(0,length(res_prophet)), col= "red", lwd = 2)

boxplot(res_prophet, ylab = "Residuals", main = "Outliers")$out

qqnorm(res_prophet, main='Residuals')
qqline(res_prophet)
plot(scale(y_validation), res_prophet, xlab = "y_validation", ylab = "Residuals prophet", main = "Scatter plot of residuals")


dyplot.prophet(prophet_model, forecast)

# training statistics
y_train_pred <- predict(prophet_model, df)
y_validation_pred <- predict(prophet_model, future)

prophet_metrics <- calculate_metrics(prophet_model, y_train_pred$yhat, y_train, y_validation_pred$yhat, y_validation)
print(prophet_metrics)

shapiro.test(res_prophet)                             # H0: normally distributed
bptest(lm(res_prophet ~ y_validation))                # H0: omoschedasticity
Box.test(res_prophet, lag = 7, type = "Ljung-Box")    # H0: no correlation
Box.test(res_prophet, lag = 30, type = "Ljung-Box")
Box.test(res_prophet, lag = 365, type = "Ljung-Box")

################################################################################
############################### LTSM
################################################################################

if(!require('quantmod')) install.packages('quantmod')
if(!require('keras')) install.packages('keras')

library(quantmod)
library(keras)


#  define the train and test split
split_data <- function(stock, lookback) {
  data_raw <- as.matrix(stock) # convert to matrix
  data <- array(dim = c(0, lookback, ncol(data_raw)))
  
  # create all possible sequences of length lookback
  for (index in 1:(nrow(data_raw) - lookback)) {
    data <- rbind(data, data_raw[index:(index + lookback - 1), ])
  }
  
  test_set_size <- round(0.2 * nrow(data)) + 1
  train_set_size <- nrow(data) - test_set_size
  
  x_train <- data[1:train_set_size, 1:(lookback - 1), drop = FALSE]
  y_train <- data[1:train_set_size, lookback, drop = FALSE]
  
  x_test <- data[(train_set_size + 1):nrow(data), 1:(lookback - 1), drop = FALSE]
  y_test <- data[(train_set_size + 1):nrow(data), lookback, drop = FALSE]
  
  return(list(x_train = x_train, y_train = y_train,
              x_test = x_test, y_test = y_test))
}

y_zoo <- xts(y, order.by = seq(from = as.Date("2016-01-01"), to = as.Date("2021-12-31"), by = 1), unique = TRUE, tzone = "UTC")

#set apple data as dataframe
data<- data.frame(
  date = seq(from = as.Date("2016-01-01"), to = as.Date("2021-12-31"), by = 1),
  AQ_nh3 = as.numeric(y_zoo))
head(data)

# Scaling the response variable
AQ_nh3 <- scale(data$AQ_nh3)
AQ_nh3 <- data.frame(AQ_nh3)
head(AQ_nh3)


#divide data into train and test 80% - 20%
lookback <- 8 # choose sequence length
split_data <- split_data(AQ_nh3, lookback) # assuming "price" is a data frame
x_train <- split_data$x_train
y_train <- split_data$y_train
x_test <- split_data$x_test
y_test <- split_data$y_test

cat(paste('x_train.shape = ', dim(x_train), '\n'))
cat(paste('y_train.shape = ', dim(y_train), '\n'))
cat(paste('x_test.shape = ', dim(x_test), '\n'))
cat(paste('y_test.shape = ', dim(y_test), '\n'))

#decide hyperparameters
input_dim <- 1
hidden_dim <- 32
num_layers <- 3
output_dim <- 1
num_epochs <- 100

# Reshape the training and test data to have a 3D tensor shape
x_train <- array_reshape(x_train, c(dim(x_train)[1], lookback-1, input_dim))
x_test <- array_reshape(x_test, c(dim(x_test)[1], lookback-1, input_dim))

# Define the LSTM model using Keras
ltsm_model <- keras_model_sequential() %>%
  layer_lstm(units = hidden_dim, return_sequences = TRUE, input_shape = c(lookback-1, input_dim)) %>%
  layer_lstm(units = hidden_dim) %>%
  layer_dense(units = output_dim)

# Compile the model using the mean squared error loss and the Adam optimizer
ltsm_model %>% compile(loss = "mean_squared_error", optimizer = optimizer_adam(learning_rate = 0.01))
# Train the model on the training data
history <- ltsm_model %>% fit(x_train, y_train, epochs = num_epochs, batch_size = 16, validation_data = list(x_test, y_test))

# Extract predictions from the stimated model
y_train_pred <- ltsm_model %>% predict(x_train)
y_test_pred <- ltsm_model %>% predict(x_test)

# Rescale the predictions and original values
y_train_pred_orig <- y_train_pred * sd(data$AQ_nh3) + mean(data$AQ_nh3)
y_train_orig <- y_train * sd(data$AQ_nh3) + mean(data$AQ_nh3)
y_test_pred_orig <- y_test_pred * sd(data$AQ_nh3) + mean(data$AQ_nh3)
y_test_orig <- y_test * sd(data$AQ_nh3) + mean(data$AQ_nh3)



# plot the predictions from traning data and train Loss
# Set up the layout of the plots
par(mfrow = c(1, 2))
options(repr.plot.width=15, repr.plot.height=5)
# Plot the training and predicted values
plot(y_train_orig, type = "l",main="Daily AQ_nh3 values", col = "green", xlab = "Days", ylab = "AQ_nh3", lwd=3)
#lines(y_train_orig, col = "green")
lines(y_train_pred_orig, col = "red")
legend(x = "topleft", legend = c("Train", "Train Predictions"), col = c("green", "red"), lwd = 2)
grid()
# Plot the loss of training data
plot(history$metrics$loss, type = "l",main="Traning Loss", xlab = "Epochs", ylab = "Loss", col = "blue",lwd=3)
grid()


# Shift the predicted values to start from where the training data predictions end
shift <- length(y_train_pred_orig)
y_test_pred_orig_shifted <- c(rep(NA, shift), y_test_pred_orig[,1])


# Plot the training and predicted values
par(mfrow = c(1, 1))
options(repr.plot.width=12, repr.plot.height=8)
plot(data$AQ_nh3, type = "l", main="LSTM AQ_nh3 predictions",col = "green", xlab = "Days", ylab = "AQ_nh3",lwd=3)
lines(y_train_pred_orig, col = "blue",lwd=3)
lines(y_test_pred_orig_shifted, col = "red",lwd=3)
legend(x = "topleft", legend = c("Original", "Train Predictions","Test-Prediction"), col = c("green","blue" ,"red"), lwd = 2)
grid()


mse <- history$metrics$val_loss[length(history$metrics$val_loss)]
mse <- round(mse, 7)
mse


# Residual distribution
res_ltsm = y_test_orig - y_test_pred_orig
rmse_ltsm_validation = sqrt(mean(res_ltsm^2))
par(mfrow = c(1,5))
hist(res_ltsm,40,
     xlab = "Value",
     main = "Empirical distribution of residuals") 

plot(res_ltsm, pch = "o", col = "blue" ,
     ylab = "Residual", main = paste0("Residual plot - mean:",round(mean(res_ltsm),digits = 4),
                                      "- var:", round(var(res_ltsm),digits = 4)))
abline(c(0,0),c(0,length(res_ltsm)), col= "red", lwd = 2)

boxplot(res_ltsm, ylab = "Residuals", main = "Outliers")$out

qqnorm(res_ltsm, main='Residuals')
qqline(res_ltsm)

plot(scale(y_validation), res_ltsm, xlab = "y_validation", ylab = "Residuals LTSM", main = "Scatter plot of residuals")


# training statistics
ltsm_metrics <- calculate_metrics(ltsm_model, y_train_pred_orig, y_train_orig, y_test_pred_orig, y_test_orig)
print(ltsm_metrics)

shapiro.test(res_ltsm)                             # H0: normally distributed
bptest(lm(res_ltsm ~ y_validation))                # H0: omoschedasticity
Box.test(res_ltsm, lag = 7, type = "Ljung-Box")    # H0: no correlation
Box.test(res_ltsm, lag = 30, type = "Ljung-Box")
Box.test(res_ltsm, lag = 365, type = "Ljung-Box")


#########################################################

par(mfrow = c(1,3))
plot(scale(y_validation), res_xgb, xlab = "y_validation", ylab = "Residuals XGBoost", main = "Scatter plot of residuals")
plot(scale(y_validation), res_prophet, xlab = "y_validation", ylab = "Residuals Prophet", main = "Scatter plot of residuals")
plot(scale(y_validation), res_ltsm, xlab = "y_validation", ylab = "Residuals LTSM", main = "Scatter plot of residuals")

par(mfrow = c(1,3))
hist(res_xgb, 40,
     xlab = "Value",
     main = "Empirical distribution of residuals - XGBoost") 
hist(res_prophet, 40,
     xlab = "Value",
     main = "Empirical distribution of residuals - Prophet") 
hist(res_ltsm, 40,
     xlab = "Value",
     main = "Empirical distribution of residuals - LTSM") 
