rm(list = ls())  # cleaning the working space

if(!require('bboot')) install.packages('bboot')
if(!require('ggplot2')) install.packages('ggplot2')
if(!require('forecast')) install.packages('forecast')
if(!require('imputeTS')) install.packages('imputeTS')
if(!require('imputeFin')) install.packages('imputeFin')
if(!require('stats')) install.packages('stats')
if(!require('tseries')) install.packages('tseries')

library(bboot)
library(ggplot2)
library(forecast)
library(imputeTS)
library(imputeFin)
library(stats)
library(tseries)


set.seed(1234)
setwd("C:/Users/fsilv/Desktop/agrimonia")
load("./Agrimonia_Dataset_v_3_0_0.Rdata") 
Schivenoglia <- AgrImOnIA_Dataset_v_3_0_0[AgrImOnIA_Dataset_v_3_0_0$IDStations == 703,]


########################################################################################
# ImputeTS tricks

ggplot_na_distribution(Schivenoglia$AQ_nh3) # shows where the missing values are nicely
ggplot_na_distribution2(Schivenoglia$AQ_nh3) # another visualization
ggplot_na_gapsize(Schivenoglia$AQ_nh3) # show barplot with frequency and length of gap with missing data

summary(Schivenoglia)

y <- Schivenoglia$AQ_nh3

# Eliminazione degli outliers

q1 <- quantile(y, 0.25, na.rm = TRUE)
q3 <- quantile(y, 0.75, na.rm = TRUE)

# Calcoliamo l'IQR (Interquartile Range)
iqr <- q3 - q1

# Definiamo gli estremi dell'intervallo considerando un fattore di estensione di 2.7 per il 99% di confidenza
lower_limit <- q1 - 1.5 * iqr
upper_limit <- q3 + 1.5 * iqr

# Selezioniamo solo i valori che cadono all'interno dell'intervallo
y[y < lower_limit | y > upper_limit] <- NaN
cleaned_data <- y


plot(y,type='l')

ggplot_na_distribution(y)

imputation <- function(y) {
  
nan_list <- which(is.na(y))

blocked.index= blockboot(y, N = 3000,K = 1000, L = 90)
blocked.index = blocked.index[1:2192,] # prendo solamente 2192 valori dei 3000 

y<- na_kalman(y)

ggplot_na_distribution(y)

valori_estrazione <- matrix(NA, nrow = nrow(blocked.index), ncol = ncol(blocked.index))

variance <- matrix(NA,nrow = 1, ncol = ncol(blocked.index))


for (i in 1:1000) {
  indexes = blocked.index[,i]
  valori_estrazione[,i] <- y[indexes]
  variance[,i] <- var(valori_estrazione[,i])
  print(var(valori_estrazione[,i]))
}

mean_variance <- mean(variance)

for(index in nan_list){
  y[index] <- y[index] + rnorm(1, mean = 0, sd = sqrt(mean_variance))
}

y[y < 0 ] <- 0

ggplot_na_distribution(y)

return(y)

}

y <- imputation(y)
ggplot_na_distribution(y)

trend_train_size = round(0.8*length(y))
trend_validation_size = length(y)-trend_train_size

subset_train = 1:trend_train_size
subset_validation = trend_train_size:length(y)

## IMPUTATION ##
Schivenoglia.imputed = {}

Schivenoglia.imputed$AQ_nh3 <- y

plot(ts(na_kalman(Schivenoglia$AQ_no2, model = "auto.arima")))
ggplot_na_distribution(Schivenoglia$AQ_no2)
Schivenoglia.imputed$AQ_no2 <- imputation(Schivenoglia$AQ_no2)
ggplot_na_distribution(Schivenoglia.imputed$AQ_no2)

plot(ts(na_kalman(Schivenoglia$AQ_nox, model = "auto.arima")))
ggplot_na_distribution(Schivenoglia$AQ_nox)
Schivenoglia.imputed$AQ_nox <- imputation(Schivenoglia$AQ_nox)
ggplot_na_distribution(Schivenoglia.imputed$AQ_nox)

plot(ts(na_kalman(Schivenoglia$AQ_pm10, model = "auto.arima")))
ggplot_na_distribution(Schivenoglia$AQ_pm10)
Schivenoglia.imputed$AQ_pm10 <- imputation(Schivenoglia$AQ_pm10)
ggplot_na_distribution(Schivenoglia.imputed$AQ_pm10)

plot(ts(na_kalman(Schivenoglia$AQ_pm25, model = "auto.arima")))
ggplot_na_distribution(Schivenoglia$AQ_pm25)
Schivenoglia.imputed$AQ_pm25 <- imputation(Schivenoglia$AQ_pm25)
ggplot_na_distribution(Schivenoglia.imputed$AQ_pm25)

plot(ts(na_kalman(Schivenoglia$AQ_co, model = "auto.arima")))
ggplot_na_distribution(Schivenoglia$AQ_co)
Schivenoglia.imputed$AQ_co <- imputation(Schivenoglia$AQ_co)
ggplot_na_distribution(Schivenoglia.imputed$AQ_co)

plot(ts(na_kalman(Schivenoglia$AQ_so2, model = "auto.arima")))
ggplot_na_distribution(Schivenoglia$AQ_so2)
Schivenoglia.imputed$AQ_so2 <- imputation(Schivenoglia$AQ_so2)
ggplot_na_distribution(Schivenoglia.imputed$AQ_so2)


# Non sono presenti NaN values
Schivenoglia.imputed$WE_wind_speed_100m_max <- Schivenoglia$WE_wind_speed_100m_max
ggplot_na_distribution(Schivenoglia.imputed$WE_wind_speed_100m_max)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_wind_speed_100m_mean <- Schivenoglia$WE_wind_speed_100m_mean
ggplot_na_distribution(Schivenoglia.imputed$WE_wind_speed_100m_mean)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_wind_speed_10m_max <- Schivenoglia$WE_wind_speed_10m_max
ggplot_na_distribution(Schivenoglia.imputed$WE_wind_speed_10m_max)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_wind_speed_10m_mean <- Schivenoglia$WE_wind_speed_10m_mean
ggplot_na_distribution(Schivenoglia.imputed$WE_wind_speed_10m_mean)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_temp_2m <- Schivenoglia$WE_temp_2m
ggplot_na_distribution(Schivenoglia.imputed$WE_temp_2m)

# Non sono presenti NaN values NO NUMBER -----------
Schivenoglia.imputed$WE_mode_wind_direction_10m <- Schivenoglia$WE_mode_wind_direction_10m
print(Schivenoglia.imputed$WE_mode_wind_direction_10m)

# Non sono presenti NaN values NO NUMBER -----------
Schivenoglia.imputed$WE_mode_wind_direction_100m <- Schivenoglia$WE_mode_wind_direction_100m
print(Schivenoglia.imputed$WE_mode_wind_direction_100m)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_tot_precipitation <- Schivenoglia$WE_tot_precipitation
ggplot_na_distribution(Schivenoglia.imputed$WE_tot_precipitation)

# Non sono presenti NaN values NO NUMBER --------------------
Schivenoglia.imputed$WE_precipitation_t <- Schivenoglia$WE_precipitation_t

# Non sono presenti NaN values
Schivenoglia.imputed$WE_solar_radiation <- Schivenoglia$WE_solar_radiation
ggplot_na_distribution(Schivenoglia.imputed$WE_solar_radiation)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_surface_pressure <- Schivenoglia$WE_surface_pressure
ggplot_na_distribution(Schivenoglia.imputed$WE_surface_pressure)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_rh_min <- Schivenoglia$WE_rh_min
ggplot_na_distribution(Schivenoglia.imputed$WE_rh_min)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_rh_mean <- Schivenoglia$WE_rh_mean
ggplot_na_distribution(Schivenoglia.imputed$WE_rh_mean)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_rh_max <- Schivenoglia$WE_rh_max
ggplot_na_distribution(Schivenoglia.imputed$WE_rh_max)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_blh_layer_max <- Schivenoglia$WE_blh_layer_max
ggplot_na_distribution(Schivenoglia.imputed$WE_blh_layer_max)

# Non sono presenti NaN values
Schivenoglia.imputed$WE_blh_layer_min <- Schivenoglia$WE_blh_layer_min
ggplot_na_distribution(Schivenoglia.imputed$WE_blh_layer_min)

# -----------------------PERIODICA ---------- !!!!!!!!!!!!!!!!!!!!!!!
#Schivenoglia.imputed$EM_nh3_livestock_mm <- Schivenoglia$EM_nh3_livestock_mm
#ggplot_na_distribution(Schivenoglia.imputed$EM_nh3_livestock_mm)
#plot(Schivenoglia.imputed$EM_nh3_livestock_mm)

# -----------------------PERIODICA ----------!!!!!!!!! 
#Schivenoglia.imputed$EM_nh3_agr_soils <- Schivenoglia$EM_nh3_agr_soils
#ggplot_na_distribution(Schivenoglia.imputed$EM_nh3_agr_soils)


# -----------------------PERIODICA ---------- !!!!!!!!!!!!!!
#Schivenoglia.imputed$EM_nh3_agr_waste_burn <- Schivenoglia$EM_nh3_agr_waste_burn
#ggplot_na_distribution(Schivenoglia.imputed$EM_nh3_agr_waste_burn)

# -----------------------PERIODICA ---------- !!!!!!!!!!!!!!!!!
#Schivenoglia.imputed$EM_nh3_sum <- Schivenoglia$EM_nh3_sum
#ggplot_na_distribution(Schivenoglia.imputed$EM_nh3_sum)

# -----------------------PERIODICA ---------- 
# Schivenoglia.imputed$EM_nox_traffic <- Schivenoglia$EM_nox_traffic
# ggplot_na_distribution(Schivenoglia.imputed$EM_nox_traffic)

# -----------------------PERIODICA ---------- 
#Schivenoglia.imputed$EM_nox_sum <- Schivenoglia$EM_nox_sum
#ggplot_na_distribution(Schivenoglia.imputed$EM_nox_sum)

# -----------------------PERIODICA ---------- 
#Schivenoglia.imputed$EM_so2_sum <- Schivenoglia$EM_so2_sum
#na_kalman(Schivenoglia.imputed$EM_nh3_livestock_mm)
#ggplot_na_distribution(Schivenoglia.imputed$EM_so2_sum)

# Non sono presenti NaN values
Schivenoglia.imputed$LI_pigs <- Schivenoglia$LI_pigs
ggplot_na_distribution(Schivenoglia.imputed$LI_pigs)

# Non sono presenti NaN values
Schivenoglia.imputed$LI_pigs_v2 <- Schivenoglia$LI_pigs_v2
ggplot_na_distribution(Schivenoglia.imputed$LI_pigs_v2)

# Non sono presenti NaN values
Schivenoglia.imputed$LI_bovine <- Schivenoglia$LI_bovine
ggplot_na_distribution(Schivenoglia.imputed$LI_bovine)

# Non sono presenti NaN values
Schivenoglia.imputed$LI_bovine_v2 <- Schivenoglia$LI_bovine_v2
ggplot_na_distribution(Schivenoglia.imputed$LI_bovine_v2)

# Non sono presenti NaN values
Schivenoglia.imputed$LA_hvi <- Schivenoglia$LA_hvi
ggplot_na_distribution(Schivenoglia.imputed$LA_hvi)

# Non sono presenti NaN values
Schivenoglia.imputed$LA_lvi <- Schivenoglia$LA_lvi
ggplot_na_distribution(Schivenoglia.imputed$LA_lvi)

#Schivenoglia.imputed$LA_land_use <- Schivenoglia$LA_land_use

#Schivenoglia.imputed$LA_soil_use <- Schivenoglia$LA_soil_use

#par(mfrow = c(3,1))
#plot(Schivenoglia$AQ_nh3, type='l')
#plot(cleaned_data, type='l', ylim =c(0,42))
#plot(y,type='l', ylim=c(0,42))
#ggplot_na_distribution(cleaned_data)
#ggplot_na_distribution(y)


#par(mfrow = c(2,1))
#hist(y,breaks = seq(0,50, by=1))
#hist(cleaned_data,seq(0,50, by=1) )

Schivenoglia.imputed_train = {}
Schivenoglia.imputed_validation = {}

for (element in names(Schivenoglia.imputed)){
  Schivenoglia.imputed_train[[element]] = Schivenoglia.imputed[[element]][subset_train]
  Schivenoglia.imputed_validation[[element]] = Schivenoglia.imputed[[element]][subset_validation]
}



# Split trend in 80% for the train and 20 % for the validation

training_data <- lapply(Schivenoglia.imputed[subset_train], unlist)

test_data <- y[subset_validation]

# regression model

model <- lm(AQ_nh3 ~ ., data=Schivenoglia.imputed_train)


backward <- step(model, direction='backward', scope=formula(model), trace=1)
plot(backward$residuals, type='l')
checkresiduals(backward$residuals)
backward$anova
forecast_value_reg <- predict(model,newdata = Schivenoglia.imputed_validation)


par(mfrow = c(2,1))
plot(y, type='l')
plot(model$fitted.values, type='l')
summary(model)
par(mfrow = c(1,1))
plot(backward$residuals, type='l')

auto.arima(backward$residuals)

checkresiduals(backward$residuals) # Ljung Box test 
jarque.bera.test(backward$residuals)
Box.test(backward$residuals, lag = 7, type = "Ljung") # H0 incorrelazione

result <- list()
arima_model <- list()

# ARIMA 1 0 1
fit1 = arima(backward$residuals, order = c(1,0,1))
checkresiduals(fit1$residuals)
Box.test(fit$residuals, lag = 7, type = "Ljung")
jarque.bera.test(fit1$residuals[2:1461])
ggplot_na_distribution(fit1$residuals)

# ARIMA 1 0 1 forecast
forecast_value_arma <- predict(fit, n.ahead=trend_validation_size+1)$pred
result[[1]] <- forecast_value_arma + forecast_value_reg
par(mfrow = c(3,1))
plot(Schivenoglia.imputed$AQ_nh3[subset_validation],type='l')
plot(forecast_value_reg, type='l')
plot(forecast_value_arma, type='l')

# ARIMA 1 0 2 
fit2 = arima(backward$residuals, order = c(1,0,2))
checkresiduals(fit2$residuals)
Box.test(fit2$residuals, lag = 7, type = "Ljung")
jarque.bera.test(fit1$residuals[2:1461])
ggplot_na_distribution(fit1$residuals)

# ARIMA 1 0 2 forecast
forecast_value_arma <- predict(fit2, n.ahead=trend_validation_size+1)$pred
result[[2]] <- forecast_value_arma + forecast_value_reg
par(mfrow = c(3,1))
plot(Schivenoglia.imputed$AQ_nh3[subset_validation],type='l')
plot(forecast_value_reg, type='l')
plot(forecast_value_arma, type='l')

# ARIMA 2 0 1 
fit21 = arima(backward$residuals, order = c(2,0,1))
checkresiduals(fit21$residuals)
Box.test(fit21$residuals, lag = 7, type = "Ljung")
jarque.bera.test(fit21$residuals[2:1461])
ggplot_na_distribution(fit21$residuals)

# ARIMA 2 0 1 forecast
forecast_value_arma <- predict(fit21, n.ahead=trend_validation_size+1)$pred
result[[3]] <- forecast_value_arma + forecast_value_reg
par(mfrow = c(3,1))
plot(Schivenoglia.imputed$AQ_nh3[subset_validation],type='l')
plot(forecast_value_reg, type='l')
plot(forecast_value_arma, type='l')

# ARIMA 2 0 2 
fit22 = arima(backward$residuals, order = c(2,0,2))
checkresiduals(fit22$residuals)
Box.test(fit22$residuals, lag = 7, type = "Ljung")
jarque.bera.test(fit22$residuals[2:1461])
ggplot_na_distribution(fit22$residuals)

# ARIMA 2 0 2 forecast
forecast_value_arma <- predict(fit22, n.ahead=trend_validation_size+1)$pred
result[[4]] <- forecast_value_arma + forecast_value_reg
par(mfrow = c(3,1))
plot(Schivenoglia.imputed$AQ_nh3[subset_validation],type='l')
plot(forecast_value_reg, type='l')
plot(forecast_value_arma, type='l')

# ARIMA 2 0 3
fit23 = arima(backward$residuals, order = c(2,0,2))
checkresiduals(fit22$residuals)
Box.test(fit22$residuals, lag = 7, type = "Ljung")
jarque.bera.test(fit22$residuals[2:1461])
ggplot_na_distribution(fit22$residuals)

# ARIMA 2 0 3 forecast
forecast_value_arma <- predict(fit22, n.ahead=trend_validation_size+1)$pred
result[[4]] <- forecast_value_arma + forecast_value_reg
par(mfrow = c(3,1))
plot(Schivenoglia.imputed$AQ_nh3[subset_validation],type='l')
plot(forecast_value_reg, type='l')
plot(forecast_value_arma, type='l')

# ARIMA 3 0 1 
fit31 = arima(backward$residuals, order = c(3,0,1))
checkresiduals(fit31$residuals)
Box.test(fit31$residuals, lag = 7, type = "Ljung")
jarque.bera.test(fit31$residuals[2:1461])
ggplot_na_distribution(fit31$residuals)

# ARIMA 3 0 1 forecast
forecast_value_arma <- predict(fit31, n.ahead=trend_validation_size+1)$pred
result[[5]] <- forecast_value_arma + forecast_value_reg
par(mfrow = c(3,1))
plot(Schivenoglia.imputed$AQ_nh3[subset_validation],type='l')
plot(forecast_value_reg, type='l')
plot(forecast_value_arma, type='l')


auto.arima(backward$residuals)

for(i in seq_along(result)){
  errore_quadratico <- (Schivenoglia.imputed_validation$AQ_nh3 - result[[i]])^2
  print(sqrt(mean(errore_quadratico)))
}

save(y, file = "response_variable.RData")
save(Schivenoglia.imputed, file = "covariate_variables.RData")