
library(ggplot2)
library(rsample)
library(rpart)
library(caTools)
library(dplyr)
library(TTR)
library(recipes)
library(glmnet)
library(caret)
library(vip)
library(earth)



setwd("E:/Pub/Use of MLMs/Data")

procd_data <- read.csv("final_processed_data.csv")

procd_data <- procd_data[,-c(1,2,4,5,6,11,12,19)]

hyper_grid <- expand.grid(
  
  degree = 1:3,
  nprune = seq(2,100, length.out = 10) %>% floor()
)


##############################

# Return prediction - 3 days

pr3_day <- procd_data[,-c(16:18)]




# data split (train - 80%; test - 20%)

row_index <- sample(nrow(pr3_day), size = 0.8*nrow(pr3_day), replace = F )

train_data3 <- pr3_day[row_index,]

test_data3 <- pr3_day[-row_index,]

test_data3_final <- as.matrix(test_data3[,-15])


# Regularized regression

x3 <- as.matrix(train_data3[,-15])

y <- as.double(train_data3$pr3)


ridge_train3 <- cv.glmnet(x = x3, y = y, alpha = 0)




ridge_pred_train3 <- predict(ridge_train3, x3)



ridge_rmse_train3 <- RMSE(ridge_pred_train3, y)

ridge_rmse_train3


ridge_pred_test3 <- predict(ridge_train3, test_data3_final)

ridge_rmse_test3 <- RMSE(ridge_pred_test3, test_data3$pr3)

ridge_rmse_test3

vip(ridge_train3)

##############################

# LASSO

lasso_train3 <- cv.glmnet(
  
  x = x3,
  
  y = y ,
  
  alpha = 1
  
)

lasso_predict_train3 <- predict(lasso_train3, x3)

lasso_rmse_train3 <- RMSE(lasso_predict_train3, y)

lasso_rmse_train3



lasso_predict_test3 <- predict(lasso_train3, test_data3_final)


lasso_rmse_test3 <- RMSE(lasso_predict_test3, test_data3$pr3)

lasso_rmse_test3

vip(lasso_train3)


# MARS (Adaptive Regression) 

# 3 day periodic return prediction





cv_mars_3 <- train(
  
  x = subset(train_data3, select = -pr3),   
  
  y = train_data3$pr3,
  
  method = "earth",
  
  metric = "RMSE",
  
  trControl = trainControl(method = "cv", number = 10),
  
  tuneGrid = hyper_grid)

summary(cv_mars_3)

ar_train_predic3 <- predict(cv_mars_3, train_data3)

ar_input_train_rmse3 <- RMSE(ar_train_predic3, train_data3$pr3)

ar_input_train_rmse3



ar_test_predict3 <- predict(cv_mars_3, test_data3)

ar_input_test_rmse3 <- RMSE(ar_test_predict3, test_data3$pr3)

ar_input_test_rmse3

vip(cv_mars_3)

##########################################################

# Return Prediction - 5 days


pr5 <- procd_data[,-c(15,17,18)]

row_index5 <- sample(nrow(pr5), size = 0.8*nrow(pr5), replace = F)

train_data5 <- pr5[row_index5,]

test_data5 <- pr5[-row_index5,]

test_data5_final <- as.matrix(test_data5[,-1])


# Regularized regression

x5 <- as.matrix(train_data5[,-15])

y <- as.double(train_data5$pr5)


ridge_train5 <- cv.glmnet(x = x5, y = y, alpha = 0)




ridge_pred_train5 <- predict(ridge_train5, x5)



ridge_rmse_train5 <- RMSE(ridge_pred_train5, y)

ridge_rmse_train5


ridge_pred_test5 <- predict(ridge_train5, test_data5_final)

ridge_rmse_test5 <- RMSE(ridge_pred_test5, test_data5$pr5)

ridge_rmse_test5

vip(ridge_train5)

##############################################

# LASSO

lasso_train5 <- cv.glmnet(
  
  x = x5,
  
  y = y ,
  
  alpha = 1
  
)

lasso_predict_train5 <- predict(lasso_train5, x5)

lasso_rmse_train5 <- RMSE(lasso_predict_train5, y)

lasso_rmse_train5



lasso_predict_test5 <- predict(lasso_train5, test_data5_final)


lasso_rmse_test5 <- RMSE(lasso_predict_test5, test_data5$pr5)

lasso_rmse_test5

vip(lasso_train5)

###############################################

# MARS 

cv_mars_5 <- train(
  
  x = subset(train_data5, select = -pr5),   
  
  y = train_data5$pr5,
  
  method = "earth",
  
  metric = "RMSE",
  
  trControl = trainControl(method = "cv", number = 10),
  
  tuneGrid = hyper_grid)

summary(cv_mars_5)

ar_train_predic5 <- predict(cv_mars_5, train_data5)

ar_input_train_rmse5 <- RMSE(ar_train_predic5, train_data5$pr5)

ar_input_train_rmse5



ar_test_predict5 <- predict(cv_mars_5, test_data5)

ar_input_test_rmse5 <- RMSE(ar_test_predict5, test_data5$pr5)

ar_input_test_rmse5

vip(cv_mars_5)



