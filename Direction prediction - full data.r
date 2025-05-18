

library(ggplot2)
library(rsample)
library(rpart)
library(caTools)
library(dplyr)
library(recipes)
library(glmnet)
library(caret)
library(vip)
library(keras)
library(kerastuneR)
library(nnet)



##################################################


# Multinomial Logistic Regression

# SVM

# Decision Tree

# Naive Bayes

# RF

# ANN



############################


setwd("E:/Pub/Use of MLMs/Data")

procd_data <- read.csv("final_processed_data.csv")

dir_day3 <- procd_data[,-c(1,2,11,12,19,23,24,26:31)]

#######################

# Multinomial Logistic Regression


# Direction Prediction - 3 days


# data split (train - 80%; test - 20%)

row_index <- sample(nrow(dir_day3), size = 0.8*nrow(dir_day3), replace = F )

train_data3 <- dir_day3[row_index,]

test_data3 <- dir_day3[-row_index,]

test_data3_final <- test_data3[-1]

x3 <- as.matrix(train_data3[,-1])
y <- factor(train_data3$dr3)


mlr3 <- cv.glmnet(x3, y , family = "multinomial")

mlr3_predict_train <- predict(mlr3, x3, type = "class")

train_cmat_mlr3 <- confusionMatrix(as.factor(mlr3_predict_train), as.factor(train_data3$dr3))

train_cmat_mlr3

mlr3_predict_test <- predict(mlr3, as.matrix(test_data3_final), type = "class")

test_cmat_mlr3 <- confusionMatrix(as.factor(mlr3_predict_test), as.factor(test_data3$dr3))

test_cmat_mlr3

combined_cmat_mlr <- rbind (train_cmat_mlr3$byClass, test_cmat_mlr3$byClass)

file.create("Confusion Matrix mlr.csv")

write.csv(combined_cmat_mlr, "Confusion Matrix mlr.csv")



###########################################

# SVM

library(e1071)
library(caret)

# Direction Prediction - 3 days


svm_row_index3 <- sample(nrow(dir_day3), size = 0.8*nrow(dir_day3), replace = F )

train_data3 <- dir_day3[svm_row_index3,]

test_data3 <- dir_day3[-svm_row_index3,]

test_data3_final <- test_data3[-1]

svm3 <- tune(svm,train.x = factor(dr3) ~ .,
                 data = train_data3,
                 tunecontrol = tune.control(cross = 10))
  

svm3_best <- svm3$best.model


svm3_predict_train <- predict(svm3_best,train_data3, type = "class" )

train_cmat_svm3 <- confusionMatrix(as.factor(svm3_predict_train), 
                                   as.factor(train_data3$dr3))

train_cmat_svm3

svm3_predict_test <- predict(svm3_best, test_data3, type="class")


test_cmat_svm3 <- confusionMatrix(as.factor(svm3_predict_test), 
                                  as.factor(test_data3_final$dr3))



test_cmat_svm3

combined_cmat_svm3 <- rbind(train_cmat_svm3$byClass, test_cmat_svm3$byClass)



file.create("Confusion Matrix svm3.csv")

write.csv(combined_cmat_svm3, "Confusion Matrix svm3.csv")

##############################

# Decision Tree

library(rpart)
library(rpart.plot)

# # Direction Prediction - 3 days

dt_row_index3 <- sample(nrow(dir_day3), size = 0.8*nrow(dir_day3), replace = F )

train_data3 <- dir_day3[dt_row_index3,]

test_data3 <- dir_day3[-dt_row_index3,]

test_data3_final <- test_data3[,-1]


dt3 <- train(factor(dr3) ~ .,
                 data = train_data3,
                 method = "rpart",
                 na.action = na.exclude,
                 trControl = trainControl(method = "cv", number = 10))

dt3_predict_train <- predict(dt3,train_data3, type = "raw")
                 

train_cmat_dt3 <- confusionMatrix(dt3_predict_train, factor(train_data3$dr3))    

train_cmat_dt3

dt3_predict_test <- predict(dt3, test_data3)


test_cmat_dt3 <- confusionMatrix(as.factor(dt3_predict_test), 
                                 as.factor(test_data3$dr3))

test_cmat_dt3

combined_cmat_dt3 <- rbind(train_cmat_dt3$byClass, test_cmat_dt3$byClass)

file.create("Confusion Matrix dt.csv")

write.csv(combined_cmat_dt3, "Confusion Matrix dt.csv")

######################

# Naive Bayes

library(e1071)


# Direction Prediction - 3 days

dir_day3$dr3 <- as.factor(dir_day3$dr3)


# data split (train - 80%; test - 20%)

nb_row_index <- sample(nrow(dir_day3), size = 0.8*nrow(dir_day3), replace = F )

train_data3 <- dir_day3[nb_row_index,]

predictors <- train_data3[,-18]

predictors <- as.matrix(predictors, ncol = 17)

test_data3 <- dir_day3[- nb_row_index,]

test_data3.1 <- as.matrix(test_data3[,-18])

test_data3_final <- as.matrix(test_data3, ncol = 17)


nb3 <- naiveBayes(x = predictors,y = as.factor(train_data3$dr3))


cmat_train-nb3 <- confusionMatrix()

nb3_predict_train <-  predict(nb3, predictors, type = "class")

cmat_train_nb3 <- confusionMatrix(nb3_predict_train, 
                                  as.factor(train_data3$dr3))

cmat_train_nb3



nb3_predict_test <- predict(nb3,test_data3_final, type = "class")

                                               
cmat_test_nb3 <- confusionMatrix(as.factor(nb3_predict_test), 
                                  as.factor(test_data3$dr3))

cmat_test_nb3




# 





###############

# Random Forest

library(ranger)




# number of features
# number of variable to split at each node is the square root of number of features
# This is for classification problem

n_features <- length(setdiff(colnames(dir_day3), "dr3"))


# hyper_grid

hyper_grid_rf <- expand.grid( 
  mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5, 10),
  replace = c(TRUE, FALSE),
  sample.fraction = c(.5, .63, .8),
  rmse = NA
)


rf_row_index3 <- sample(nrow(dir_day3), size = 0.8*nrow(dir_day3), replace = F )

train_data3 <- dir_day3[rf_row_index3,]

test_data3 <- dir_day3[-rf_row_index3,]


for (i in seq_len(nrow(hyper_grid_rf))){
  
  
  dr3_rf <- ranger(
    
    formula = factor(dr3) ~ .,
    
    data = train_data3,
    
    num.trees = n_features * 10,
    
    mtry = hyper_grid_rf$mtry[i],
    
    min.node.size = hyper_grid_rf$min.node.size[i],
    
    replace = hyper_grid_rf$replace[i],
    
    sample.fraction = hyper_grid_rf$sample.fraction[i],
    
    respect.unordered.factors = 'order')
  
  hyper_grid_rf$rmse[i] <- sqrt(dr3_rf$prediction.error)
    
  
}

rf_model_error<-   hyper_grid_rf %>% arrange(rmse)

rf_model_error

# Least Error Model : mtry - 6, min.node.size = 5, replace=False, sample.fraction = 0.63, rmse = 0.4685349

library(randomForest)

  df3_rf1 <- randomForest(
    formula = as.factor(dr3) ~ . ,
    data = train_data3, 
    
    mtry = 6,
    nodesize = 5,
    maxnodes = 500,
    replace = FALSE,
    samplesize = 0.63
    
    
  )
  
rf3_predict_train <- predict(df3_rf1, train_data3, type ="class")

train_cmat_rf3 <- confusionMatrix(as.factor(rf3_predict_train), 
                                  as.factor(train_data3$dr3))

train_cmat_rf3



rf3_predict_test <- predict(df3_rf1, test_data3, type = "class")

train_cmat_rf3 <- confusionMatrix(as.factor(rf3_predict_test), 
                                  as.factor(test_data3$dr3))

train_cmat_rf3


combined_cmat_rf3 <- rbind(train_cmat_rf3$byClass, test_cmat_rf3$byClass)

file.create("Confusion Matrix RF3.csv")

write.csv(combined_cmat_dt3, "Confusion Matrix RF3.csv")

##############################################


# MLP / ANN
library(tensorflow)
library(keras)
library(kerastuneR)
library(tfruns)
library(dplyr)
library(reticulate)


setwd("E:/Pub/Use of MLMs/Data")

procd_data <- read.csv("final_processed_data.csv")

dir_day3 <- procd_data[,-c(1,2,11,12,19,23,24,26:31)]



ann_row_index3 <- sample(nrow(dir_day3), size = 0.8*nrow(dir_day3), replace = F )

train_data3 <- dir_day3[ann_row_index3,]

predictors <- train_data3[, -18]

test_data3 <- dir_day3[- ann_row_index3,]

test_data3_final <- test_data3[,-1]

ann3 <- keras_model_sequential() %>%
  layer_dense(units = 128,activation = "relu" ,input_shape = 17) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10 , activation = "softmax") %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_rmsprop(),
metrics = c("accuracy"))


summary(ann3)


fit1 <- ann3 %>% 
  fit(x = predictors,
      y = train_data3$dr3,
      epochs = 25,
      validaiton_split = 0.02,
      batch_size = 128
      )


####################

# 5 day ahead prediction

# Multinomial Regression

# Direction Prediction - 5 days

setwd("E:/Pub/Use of MLMs/Data")

procd_data <- read.csv("final_processed_data.csv")

dir_day5 <- procd_data[,-c(1,2,4,5,11,12,19,23:25,27:31)]


row_index5 <- sample(nrow(dir_day5), size = 0.8*nrow(dir_day5), replace = F)

train_data5 <- dir_day5[row_index5,]

test_data5 <- dir_day5[-row_index5,]

test_data5_final <- test_data5[-1]

x5 <- as.matrix(train_data5[,-1])
y <- factor(train_data5$dr5)



mlr5 <-  cv.glmnet(x5, y , family = "multinomial")


mlr5_predict_train <- predict(mlr5, x5, type = "class")

train_cmat_mlr5 <- confusionMatrix(as.factor(mlr5_predict_train), 
                                   as.factor(train_data5$dr5))

train_cmat_mlr5

mlr5_predict_test <- predict(mlr5, as.matrix(test_data5_final), type = "class")

test_cmat_mlr5 <- confusionMatrix(as.factor(mlr5_predict_test), as.factor(test_data5$dr5))

test_cmat_mlr5

combined_cmat_mlr <- rbind (train_cmat_mlr5$byClass, test_cmat_mlr5$byClass)

file.create("Confusion Matrix mlr5.csv")

write.csv(combined_cmat_mlr, "Confusion Matrix mlr5.csv")


#################################

# SVM

# Direction Prediction - 5 days

library(e1071)

setwd("E:/Pub/Use of MLMs/Data")

procd_data <- read.csv("final_processed_data.csv")

dir_day5 <- procd_data[,-c(1,2,4,5,11,12,19,23:25,27:31)]


svm_row_index5 <- sample(nrow(dir_day5), size = 0.8*nrow(dir_day5), replace = F)

train_data5 <- dir_day5[svm_row_index5,]

test_data5 <- dir_day5[- svm_row_index5,]

test_data5_final <- test_data5[-1]


svm5 <- tune(svm, factor(dr5) ~ ., data = train_data5, 
                 tunecontrol = tune.control(cross = 10))

svm_dir5_best <- svm5$best.model

svm_dir5_best


svm5_predict_train <- predict(svm_dir5_best,train_data5, type ="class")

train_cmat_svm5 <- confusionMatrix(as.factor(svm5_predict_train),
                                   as.factor(train_data5$dr5))

train_cmat_svm5


svm5_predict_test <- predict(svm_dir5_best,test_data5, type ="class")

test_cmat_svm5 <- confusionMatrix(as.factor(svm5_predict_test),
                                   as.factor(test_data5$dr5))

test_cmat_svm5







####################################

# Decision Tree

# Direction Prediction - 5 days



dt_row_index5 <- sample(nrow(dir_day5), size = 0.8*nrow(dir_day5), replace = F)


train_data5 <- dir_day5[dt_row_index5,]

test_data5 <- dir_day5[- dt_row_index5,]

test_data5_final <- test_data5[-1]



dt5 <- train(factor(dr5) ~ .,
                 data = train_data5,
                 method = "rpart",
                 na.action = na.exclude,
                 trControl = trainControl(method = "cv", number = 10))

dt5_predict_train <- predict(dt5,train_data5, type = "raw")


train_confMat_dt5 <- confusionMatrix(dt5_predict_train, 
                               factor(train_data5$dr5))    

train_confMat_dt5


dt5_predict_test <- predict(dt5,test_data5, type = "raw")


test_confMat_dt5 <- confusionMatrix(dt5_predict_test, 
                                     factor(test_data5$dr5))    

test_confMat_dt5

combined_cmat_dt5 <- rbind (train_confMat_dt5$byClass, 
                            test_confMat_dt5$byClass)

file.create("Confusion Matrix dt5.csv")

write.csv(combined_cmat_dt5, "Confusion Matrix dt5.csv")






################################################

# Direction Prediction - 5 days

# Random Forest

library(ranger)
library(randomForest)
library(dplyr)

setwd("E:/Pub/Use of MLMs/Data")

procd_data <- read.csv("final_processed_data.csv")

dir_day5 <- procd_data[,-c(1,2,4,5,11,12,19,23:25,27:31)]



rf_row_index5 <- sample(nrow(dir_day5), size = 0.8*nrow(dir_day5), replace = F)


train_data5 <- dir_day5[rf_row_index5,]

test_data5 <- dir_day5[- rf_row_index5,]



n_features <- length(setdiff(colnames(dir_day5), "dr5"))


# hyper_grid

hyper_grid_rf <- expand.grid( 
  mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5, 10),
  replace = c(TRUE, FALSE),
  sample.fraction = c(.5, .63, .8),
  rmse = NA
)



for (i in seq_len(nrow(hyper_grid_rf))){
  
  
  dr5_rf <- ranger(
    
    formula = factor(dr5) ~ .,
    
    data = train_data5,
    
    num.trees = n_features * 10,
    
    mtry = hyper_grid_rf$mtry[i],
    
    min.node.size = hyper_grid_rf$min.node.size[i],
    
    replace = hyper_grid_rf$replace[i],
    
    sample.fraction = hyper_grid_rf$sample.fraction[i],
    
    respect.unordered.factors = 'order')
  
  hyper_grid_rf$rmse[i] <- sqrt(dr5_rf$prediction.error)
  
  
}

rf_model_error<-   hyper_grid_rf %>% arrange(rmse)

rf_model_error

# Best model : mtry - 3, min.node.size = 3, replace = T, sample.fraction = 0.63, rmse = 0.479

library(randomForest)

df5_rf1 <- randomForest(
  formula = as.factor(dr5) ~ . ,
  data = train_data5, 
  
  mtry = 3,
  nodesize = 3,
  maxnodes = 5000,
  replace = T,
  samplesize = 0.63
  
)

rf5_predict_train <- predict(df5_rf1, train_data5, type ="class")

train_cmat_rf5 <- confusionMatrix(as.factor(rf5_predict_train), 
                                  as.factor(train_data5$dr5))

train_cmat_rf5



rf5_predict_test <- predict(df5_rf1, test_data5, type = "class")

test_cmat_rf5 <- confusionMatrix(as.factor(rf5_predict_test), 
                                  as.factor(test_data5$dr5))

test_cmat_rf5


####################################################

# 5 day ahead prediction
# Naive Bayes


library(e1071)
library(caret)



# data split (train - 80%; test - 20%)

nb_row_index <- sample(nrow(dir_day5), size = 0.8*nrow(dir_day5), replace = F )

train_data5 <- dir_day5[nb_row_index,]

predictors <- train_data5[,-18]

predictors <- as.matrix(predictors, ncol = 17)

test_data5 <- dir_day5[- nb_row_index,]

test_data5.1 <- as.matrix(test_data5[,-18])

test_data5_final <- as.matrix(test_data5, ncol = 17)


nb5 <- naiveBayes(x = predictors,y = as.factor(train_data5$dr5))



nb5_predict_train <-  predict(nb5, predictors, type = "class")

cmat_train_nb5 <- confusionMatrix(nb5_predict_train, 
                                  as.factor(train_data5$dr5))

cmat_train_nb5



nb5_predict_test <- predict(nb5,test_data5_final, type = "class")


cmat_test_nb5 <- confusionMatrix(as.factor(nb5_predict_test), 
                                 as.factor(test_data5$dr5))

cmat_test_nb5

#################################################

