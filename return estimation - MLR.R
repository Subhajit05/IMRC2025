







# Return estimation - 3 days -  full data

setwd("E:/Pub/ML - Trade/Data")

procd_data <- read.csv("final_processed_data.csv")

dir_day3 <- procd_data[,-c(1,2,11,12,19,23,24,26:31)]


library("readxl")
nifty <- read_xlsx("NIFTY50.xlsx")


predictors <- dir_day3[,-18]

mlr3_predict_prob <- predict(mlr3, as.matrix(predictors), type = "response")

mlr3_predict_prob <- as.data.frame(mlr3_predict_prob)

nifty_close <- nifty$Close[-(1:33)]


return_day3 <- as.data.frame(cbind(nifty_close,dir = dir_day3$dr3, mlr3_predict_prob))

file.create("return day3_MLR.csv")

write.csv(return_day3, "return day3_MLR.csv")













# Return Estimation - 5 days

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

# Trading recommendation (3 day) - MLR - full data


library("readxl")

nifty <- read_xlsx("NIFTY50.xlsx")

colnames(dir_day5)

predictors <- dir_day5[,-16]

mlr3_predict_prob <- as.data.frame(round(predict(mlr5, as.matrix(predictors), type = "response"),2))


nifty_close <- nifty$Close[-c(1:33)]


return_day5 <- as.data.frame(cbind(nifty_close,dir = dir_day5$dr5, mlr3_predict_prob))

file.create("return day5_MLR.csv")

write.csv(return_day5, "return day5_MLR.csv")

