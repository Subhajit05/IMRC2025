

library(ggplot2)
library(rsample)
library(rpart)
library(caret)
library(caTools)
library(dplyr)
library(TTR)
library(recipes)
library(glmnet)
library(caret)
library(vip)
library(e1071)
library(readxl)

########################################################




setwd("E:/Pub/Use of MLMs/Data")


# Computation of Technical Indicators

library(TTR)
library(readxl)




nifty <- read_xlsx("NIFTY50.xlsx")

nifty_hlc <- data.frame(cbind(nifty$High, nifty$Low, nifty$Close))

nifty_vix <- read.csv("vix.csv")

######################

windows(10,10)

par(mfrow = c(2,1))

plot(nifty$Date,nifty$Close, col ="red", 
     xlab = "Years", ylab = "Nifty Close", type = "l", lwd = 2.5)

plot(nifty$Date,nifty$vix.Close, col = "blue", type = "l", xlab = "Years",
     ylab = "VIX close value")



nifty_returns <- as.data.frame(cbind( ret_3Day = nifty$pr3, 
                                    ret_5Day = nifty$pr5))


boxplot(nifty_returns, main = "Distribution of 3-day & 5-day Return")


#########################################

# Nifty - TI

# Technical Indicators (Estimation period 7 days)

## AD - Chaikin oscillator
## ADX - Average Directional Index
## ADXR - Average Directional Index Rating
## Aroon's Oscillator
## Bollinger Bands (Up, Low)
## ATR - Average True Range
## NATR - Normalized Average True Range
## RSI - Relative Strength Index
## OBV - On Balance Volume
## William's R
## William's AD
## TRIX - Triple Exponential Moving Average



#AD

ad_nifty <- chaikinAD(nifty_hlc, nifty$Shares.Traded)



# ADX

nifty_adx <- ADX(nifty_hlc, n = 7)

# ADXR

adx_niftyclose <- as.data.frame(cbind(adx = nifty_adx,close = nifty$Close))

adx_niftyclose <- na.omit(adx_niftyclose)


nifty_adxr <- 0

for(i in 8:nrow(adx_niftyclose)){
  
  nifty_adxr[i] = (adx_niftyclose$ADX[i] + adx_niftyclose$ADX[i-7])/2
}


# ATR

nifty_atr <- as.data.frame(ATR(nifty_hlc, n = 7))

nifty_atr1 <- nifty_atr$atr


# NATR

nifty_natr <- 0

for(i in 1: length(nifty_atr1)){
  
  nifty_natr[i] <- nifty_atr1[i]/nifty$Close[i] * 100
    
}




# Aroon's Oscillator

nifty_hl <- as.data.frame( cbind(high = nifty$High, low = nifty$Low))
nifty_aroon <- aroon(nifty_hl, n = 7)



# Bollinger Bands


nifty_bb <- as.data.frame(BBands(nifty_hlc, n = 7, sd = 2))

nifty_bb_up <- nifty_bb$up

nifty_bb_dn <- nifty_bb$dn



# OBV

nifty_obv <- OBV(nifty$Close, volume = nifty$Shares.Traded)


# RSI

nifty_rsi <- na.omit(RSI(nifty$Close, n = 7))


# TRIX


nifty_trix <- TRIX(nifty$Close, n = 7)


# Williams R

nifty_WR <- WPR(nifty_hlc, n = 7)

 


nifty_ti <- as.data.frame(cbind(ad_nifty, nifty_adx, nifty_adxr, nifty_atr1, nifty_natr,
        nifty_aroon,nifty_bb_up,nifty_bb_dn ,nifty_obv, nifty_rsi,nifty_trix, 
        nifty_WR, vix_close = nifty_vix$Close, vol = nifty$Turnover....Cr., pr3 = nifty$pr3, 
        pr5 =nifty$pr5, dr3 = nifty$dir3, dr5 = nifty$dir5))



complete_data <- na.omit(nifty_ti)


file.create("complete data.csv")

write.csv(complete_data, "complete data.csv")

comp_data <- read.csv("complete data.csv")

tech_indicators <- comp_data[,-c(22:25)]

scaled_tech_indicators <- scale(tech_indicators)

file.create("final_processed_data.csv")

final_processed_data <- cbind(scaled_tech_indicators, complete_data$pr3,
                        complete_data$pr5, complete_data$dr3,
                        complete_data$dr5)

write.csv(final_processed_data, "final_processed_data.csv")

#######

# Target Variables

# Periodic Return

pd3 <- 0

for(i in 1:nrow(nifty)){
  
  pd3[i] <- (nifty$Close[i+3] - nifty$Close[i])/nifty$Close[i]
  
}

pd5 <- 0


for(i in 1:nrow(nifty)){
  
  pd5[i] <- (nifty$Close[i+5] - nifty$Close[i])/nifty$Close[i]
  
}


nifty_returns <- cbind(nifty)

summary(nifty_returns)

# Prediction of Direction

dir3 <- 0



