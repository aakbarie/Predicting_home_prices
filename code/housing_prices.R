

# load needed libraries -------------------------------------------------------------------------------------------

library(tidyverse)
library(caret)
library(mlbench)
library(reshape2)
library(corrplot)
library(MCMCpack)
library(ggfortify)
# read-in data ----------------------------------------------------------------------------------------------------

rl_stt_data <- read.table("C:/Users/akbar/Documents/R/Predicting_home_prices/data/RealEstateSales.txt",header=T)

index <- createDataPartition(rl_stt_data$price, p = 0.7, list = FALSE)
training_set <- rl_stt_data[index, ]
testing_set <- rl_stt_data[-index, ]


# data exploration ------------------------------------------------------------------------------

# calculate correlation matrix
cor_matrix <- rl_stt_data %>%
     dplyr::select(-price) %>%
     cor(.) %>%
     melt(.)

ggplot(data = cor_matrix, aes(x=Var1, y=Var2, fill=value)) + 
     geom_tile() +
     labs(title = "Correlation Matrix", x = "", y = "", fill = "Correlation \nDistribution\n")

autoplot(prcomp(rl_stt_data[,-1]), data = rl_stt_data, colour = "price") +
     xlab("First Principle Component") +
     ylab("Second Principle Component")
# model selection -------------------------------------------------------------------------------------------------

# forward step-wise regression - variable selection
lm_fit_forward <- train(price ~ ., data = training_set, "lmStepAIC", scope = 
                    list(lower = ~., upper = ~.^2), direction = "forward")
summary(lm_fit_forward)

training_set %>%
     dplyr::select(price) %>%
     bind_cols(residuals = resid(lm_fit_forward)) %>%
     ggplot(aes(x = price, y = residuals)) +
     geom_point() +
     geom_abline(intercept = 0, slope = 0 , col = "red")+
     xlab("Fitted values of Price") +
     ylab("Model Residuals") + 
     ggtitle("Model Diagnostic") +
     theme_bw()

frwrd_pred <- predict(lm_fit_forward, newdata = testing_set[,-1]) 

testing_set %>%
     dplyr::select(price) %>%
     bind_cols(prediction = frwrd_pred) %>%
     ggplot(aes(x = price, y = prediction)) +
     geom_point() +
     geom_abline(intercept = 0, col = "red")+
     xlab("Actual values of Price") +
     ylab("Predicted values of Price") + 
     ggtitle("Prediction Diagnostics") +
     theme_bw()

# forward step-wise regression - variable selection
lm_fit_backward <- train(price ~ ., data = training_set, "lmStepAIC", scope = 
                    list(lower = ~., upper = ~.^2), direction = "backward")
training_set %>%
     dplyr::select(price) %>%
     bind_cols(residuals = resid(lm_fit_backward)) %>%
     ggplot(aes(x = price, y = residuals)) +
     geom_point() +
     geom_abline(intercept = 0, slope = 0 , col = "red")+
     xlab("Fitted values of Price") +
     ylab("Model Residuals") + 
     ggtitle("Model Diagnostic") +
     theme_bw()

backward_pred <- predict(lm_fit_backward, newdata = testing_set[,-1]) 

testing_set %>%
     dplyr::select(price) %>%
     bind_cols(prediction = backward_pred) %>%
     ggplot(aes(x = price, y = prediction)) +
     geom_point() +
     geom_abline(intercept = 0, col = "red")+
     xlab("Actual values of Price") +
     ylab("Predicted values of Price") + 
     ggtitle("Prediction Diagnostics") +
     theme_bw()

# cross-validation 

ctrl<-trainControl(method = "cv", number = 10)

lm_cv_fit <- train(price ~ ., data = rl_stt_data, method = "lm", trControl = ctrl, metric="Rsquared")

summary(lm_cv_fit)

rl_stt_data %>%
     dplyr::select(price) %>%
     bind_cols(residuals = resid(lm_cv_fit)) %>%
     ggplot(aes(x = price, y = residuals)) +
     geom_point() +
     geom_abline(intercept = 0, slope = 0 , col = "red")+
     xlab("Fitted values of Price") +
     ylab("Model Residuals") + 
     ggtitle("Model Diagnostic") +
     theme_bw()

cv_pred <- predict(lm_cv_fit) 

rl_stt_data %>%
     dplyr::select(price) %>%
     bind_cols(prediction = cv_pred) %>%
     ggplot(aes(x = price, y = prediction)) +
     geom_point() +
     geom_abline(intercept = 0, col = "red")+
     xlab("Actual values of Price") +
     ylab("Predicted values of Price") + 
     ggtitle("Prediction Diagnostics") +
     theme_bw()

# going the extra mile

vars <- noquote(paste(prmt[-1], collapse = " + "))
bayes_model_check <- MCMCregress(price ~ area + bed + bath + 
                                      ac + grsize + pool + 
                                      age + qu + style + 
                                      lotsize + hgway + age:qu + 
                                      qu:lotsize + area:bed + 
                                      area:qu + qu:style + grsize:lotsize + 
                                      ac:lotsize + ac:age + age:lotsize + 
                                      bed:style + bed:ac + bed:qu + 
                                      age:style + area:age + bed:bath + 
                                      ac:pool + pool:lotsize + area:style,
                                 data = training_set)

raftery.diag(bayes_model_check)


# machine learning method

metric <- "RMSE"
trainControl <- trainControl(method = "cv", number = 10)

caretGrid <- expand.grid(interaction.depth=c(1, 3, 5), n.trees = (0:50)*50,
                         shrinkage=c(0.01, 0.001),
                         n.minobsinnode=10)

set.seed(99)

gbm_fit <- train(price ~ .
                   , data=training_set
                   , distribution="gaussian"
                   , method="gbm"
                   , trControl=trainControl
                   , verbose=FALSE
                   , tuneGrid=caretGrid
                   , metric=metric
                   , bag.fraction=0.75
)                  

print(gbm.caret)

caret.predict <- predict(gbm.caret, newdata = testing_set, type = "raw")

rmse.caret<-rmse(testing_set$price, caret.predict)
print(rmse.caret)

R2.caret <- cor(gbm.caret$finalModel$fit, training_set$price)^2
print(R2.caret)

testing_set %>%
  dplyr::select(price) %>%
  bind_cols(prediction = caret.predict) %>%
  ggplot(aes(x = price, y = prediction)) +
  geom_point() +
  geom_abline(intercept = 0, col = "red")+
  xlab("Actual values of Price") +
  ylab("Predicted values of Price") + 
  ggtitle("Prediction Diagnostics") +
  theme_bw()



