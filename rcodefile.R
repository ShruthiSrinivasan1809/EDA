getwd()
setwd("C:/U")
data<-read.csv("house-data.csv")
summary(data)
ncol(data) #Total of 51 columns
nrow(data) #Total of 1460 rows
names(which(colSums(is.na(data))>0)) #displays the column names which has missing values
colSums(is.na(data)) #displays the number of missing values for a column
colSums(data==0) #displays the number of zero values for a column


library(ggplot2)
ggplot(data = data, aes(x = SalePrice)) +
  geom_density(fill = "black", alpha = 0.5) +  # Adjust fill and transparency
  labs(title = "Visual representation of SalePrice", x = "SalePrice", y = "Density")



boxplot(data$LotFrontage,main="Box plot for Lotfrontage to identify outliers")
#From the boxplot it is evident that the value 313 is an outlier, 
#Confirming the outlier through grubbs test.

library(outliers)
grubbs.test(data$LotFrontage) #we have identified that 313 is an outlier

boxplot(data$MasVnrArea,main="Box plot for MasvnArea to identify outliers") #It is evident that the highest values are outliers

x=rnorm(10)
grubbs.test(x,type=20) #It is confirmed from the grubss test that the highest values are outliers

library(tidyr)

library(dplyr)
library(Hmisc)




#Median imputing the columns
data$LotFrontage=impute(data$LotFrontage,median)
data$MasVnrArea=impute(data$MasVnrArea,median)

data$LotFrontage
data$MasVnrArea

data
#converting NA values to reasonable values

data <- data %>% mutate_if(is.character, ~replace_na(.,"Not Available"))
#addig the column depending on the overall condition
data$House_Qual <- ifelse(data$OverallCond %in% 1:3, "Poor", 
                          ifelse(data$OverallCond %in% 4:6, "Average", 
                                 ifelse(data$OverallCond %in% 7:10, "Good", NA)))



#performing encoding
columns = c("Street","Alley","Utilities","LotConfig","Neighborhood","Condition1",
            "Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st",
            "ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","TotalBsmtSF","Heating",
            "KitchenQual","Functional","GarageType","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature",
            "SaleType","SaleCondition")

encode <- as.formula(paste0("~",paste(columns,collapse="+")))

library(lattice)
library(caret)
library(dplyr)
encoded_data<-predict(dummyVars(encode,data),newdata = data)
encoded_data

#creating new_data as clean with imputed and encoded variables.
new_data <- data %>% select(-c(Street,Alley,Utilities,LotConfig,Neighborhood,Condition1,
                               Condition2,BldgType,HouseStyle,RoofStyle,RoofMatl,Exterior1st,
                               ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,TotalBsmtSF,Heating,
                               KitchenQual,Functional,GarageType,GarageCond,PavedDrive,PoolQC,Fence,MiscFeature,
                               SaleType,SaleCondition))

new_data <-cbind(new_data,encoded_data)
colSums(is.na(new_data))

new_data$LotFrontage



#performing correlation analysis before feature selection to drop out non significant columns

cor_matrix <-cor(new_data[ , -which(names(new_data) == "House_Qual")])
cor_matrix

# Modify correlation matrix
cor_matrix[upper.tri(cor_matrix)] <- 0
diag(cor_matrix) <- 0
cor_matrix

data_new <- new_data[ , -which(names(new_data) == "House_Qual")][ , !apply(cor_matrix, 2, function(x) any(abs(x) > 0.95,na.rm =TRUE))]
head(data_new)
ncol(data_new)




House_quality <- new_data$House_Qual
data_new=cbind(data_new,House_quality)
data_new


colSums(is.na(data_new))


#feature selection


# Load required libraries
library(MASS)
library(nnet)
data_new =na.omit(data_new)
data_new

colSums(is.na(data_new))
#splitting data to train adn test

train_id <- sample(nrow(data_new), 0.8 * nrow(data_new))
train_data <- data_new[train_id,]
test_data <- data_new[-train_id,]

dim(train_data)
dim(test_data)

colSums(is.na(train_data))

library(randomForest)

train_data$House_quality <- as.factor(train_data$House_quality)
# Use the random forest algorithm for feature selection
rf_model <- randomForest(train_data[, -which(names(data_new) == "House_quality")], train_data$House_quality, importance=TRUE, ntree=500)
importance <- importance(rf_model)
importance
varImpPlot(rf_model)

# Use caret package to select the top features
top_features <- varImp(rf_model, useModel=FALSE)
top_features <- rownames(top_features)[order(top_features[,1], decreasing=TRUE)][1:20]

top_features





#new df with the resultant features
df <- data_new %>% 
  dplyr::select( OverallCond, OverallQual, YearBuilt,KitchenQualGd,GrLivArea,TotalBsmtSF,SalePrice, House_quality
          )

df
set.seed(1234)

#Splitting the data
library(caret)
predictors= df %>% dplyr::select(-House_quality)
predictors
target =df$House_quality
target

# Assuming your predictor variables are in a data frame called "predictors" 
# and your target variable is in a vector called "target"

# Split the data into training and testing sets with 80% in the training set
library(e1071)
library(caTools)
split <- sample.split(target, SplitRatio = 0.8)

set.seed(1234)

train_data <- df[split, ]
test_data <- df[!split, ]

dim(train_data)
train_data
dim(test_data)
test_data


#Implementing Task 2

#Peforming multinominal logistic regression

train_data$House_quality 


# fit the multinomial logistic regression model
mlog_model <- multinom(train_data$House_quality ~ .,data = train_data[, -which(names(train_data) == "House_quality")])
summary(mlog_model)




mlog_predictions <- predict(mlog_model, newdata=test_data)
print(mlog_predictions)

confusion_matrix_mlog <- table( test_data$House_quality, mlog_predictions)
print(confusion_matrix_mlog)

accuracy_mlog <- mean(mlog_predictions == train_data$House_quality)
print(paste0("Accuracy: ", accuracy_mlog))

#performing naive bayes classification technique


NB <- naiveBayes(House_quality ~ ., data = train_data)
summary(NB)

predictions_NB <- predict(NB, newdata = test_data)

# Confusion Matrix
confusion_matrix_NB <- table(test_data$House_quality, predictions_NB)
confusion_matrix_NB

accuracy_NB <- mean(predictions == test_data$House_quality)
print(paste0("Accuracy: ", accuracy_NB))

#Task 3
df <- df [, -which(names(df) == "House_quality")]
df

split <- sample.split(df$SalePrice, SplitRatio = 0.8)

train_data <- df[split, ]
test_data <- df[!split, ]

dim(train_data)
train_data
dim(test_data)
test_data

#Performing RandomForest  Regression
library(randomForest)

RandomForest_model <- randomForest(SalePrice~ OverallQual+GrLivArea+TotalBsmtSF+OverallCond+YearBuilt+KitchenQualGd , data = train_data, method = "rf", metric="RMSE",ntree = 100,verbose = TRUE)
summary(RandomForest_model)

prediction_random<-predict(RandomForest_model,test_data)
prediction_random

#Calculating Root Mean Squared Error
RMSE_score_random<-RMSE(prediction_random,test_data$SalePrice)
RMSE_score_random

#Calculating R2_Score
R2_score_random<-R2(prediction_random,test_data$SalePrice)
R2_score_random


#Performing Decision tree regression model

library(rpart)
tree_model <- rpart(SalePrice~  OverallQual+GrLivArea+TotalBsmtSF+OverallCond+YearBuilt+KitchenQualGd , data = train_data)
summary(tree_model)

prediction_dtree<-predict(tree_model,test_data)
prediction_dtree


RMSE_score_dtree<-RMSE(prediction_dtree,test_data$SalePrice)
RMSE_score_dtree

R2_Score_dtree<-R2(prediction_dtree,test_data$SalePrice)
R2_Score_dtree


#Implementing re-sampling methods
# Cross Validation For RandomForest Preditor
library(ipred)
errorest(SalePrice ~ OverallQual+GrLivArea+TotalBsmtSF+OverallCond+YearBuilt+KitchenQualGd,
         data = test_data, model = randomForest, predict = prediction_random,estimator = "cv", est.para = control.errorest(k = 10))

#BootStrapping for DecisionTree Predictor
errorest(SalePrice ~ OverallQual+GrLivArea+TotalBsmtSF+OverallCond+YearBuilt+KitchenQualGd,
         data = test_data, model = rpart, predict = prediction_dtree,estimator = "boot")



