require(caret)
require(dplyr)

set.seed(3433)

## remove(list=ls())
setwd("C:/Users/timt/Documents/DataScience/PracticalMachineLearning/Project/data")

## Load training data
training <- read.csv("pml-training.csv", header=TRUE, row.names = 1, stringsAsFactors = FALSE, na.strings = c("NA",""))
## Load testing data
testing <- read.csv("pml-testing.csv", header=TRUE, row.names = 1, stringsAsFactors = FALSE, na.strings = c("NA",""))

#############################################################################################
## Random Forest
## Chose random forest since outcome variable is a multiclass (factor) variable

## Cleanup
## Convert "classe" (the outcome variable) to a factor variable in both data sets
training$classe <- as.factor(training$classe)
testing$classe <- as.factor(testing$classe)

ignoreNAs <- apply(!is.na(training),2,sum) >= nrow(training)
dimnames(training[,ignoreNAs])[[2]]


?sum
## Get column names and and ordered by name for easier analysis
## This was used to decide which predictors to include in final training model
colNames <- colnames(training)
ordered.training <- training[,order(colnames(training))]

## After visual analysis of the data, excluded columns composed of mostly NA values
## To avoid creating a massive vector of column names, I used a vector of strings 
## to match to the beginning of the column name in the data set
predictor.variables <- c("^accel", "^gyros", "^magnet", "^pitch", "^roll", "^yaw","^total")
## Match Column names to find all predictor (feature) variables
matches <- unique(grep(paste0(predictor.variables, collapse="|"), colNames, value=TRUE))
## Add back the outcome variable (classe) - this would have been excluded if I were to use PCA for feature selection
matches <- c(matches, "classe")

## Slice training data into training and test subsets
inTrain <- createDataPartition(y=training$classe, p=0.70, list = FALSE)
rawRfTrain <- training[inTrain,]
rawRfTest <- training[-inTrain,]

## Only include selected features/predictors and outcome variable in training and test subsets
rfTrain <- rawRfTrain[,matches]
rfTest <- rawRfTest[,matches]

## Train model using "classe" as outcome and all other variables as predictors
## Used 5-fold Cross Validation to train model (use parallel processing to speed up model training)
modFit <- train(classe ~., data = rfTrain, method="rf", proxy=TRUE, 
                trControl=trainControl(method="cv", number = 5), allowParallel = TRUE)
modFit
print(modFit, digits = 4)
## Side Note: Run time for RF model training was approximately 12 minutes on a 6 core, 3.33 GHz machine with 18 GB memory

## Show model fit output and confusion matrix (how many right vs how many wrong per "classe" using model)
print(modFit$finalModel)

## Prediect new values using test subset that was created from the original training set
pred <- predict(modFit, rfTest)
rfTest$predRight <- pred==rfTest$classe
table(pred, rfTest$classe)


## Out of sample error
## Calculate accuracty by summing correct predtions divided by the total number of predictions
accuracy <- nrow(rfTest[rfTest$predRight,]) / nrow(rfTest)
## Calculate error by substracting error from 1
error <- (1 - accuracy) * 100
error

#############################################################################################
## USING THE ORIGINAL TEST SET
## Apply model to original test set
matches <- unique(grep(paste0(predictor.variables, collapse="|"), colNames, value=TRUE))
testPred <- predict(modFit, testing[,matches])
testingPrediction <- testing

## Convert Factor "classe" to character
testingPrediction$classe <- as.character(testPred)

## Ouput answers to dataframe with classe
answers <- testingPrediction[,c("classe")]

## Function to create output files for each problem_id/classe answer
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)


