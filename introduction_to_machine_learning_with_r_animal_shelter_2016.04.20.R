# Introduction to Machine Learning with R
# Justin Meyer
# 4/20/16

# This example is intended to demonstrate how machine learning works in R.
# It is not intended to arrive at the best predictions possible.

##########
# Get Data
##########

# Data are from https://www.kaggle.com/c/shelter-animal-outcomes
# Data are the training set modified so that all Adoption outcomes are Adopted
# and all other outcomes are Not_Adopted

# I found the data interesting but wanted a binary outcome for an example

# This purpose of this project is to predict the outcome for animals at a shelter
# Make sure this points to wherever you have saved the files on your computer
animals <- read.csv("C:/Users/meyerjm/Documents/animal_shelter_train_test.csv",
                          stringsAsFactors = FALSE) # Avoids factors

##################################
# Summarize Predictors and Outcome
##################################

str(animals)
summary(animals)

# OutcomeType
summary(as.factor(animals$OutcomeType))

library(ggplot2)
ggplot(animals, aes(OutcomeType)) + 
  geom_bar()

# SexuponOutcome
animals$SexuponOutcome <- as.factor(animals$SexuponOutcome)
summary(animals$SexuponOutcome)

ggplot(animals, aes(SexuponOutcome)) + 
  geom_bar()

# AnimalType
animals$AnimalType <- as.factor(animals$AnimalType)
summary(animals$AnimalType)

ggplot(animals, aes(AnimalType)) + 
  geom_bar()

# Breed
summary(as.factor(animals$Breed))

# Color
summary(as.factor(animals$Color))

# AgeuponOutcome
summary(as.factor(animals$AgeuponOutcome))

# Recode all the ages to days so we have a numeric predictor

animals$AgeuponOutcomeNumber <- as.numeric(gsub("([0-9]+).*$", "\\1", animals$AgeuponOutcome))

animals$AgeuponOutcomeDays <- NA

animals$AgeuponOutcomeDays[grepl("(day)|(days)", animals$AgeuponOutcome)] <- 
  animals$AgeuponOutcomeNumber[grepl("(day)|(days)", animals$AgeuponOutcome)]

animals$AgeuponOutcomeDays[grepl("(week)|(weeks)", animals$AgeuponOutcome)] <- 
  animals$AgeuponOutcomeNumber[grepl("(week)|(weeks)", animals$AgeuponOutcome)] * 7

animals$AgeuponOutcomeDays[grepl("(month)|(months)", animals$AgeuponOutcome)] <- 
  animals$AgeuponOutcomeNumber[grepl("(month)|(months)", animals$AgeuponOutcome)] * 31

animals$AgeuponOutcomeDays[grepl("(year)|(years)", animals$AgeuponOutcome)] <- 
  animals$AgeuponOutcomeNumber[grepl("(year)|(years)", animals$AgeuponOutcome)] * 365

summary(animals$AgeuponOutcomeDays)

######################################################
# Relationship of Predictors to Each Other and Outcome
######################################################

### Predictors to each other

# Nearly all cases with unknown sex are cats
# Dogs are more likely to be spayed or neutered
table(animals$SexuponOutcome, animals$AnimalType)
prop.table(table(animals$SexuponOutcome, animals$AnimalType), 1)

# Could look at other predictors if needed

### Predictors to outcome

# Dogs are slightly more likely to be adopted than cats
table(animals$OutcomeType, animals$AnimalType)
temp <- prop.table(table(animals$OutcomeType, animals$AnimalType), 2)
temp

ggplot(as.data.frame(temp), aes(x = factor(Var2), y = Freq * 100, fill = factor(Var1))) +
  geom_bar(stat="identity") +
  labs(x = "AnimalType", y = "Percent", fill = "Outcome")
rm(temp)

# Spayed or neutered animals are much more likely to be adopted
table(animals$OutcomeType, animals$SexuponOutcome)
temp <- prop.table(table(animals$OutcomeType, animals$SexuponOutcome), 2)
temp

ggplot(as.data.frame(temp), aes(x = factor(Var2), y = Freq * 100, fill = factor(Var1))) +
  geom_bar(stat="identity") +
  labs(x = "SexuponOutcome", y = "Percent", fill = "Outcome")

table(animals$OutcomeType, animals$AgeuponOutcomeDays)
temp <- prop.table(table(animals$OutcomeType, animals$AgeuponOutcomeDays), 2)
temp

ggplot(as.data.frame(temp), aes(x = factor(Var2), y = Freq * 100, fill = factor(Var1))) +
  geom_bar(stat="identity") +
  labs(x = "AgeuponOutcomeDays", y = "Percent", fill = "Outcome")
rm(temp)

#########################################
# Feature Selection / Feature Engineering
#########################################

### Remove unneeded predictors
# This is supposed to be a simple example so only use a few predictors

data <- subset(animals, select = c("AnimalType",
                                   "SexuponOutcome", 
                                   "AgeuponOutcomeDays", 
                                   "OutcomeType"))

### Replace spaces in values since these will become part of variable names later
# R doesn't like spaces in variable names
data$SexuponOutcome <- gsub(" ", "_", data$SexuponOutcome)

######################
# Replace Missing Data
######################

# We could remove cases with missing data but we'd like to be able to produce
# predictions for all cases.
summary(data$AnimalType)
summary(as.factor(data$SexuponOutcome))
summary(data$AgeuponOutcomeDays)
summary(as.factor(data$OutcomeType))

# Recode missing sex as unknown
data$SexuponOutcome[data$SexuponOutcome == ""] <- "Unknown"
data$SexuponOutcome <- factor(data$SexuponOutcome) # remove unused factor level

# Replace missing age with median
data$AgeuponOutcomeDays[is.na(data$AgeuponOutcomeDays)] <- 
  median(data$AgeuponOutcomeDays, na.rm = TRUE)

summary(data)

# Some algorithms such as a linear model won't work properly with categorical predictors
# Recode categorical predictors into a binary predictor for each category
# The -1 removes the intercept since we don't want this
temp <- as.data.frame(model.matrix(~ . -1, data = data[c("SexuponOutcome", "AnimalType")]))

# Don't need the SexuponOutcomeUnknown predictor as it is redundant
temp$SexuponOutcomeUnknown <- NULL 

data <- cbind(temp, data)
rm(temp)
data$SexuponOutcome <- NULL
data$AnimalType <- NULL

summary(data)

# Make binary variables factors to keep caret from rescaling them later
data$SexuponOutcomeIntact_Female <- factor(data$SexuponOutcomeIntact_Female)
data$SexuponOutcomeIntact_Male <- factor(data$SexuponOutcomeIntact_Male)
data$SexuponOutcomeNeutered_Male <- factor(data$SexuponOutcomeNeutered_Male)
data$SexuponOutcomeSpayed_Female <- factor(data$SexuponOutcomeSpayed_Female)
data$AnimalTypeDog <- factor(data$AnimalTypeDog)
data$OutcomeType <- factor(data$OutcomeType)

summary(data)

######################################
# Create training and testing datasets
######################################

# Caret is a wrapper that primarily allows the user to access lots of machine 
# learning algorithms from various packages using the same commands
# It also has useful tools for tasks associated with machine learning
library(caret)

# Create training and testing datasets
# The createDataPartition command does a stratified random split of the data
# This is important because it preserves the distribution of the outcome
# For example, if the outcome is distributed 50/50 in the data 
# both the train and test set will have the outcome distributed 50/50.
set.seed(73)
trainIndex <- createDataPartition(data$OutcomeType, 
                                  p = .8, # Proportion in the index
                                  list = FALSE, # Output type
                                  times = 1) # Number of partitions
train <- data[ trainIndex,]
test  <- data[-trainIndex,]
rm(trainIndex)

# The outcomes are distributed equally in both datasets
# The predictors are also similar
summary(train)
summary(test)

######################################
# Center and Scale Numeric Predictors
######################################

# Some algorithms require centering and scaling of numeric variables
# or variables with larger values will have more
# impact than variables with smaller values

# You can do this during the algorithm creation step instead if you prefer

# The preProcess command creates the values that center and scale the data
# It is useful to have these values in case you want to apply them to new data later
# Use only the train data to create the preprocess values.
preprocess_values <- preProcess(train, method = c("center", "scale"))

# The predict command applies the preprocess values to data
train_transformed <- predict(preprocess_values, train)
test_transformed <- predict(preprocess_values, test)

# AgeuponOutcomeDays is now transformed
summary(train_transformed)
summary(test_transformed)

###################
# Check assumptions
###################

# Use the nearZeroVar command to look for variables with zero or near zero variance.
# Zero variance is when a combination of categorical variables 
# (example: native_us = 1, raceBlack = 1, sexMale = 1) doesn't exist.
# Good explanation: https://tgmstat.wordpress.com/2014/03/06/near-zero-variance-predictors

# Including variables that have zero or near zero variance can cause problems for some algorithms.
nearZeroVar(train_transformed, saveMetrics = TRUE) # saveMetrics keeps the results rather than a list of variables
nearZeroVar(test_transformed, saveMetrics = TRUE)

# May also want to remove any correlated variables depending on the algorithm used
# Not showing this step for simplicity/time reasons

########################
# Train Some Algorithms
########################

# Finally, the machine learning part!

### Simplest example
glmFit <- train(OutcomeType ~ ., # OutcomeType is the outcome, everything else is a predictor
                method = 'glm', # Use the glm algorithm
                data = train_transformed) # Use the train dataset
glmFit # Some info about the results of the algorithm
summary(glmFit) # More info
varImp(glmFit) # What predictors are most important?
getTrainPerf(glmFit) # Another way to get accuracy
confusionMatrix(glmFit) # Report the confusion matrix

### More complicated method so that we can find out the area under the ROC curve
# Create a trainControl object
# This isn't required but we want to produce ROC value
ctrl <- trainControl(method = "cv", # Use crossvalidation (break data into multiple folds)
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, # Get class probabilities
                     savePredictions = TRUE) # Save predictions

# Create algorithms using the training data
# All available caret algorithms: https://topepo.github.io/caret/modelList.html

# Generalized linear model
glmFit <- train(OutcomeType ~ .,
                method = 'glm',
                trControl = ctrl, # Use the specifications from the trainControl object
                metric = "ROC", # Use area under the ROC curve to evaluate results
                data = train_transformed)
glmFit
summary(glmFit)
varImp(glmFit)
getTrainPerf(glmFit)
confusionMatrix(glmFit)

# Classification tree
rpartFit <- train(train$OutcomeType ~ ., 
                  method = 'rpart', 
                  trControl = ctrl,
                  metric = "ROC",
                  data = train_transformed)
rpartFit
summary(rpartFit)
varImp(rpartFit)
getTrainPerf(rpartFit)
confusionMatrix(rpartFit)
plot(rpartFit) # Plot ROC by tuning parameters
plot(rpartFit$finalModel) # Create tree plot
text(rpartFit$finalModel) # Add text to tree plot

# Gradient boosted machine
# Slow
gbmFit <- train(OutcomeType ~ .,
                method = 'gbm',
                trControl = ctrl,
                metric = "ROC",
                data = train_transformed)
gbmFit
summary(gbmFit)
varImp(gbmFit)
getTrainPerf(gbmFit)
confusionMatrix(gbmFit)
plot(gbmFit)

# Conditional inference tree
# A more complicated variation on a tree
ctreeFit <- train(OutcomeType ~ .,
                  method = 'ctree', 
                  trControl = ctrl,
                  metric = "ROC",
                  data = train_transformed)
ctreeFit
summary(ctreeFit)
varImp(ctreeFit)
getTrainPerf(ctreeFit)
confusionMatrix(ctreeFit)
plot(ctreeFit)
plot(ctreeFit$finalModel) # Example of difficult to interpret tree plot

# Random forest
# Too slow for an example in a presentation but a commonly used algorithm
# rfFit <- train(OutcomeType ~ .,
#                 method = 'rf',
#                 trControl = ctrl,
#                 metric = "ROC",
#                 data = train_transformed)
# rfFit
# summary(rfFit)
# varImp(rfFit)
# getTrainPerf(rfFit)
# confusionMatrix(rfFit)
# plot(rfFit)
# plot(rfFit$finalModel) # Plot trees

# K nearest neighbor
# Too slow for an example in a presentation but a commonly used algorithm
# knnFit <- train(OutcomeType ~ .,
#                 method = 'knn',
#                 trControl = ctrl,
#                 data = train_transformed)
# knnFit
# summary(knnFit)
# varImp(knnFit)
# getTrainPerf(knnFit)
# confusionMatrix(knnFit)
# plot(knnFit)

# Support vector machine
# Too slow for an example in a presentation but a commonly used algorithm
# svmLinearFit <- train(OutcomeType ~ .,
#                         method = 'svmLinear',
#                         trControl = ctrl,
#                         metric = "ROC",
#                         data = train_transformed)
# svmLinearFit
# summary(svmLinearFit)
# varImp(svmLinearFit)
# getTrainPerf(svmLinearFit)
# confusionMatrix(svmLinearFit)

# Linear discriminant analysis
ldaFit <- train(OutcomeType ~ ., 
                method = 'lda', 
                trControl = ctrl,
                metric = "ROC",
                data = train_transformed)
ldaFit
summary(ldaFit)
varImp(ldaFit)
getTrainPerf(ldaFit)
confusionMatrix(ldaFit)

# Naive Bayes
# Too slow for an example in a presentation but a commonly used algorithm
# nbFit <- train(OutcomeType ~ .,
#                method = 'nb',
#                trControl = ctrl,
#                metric = "ROC",
#                data = train_transformed)
# nbFit
# summary(nbFit)
# varImp(nbFit)
# getTrainPerf(nbFit)
# confusionMatrix(nbFit)
# plot(nbFit)
# 
# rm(ctrl)

#############################################
# Check Algorithm Performance Using Test Data
#############################################

# Accuracy on the training data doesn't mean much. A good result could mean
# that the algorithm has overfit the data.

# Predict the outcome (OutcomeType) for cases in the test set
# This produces a category (Adopted, Not_Adopted) for each case
glm_test_predictions <- predict(glmFit, newdata = test_transformed)
head(glm_test_predictions)

# We could choose to output the probability of each outcome instead
glm_test_probabilities <- predict(glmFit, newdata = test_transformed, type = "prob")
head(glm_test_probabilities)

rm(glm_test_predictions, glm_test_probabilities)

# Compare the prediction with the actual outcome
# Need to do better than the No Information Rate for the algorithm to be 
# better than guessing

# From 
# https://www.r-project.org/nosvn/conferences/useR-2013/Tutorials/kuhn/user_caret_2up.pdf

# glm
glm_test_predictions <- predict(glmFit, 
                                newdata = test_transformed, 
                                type = "prob") # Produce probability not binary outcome

library(pROC)
glmROC <- roc(predictor = glm_test_predictions$Adopted,
              response = test_transformed$OutcomeType,
              level = rev(levels(test_transformed$OutcomeType)))
glmROC
plot(glmROC, type = "l") # p = points, l = lines, s = stair steps

# rpart
rpart_test_predictions <- predict(rpartFit, 
                                  newdata = test_transformed, 
                                  type = "prob")
rpartROC <- roc(predictor = rpart_test_predictions$Adopted,
                response = test_transformed$OutcomeType,
                level = rev(levels(test_transformed$OutcomeType)))
rpartROC
plot(rpartROC, add = TRUE, col = "blue", type = "l")

# gbm
gbm_test_predictions <- predict(gbmFit, 
                                newdata = test_transformed, 
                                type = "prob")
gbmROC <- roc(predictor = gbm_test_predictions$Adopted,
              response = test_transformed$OutcomeType,
              level = rev(levels(test_transformed$OutcomeType)))
gbmROC
plot(gbmROC, add = TRUE, col = "orange", type = "l")

# ctree
ctree_test_predictions <- predict(ctreeFit, 
                                  newdata = test_transformed, 
                                  type = "prob")
ctreeROC <- roc(predictor = ctree_test_predictions$Adopted,
                response = test_transformed$OutcomeType,
                level = rev(levels(test_transformed$OutcomeType)))
ctreeROC
plot(ctreeROC, add = TRUE, col = "red", type = "l")

# lda
lda_test_predictions <- predict(ldaFit, 
                                  newdata = test_transformed, 
                                  type = "prob")
ldaROC <- roc(predictor = ctree_test_predictions$Adopted,
                response = test_transformed$OutcomeType,
                level = rev(levels(test_transformed$OutcomeType)))
ldaROC
plot(ldaROC, add = TRUE, col = "green", type = "l")

# Not a great plot but it shows the ROC curve for each algorithm

#################################
# Ensemble Two or More Algorithms
#################################

# It's possible to ensemble two or more algorithms together
# For example, a simple method of ensembling is to average the probabilities 
# produced by the algorithms and use that average probability to predict the outcome

# From caretEnsemble vignette: 
# https://cran.r-project.org/web/packages/caretEnsemble/vignettes/caretEnsemble-intro.html

library("caret")
library("mlbench")
library("pROC")
set.seed(107)

# Specify control
my_control <- trainControl(
  method = "boot",
  number = 25,
  savePredictions = "final",
  classProbs = TRUE,
  index = createResample(train$OutcomeType, 25),
  summaryFunction = twoClassSummary
)

# Specify algorithms
library("rpart")
library("caretEnsemble")
model_list <- caretList(
  OutcomeType ~ .,
  data = train_transformed,
  trControl = my_control,
  methodList = c("glm", "ctree")
)

# Train algorithms
ensemble <- caretEnsemble(
  model_list, 
  metric = "ROC",
  trControl = trainControl(
    number = 2,
    summaryFunction = twoClassSummary,
    classProbs = TRUE
  ))

# The ensemble is slightly better than either of the models alone on the training set
summary(ensemble)

# Apply ensemble to test data
library("caTools")
model_preds <- lapply(model_list, predict, newdata=test_transformed, type="prob")
model_preds <- lapply(model_preds, function(x) x[,"Adopted"])
model_preds <- data.frame(model_preds)
ens_preds <- predict(ensemble, newdata=test_transformed, type="prob")
model_preds$ensemble <- ens_preds
caTools::colAUC(model_preds, test_transformed$OutcomeType)

#################################
# Apply an algorithm to new data
#################################

# Get new data
# Make sure this points to wherever you have saved the files on your computer
new_animals <- read.csv("C:/Users/meyerjm/Documents/animal_shelter_new.csv",
                    stringsAsFactors = FALSE) # Avoids factors

# Apply all recoding and transformations to the new data

new_animals$SexuponOutcome <- as.factor(new_animals$SexuponOutcome)
new_animals$AnimalType <- as.factor(new_animals$AnimalType)


new_animals$AgeuponOutcomeNumber <- as.numeric(gsub("([0-9]+).*$", "\\1", new_animals$AgeuponOutcome))

new_animals$AgeuponOutcomeDays <- NA

new_animals$AgeuponOutcomeDays[grepl("(day)|(days)", new_animals$AgeuponOutcome)] <- 
        new_animals$AgeuponOutcomeNumber[grepl("(day)|(days)", new_animals$AgeuponOutcome)]

new_animals$AgeuponOutcomeDays[grepl("(week)|(weeks)", new_animals$AgeuponOutcome)] <- 
        new_animals$AgeuponOutcomeNumber[grepl("(week)|(weeks)", new_animals$AgeuponOutcome)] * 7

new_animals$AgeuponOutcomeDays[grepl("(month)|(months)", new_animals$AgeuponOutcome)] <- 
        new_animals$AgeuponOutcomeNumber[grepl("(month)|(months)", new_animals$AgeuponOutcome)] * 31

new_animals$AgeuponOutcomeDays[grepl("(year)|(years)", new_animals$AgeuponOutcome)] <- 
        new_animals$AgeuponOutcomeNumber[grepl("(year)|(years)", new_animals$AgeuponOutcome)] * 365

new_data <- subset(new_animals, select = c("AnimalType",
                                   "SexuponOutcome", 
                                   "AgeuponOutcomeDays"))

new_data$SexuponOutcome <- gsub(" ", "_", new_data$SexuponOutcome)

new_data$SexuponOutcome[new_data$SexuponOutcome == ""] <- "Unknown"
new_data$SexuponOutcome <- factor(new_data$SexuponOutcome) # remove unused factor level

new_data$AgeuponOutcomeDays[is.na(new_data$AgeuponOutcomeDays)] <- 
        median(new_data$AgeuponOutcomeDays, na.rm = TRUE)

temp <- as.data.frame(model.matrix(~ . -1, data = new_data[c("SexuponOutcome", "AnimalType")]))

temp$SexuponOutcomeUnknown <- NULL 

new_data <- cbind(temp, new_data)
rm(temp)
new_data$SexuponOutcome <- NULL
new_data$AnimalType <- NULL

new_data$SexuponOutcomeIntact_Female <- factor(new_data$SexuponOutcomeIntact_Female)
new_data$SexuponOutcomeIntact_Male <- factor(new_data$SexuponOutcomeIntact_Male)
new_data$SexuponOutcomeNeutered_Male <- factor(new_data$SexuponOutcomeNeutered_Male)
new_data$SexuponOutcomeSpayed_Female <- factor(new_data$SexuponOutcomeSpayed_Female)
new_data$AnimalTypeDog <- factor(new_data$AnimalTypeDog)

nearZeroVar(new_data, saveMetrics = TRUE) # saveMetrics keeps the results rather than a list of variables

new_data_transformed <- predict(preprocess_values, new_data) # Use the preprocess values from the training data set

summary(new_data_transformed)

### Make predictions on new data

# Make binary predictions
predictions <- predict(glmFit, newdata = new_data_transformed)

str(predictions)
summary(predictions)

# Make probability predictions
predictions <- predict(ctreeFit, newdata = new_data_transformed, type = "prob")

str(predictions)
summary(predictions)
