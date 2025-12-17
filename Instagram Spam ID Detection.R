#Instagram Fake spam detection

#Install required packages and libraries
## remove hash before every line below to install packages
#used #to avoid reinstalling everytime

#install.packages("dplyr")
#install.packages("tidyverse")
#install.packages("ggplot2")
#install.packages("caret")
#install.packages("randomForest")
#install.packages("xgboost")
#install.packages("pROC")
#install.packages("corrplot")
#install.packages("PRROC")

library(dplyr)
library(ggplot2)
library(tidyverse)
library(corrplot)
library(gridExtra)
library(caret)
library(randomForest)
library(pROC)

#Load test abd train data

train <- read.csv("C:/Users/abdul/Downloads/train.csv")
test <- read.csv("C:/Users/abdul/Downloads/test.csv")

#train set preview

head(train)                #overview of first few rows
colSums(is.na(train))      #check any missing value
dim(train)                 #data dimention
str(train)                 #show each column and its contetnt
summary(train)             #dataset summary
sum(duplicated(train))     #check duplicate values

#test set preview

head(test)                #overview of first few rows
colSums(is.na(test))      #check any missing value
dim(test)                 #data dimention
str(test)                 #show each column and its contetnt
summary(test)             #dataset summary
sum(duplicated(test))     #check duplicate values

#remove and chec duplicate values

train <- train[!duplicated(train), ]
test <- test[!duplicated(test), ]
sum(duplicated(train))
sum(duplicated(test))     #check duplicate values

#EDA

#Class distribution of fake and genuine accounts

table(train$fake)
cat("\nClass Distribution (%):\n")
round(prop.table(table(train$fake)) * 100, 2)

ggplot(train, aes(x = factor(fake), fill = factor(fake))) +
  geom_bar() +
  labs(title = "Distribution of Target Variable (Fake vs Genuine)",
       x = "Fake (1) / Genuine (0)", y = "Count") +
  scale_fill_manual(values = c("#4D79FF", "#FF4D4D")) +
  theme_minimal()

#Histograms for Numerical Features

num_cols <- train %>% select_if(is.numeric) %>% select(-fake)

hist_plots <- lapply(names(num_cols), function(col) {
  ggplot(train, aes_string(x = col)) +
    geom_histogram(bins = 30, fill = "#4D79FF", color = "black") +
    labs(title = paste("Histogram of", col)) +
    theme_minimal()
})

do.call(grid.arrange, c(hist_plots, ncol = 3))

#correlation matrix

corr_matrix <- cor(num_cols)

corrplot(corr_matrix,
         method = "color",
         tl.cex = 0.8,
         addCoef.col = "black",
         number.cex = 0.5,
         title = "Correlation Heatmap",
         mar = c(0, 0, 1, 0))

#Outliers  check (box plot)

box_plots <- lapply(names(num_cols), function(col) {
  ggplot(train, aes_string(y = col)) +
    geom_boxplot(fill = "#FF704D") +
    labs(title = paste("Boxplot of", col)) +
    theme_minimal()
})

do.call(grid.arrange, c(box_plots, ncol = 3))

#relationship with target variable

relation_plots <- lapply(names(num_cols), function(col) {
  ggplot(train, aes_string(x = "factor(fake)", y = col, fill = "factor(fake)")) +
    geom_boxplot() +
    labs(title = paste(col, "vs Fake/Genuine"),
         x = "Fake (1) / Genuine (0)") +
    theme_minimal()
})

do.call(grid.arrange, c(relation_plots, ncol = 3))


#feature engineering

# TRAIN DATA
# Username characteristics
train$username_numeric_ratio <- train$nums.length.username /
  pmax(nchar(as.character(train$name..username)), 1)

# Full name characteristics
train$fullname_numeric_ratio <- train$nums.length.fullname /
  pmax(train$fullname.words, 1)

# Account activity ratios
train$followers_following_ratio <- train$X.followers /
  pmax(train$X.follows, 1)

train$posts_followers_ratio <- train$X.posts /
  pmax(train$X.followers, 1)

# Profile completeness indicators
train$has_profile_pic <- ifelse(train$profile.pic == 1, 1, 0)

train$has_external_url <- ifelse(train$external.URL == 1, 1, 0)

# Privacy indicator
train$is_private <- ifelse(train$private == 1, 1, 0)

# TEST DATA
# Username characteristics
test$username_numeric_ratio <- test$nums.length.username /
  pmax(nchar(as.character(test$name..username)), 1)

# Full name characteristics
test$fullname_numeric_ratio <- test$nums.length.fullname /
  pmax(test$fullname.words, 1)

# Account activity ratios
test$followers_following_ratio <- test$X.followers /
  pmax(test$X.follows, 1)

test$posts_followers_ratio <- test$X.posts /
  pmax(test$X.followers, 1)

# Profile completeness indicators
test$has_profile_pic <- ifelse(test$profile.pic == 1, 1, 0)

test$has_external_url <- ifelse(test$external.URL == 1, 1, 0)

# Privacy indicator
test$is_private <- ifelse(test$private == 1, 1, 0)


#Save feature engineered data

write.csv(train,"C:/Users/abdul/Downloads/train_fe.csv", row.names = FALSE)
write.csv(test,"C:/Users/abdul/Downloads/test_fe.csv", row.names = FALSE)

#load engineered data
train_fe <- read.csv("C:/Users/abdul/Downloads/train_fe.csv")
test_fe <- read.csv("C:/Users/abdul/Downloads/test_fe.csv")

# Check new structure
glimpse(train_fe)
glimpse(test_fe)

#Feature Selection

#correlation analysis
# Select only numeric columns
num_vars <- train_fe %>% select(where(is.numeric))

# Correlation matrix
corr_matrix <- cor(num_vars, use = "pairwise.complete.obs")

# Visualize the top correlations
corrplot::corrplot(corr_matrix, method = "color", tl.cex = 0.7)

# Identify highly correlated pairs
high_corr <- findCorrelation(corr_matrix, cutoff = 0.85)

# Remove them
train_reduced <- num_vars[, -high_corr]
names(train_reduced)

#eliminate recursive feature

set.seed(123)

control <- rfeControl(
  functions = rfFuncs,
  method = "cv",
  number  = 5
)

# Define predictors & target
X <- train_reduced %>% select(-fake)    
y <- train_reduced$fake

# Perform RFE
rfe_results <- rfe(
  x = X,
  y = y,
  sizes = c(5, 10, 15, 20),
  rfeControl = control
)

# Selected features
selected_features1 <- predictors(rfe_results)
selected_features1


#random forest variable importance 

library(randomForest)
set.seed(123)

rf_model <- randomForest(
  fake ~ .,           
  data = train_fe,
  importance = TRUE
)

importance_df <- data.frame(
  Feature = rownames(rf_model$importance),
  Importance = rf_model$importance[, 1]
)

# Plot importance
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Feature", y = "Importance")

#select important features
selected_features <- c(
  "profile.pic",
  "nums.length.username",
  "fullname.words",
  "description.length",
  "X.posts",
  "X.followers",
  "X.follows",
  "followers_following_ratio",
  "posts_followers_ratio",
  "username_numeric_ratio",
  "has_profile_pic",
  "fake"   # include target ONLY for training
)

final_selected <- intersect(selected_features, selected_features1)
final_selected <- union(final_selected, "fake")
final_selected
train_final <- train_fe %>% select(all_of(final_selected))

test_final <- test_fe %>% select(-fake) %>%   # test has no labels logically
  select(all_of(setdiff(final_selected, "fake")))

glimpse(train_final)       #view final train data
glimpse(test_final)        #view final test data

#MODEL BUILDING

#train-control setup

set.seed(123)

train_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

#logisctic regression

train_final$fake <- factor(train_final$fake,
                           levels = c(0, 1),
                           labels = c("Real", "Fake"))
levels(train_final$fake)
# "Real" "Fake"

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)
log_model <- train(
  fake ~ .,
  data = train_final,
  method = "glm",
  family = "binomial",
  metric = "ROC",
  trControl = ctrl
)
log_model

#random forest
set.seed(123)
model_rf <- train(
  fake ~ .,
  data = train_final,
  method = "rf",
  trControl = ctrl,
  metric = "ROC"
)
model_rf

#XGBoost
set.seed(123)

xgb_model <- train(
  fake ~ .,
  data = train_final,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl
)
xgb_model

#SVM (Radial Kernel)
set.seed(123)
model_svm <- train(
  fake ~ .,
  data = train_final,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC"
)
model_svm

#Compare model performance
results <- resamples(list(
  Logistic = log_model,
  RandomForest = model_rf,
  XGBoost = xgb_model,
  SVM = model_svm
))

summary(results)
bwplot(results, metric = "ROC")
dotplot(results, metric = "ROC")


## Project Done