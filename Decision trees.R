#---------------------------------------------------
#---------------------------------------------------
#----- Decision Trees
#---------------------------------------------------
#---------------------------------------------------


##-------------------------------------
##----- 1. Prepare the environment and load data
##-------------------------------------

# Clear the workspace
rm(list = ls()) # Clear environment
gc()            # Clear memory
cat("\f")       # Clear the console
options(scipen = 5) # Remove scientific notation for numbers
# Prepare needed libraries
packages <- c("ggplot2"     # Best plotting
              , "gridExtra" # Arrange multiple plots in a grid
              , "ISLR2"     # Textbook datasets
              , "labelled"  # Label variables like good ole Stata
              , "stargazer" # Nice output tables
              , "tree"      # Classification and regression trees
              , "randomForest"  # Random forest
              , "xgboost"       # Boosting
)
for (i in 1:length(packages)) {
  if (!packages[i] %in% rownames(installed.packages())) {
    install.packages(packages[i]
                     , repos = "https://cran.rstudio.com/"
                     , dependencies = TRUE
    )
  }
  library(packages[i], character.only = TRUE)
}
rm(packages)

# Load Carseats dataset
data.carseats <- Carseats

# Assign labels to variables
var_label(data.carseats$Sales) <- 
  "Unit sales (in thousands) at each location"

var_label(data.carseats$CompPrice) <- 
  "Price charged by competitor at each location"

var_label(data.carseats$Income) <- 
  "Community income level (in thousands of dollars)"

var_label(data.carseats$Advertising) <- 
  "Local advertising budget for company at each location (in thousands of dollars)"

var_label(data.carseats$Population) <- 
  "Population size in region (in thousands)"

var_label(data.carseats$Price) <- 
  "Price company charges for car seats at each site"

var_label(data.carseats$ShelveLoc) <- 
  "A factor with levels Bad, Good, and Medium indicating the quality
   of the shelving location for the car seats at each site"

var_label(data.carseats$Age) <- 
  "Average age of the local population"

var_label(data.carseats$Education) <- 
  "Education level at each location"

var_label(data.carseats$Urban) <- 
  "A factor with levels No and Yes to indicate whether the store is in an 
   urban or rural location"

var_label(data.carseats$US) <- 
  "A factor with levels No and Yes to indicate whether the store is in the 
   US or not"

# View the data with the labels
View(data.carseats)

# Return the structure of the dataset
str(data.carseats) # PollEv

# Let's create a binary variable as the response variable for the classification tree
data.carseats$High <- factor(ifelse(data.carseats$Sales <= 8, "No", "Yes"))

# Let's do a train/test split
set.seed(100)
rows.train <- sample(nrow(data.carseats), 0.8*nrow(data.carseats))
data.train <- data.carseats[rows.train, ]
data.test <- data.carseats[-rows.train, ]

##-------------------------------------
## 2. Classification Trees
##-------------------------------------

# Fit the classification tree
tree.carseats <- tree(High ~ . - Sales, data.train) # mincut = 5: obs in child node
                                                    # minsize = 10: obs for split
                                                    # mindev = 0.01: entropy improvement

# View a summary of the fit (error rate, number of terminal nodes)
summary(tree.carseats) # PollEv

# Plot the tree structure
plot(tree.carseats)
text(tree.carseats, pretty = 0) # PollEv

# See the numerical breakdown of the nodes
tree.carseats

# Use cross-validation to determine the optimal level of complexity (PollEv)
set.seed(7)
cv.carseats <- cv.tree(tree.carseats)
names(cv.carseats) # PollEv
cv.carseats

# Diagnostic plots used to determine the optimal level of tree complexity (PollEv)
par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")

# Prune the tree
prune.carseats <- prune.misclass(tree.carseats, best = 4)
plot(prune.carseats)
text(prune.carseats, pretty = 0) # PollEv

# Compare the prediction performance of the full tree vs pruned tree
# Predict using the unpruned tree
tree.pred.unpruned <- predict(tree.carseats, data.test, type = "class")

# Create the confusion matrix
conf.matrix.unpruned <- table(tree.pred.unpruned, data.test$High)
print(conf.matrix.unpruned)

# Calculate the accuracy
accuracy.unpruned <- sum(diag(conf.matrix.unpruned))/sum(conf.matrix.unpruned)
print(accuracy.unpruned)

# Predict using the pruned tree
tree.pred.pruned <- predict(prune.carseats, data.test, type = "class")

# Create the confusion matrix
conf.matrix.pruned <- table(tree.pred.pruned, data.test$High)
print(conf.matrix.pruned)

# Calculate the accuracy
accuracy.pruned <- sum(diag(conf.matrix.pruned))/sum(conf.matrix.pruned)
print(accuracy.pruned) # PollEv

##-------------------------------------
## 3. Regression Trees
##-------------------------------------

# Fit the regression tree
regtree.carseats <- tree(Sales ~ . - High, data.train)

# View the summary
summary(regtree.carseats)

# Plot the tree structure
dev.off()
plot(regtree.carseats)
text(regtree.carseats, pretty = 0)

# See the numerical breakdown of the nodes
regtree.carseats

# Use cross-validation to determine the optimal level of complexity
set.seed(7)
cv.reg.carseats <- cv.tree(regtree.carseats)
names(cv.reg.carseats)
cv.reg.carseats

# Diagnostic plots used to determine the optimal level of tree complexity
par(mfrow = c(1, 2))
plot(cv.reg.carseats$size, cv.reg.carseats$dev, type = "b")
plot(cv.reg.carseats$k, cv.reg.carseats$dev, type = "b")

# Prune the tree
pruned_tree_reg <- prune.tree(regtree.carseats, best = 14)
dev.off()
plot(pruned_tree_reg)
text(pruned_tree_reg, pretty = 0)

# Compare the prediction performance of the full tree vs pruned tree
# Predict using the unpruned tree
y_hat_unpruned_tree <- predict(regtree.carseats, data.test)
y_actual <- data.test$Sales

# Calculate test MSE
test_mse_unpruned <- mean((y_hat_unpruned_tree - y_actual)^2)
print(test_mse_unpruned)

# Predict using the pruned tree
y_hat_pruned_tree <- predict(pruned_tree_reg, data.test)

# Calculate test MSE
test_mse_pruned <- mean((y_hat_pruned_tree - y_actual)^2)
print(test_mse_pruned)


##-------------------------------------
## 4. Bagging for classification
##-------------------------------------

# Set random seed for bagging (PollEv)
set.seed(4)

# Fit the bagging ensemble for the classification model (PollEv)
cla.bag <- randomForest(High ~ . - Sales, data = data.train,
                    mtry = 10, importance = TRUE)

# Return a summary of the model
cla.bag # PollEv

# Predict the values of the High variable using the test observations
yhat.cla.bag <- predict(cla.bag, newdata = data.test, type = "class")

# Create the confusion matrix
cla.bag.matrix <- table(Predict = yhat.cla.bag, Actual = data.test$High)
print(cla.bag.matrix)

# calculate the accuracy (PollEv)
accuracy.cla.bag <- sum(diag(cla.bag.matrix))/sum(cla.bag.matrix)
print(accuracy.cla.bag)

##-------------------------------------
## 5. Bagging for regression
##-------------------------------------

# Set random seed for bagging
set.seed(5)

# Fit the bagging ensemble
reg.bag <- randomForest(Sales ~ . - High, data = data.train,
                    mtry = 10, importance = TRUE)

# Return a summary of the model
reg.bag # PollEv

# Predict the Sales values using the test observations
yhat.reg.bag <- predict(reg.bag, newdata = data.test)

# Creates a scatter plot comparing the predictions against the actual values
plot(yhat.reg.bag, data.test$Sales)

# Add a 45-degree reference line
abline(0, 1)

# Calculate test MSE (PollEv)
mse.reg.bag <- mean((yhat.reg.bag - data.test$Sales)^2)
print(mse.reg.bag)


##-------------------------------------
## 6. Random Forest for classification
##-------------------------------------

# Set random seed
set.seed(4)

# Fit a random forest model with a maximum variable size of 5 at each split
cla.rf <- randomForest(High ~ . - Sales, data = data.train, 
                       mtry = 5, importance = TRUE)

# Return a summary of the model
cla.rf # PollEv

# Predict the values of the High variable using the test observations
yhat.cla.rf <- predict(cla.rf, newdata = data.test, type = "class")

# Create the confusion matrix
cla.rf.matrix <- table(Predict = yhat.cla.rf, Actual = data.test$High)
print(cla.rf.matrix)

# Calculate the accuracy
accuracy.cla.rf <- sum(diag(cla.rf.matrix))/sum(cla.rf.matrix)
print(accuracy.cla.rf) # PollEv

# Output the table of importance measures
importance(cla.rf)

# Plot the importance of features
varImpPlot(cla.rf)


##-------------------------------------
## 7. Random forest for regression
##-------------------------------------

# Set random seed
set.seed(5)

# Fit the random forest
reg.rf <- randomForest(Sales ~ . - High, data = data.train,
                        mtry = 6, importance = TRUE)

# Return a summary of the model
reg.rf # PollEv

# Predict the Sales values using the test observations
yhat.reg.rf <- predict(reg.rf, newdata = data.test)

# Creates a scatter plot comparing the predictions against the actual values
plot(yhat.reg.rf, data.test$Sales)

# Add a 45-degree reference line
abline(0, 1)

# Calculate test MSE (PollEv)
mse.reg.rf <- mean((yhat.reg.rf - data.test$Sales)^2)
print(mse.reg.rf)


##-------------------------------------
## 8. Boosting for classification
##-------------------------------------

# Isolate the outcome variable from the features
train_y_cla <- as.numeric(data.train$High) - 1
test_y_cla <- as.numeric(data.test$High) - 1

# Convert factor variables into dummies
train_x_cla <- model.matrix(High ~. - Sales - High, data = data.train)[, -1]
test_x_cla <- model.matrix(High ~. - Sales - High, data = data.test)[, -1]

# Convert to DMatrix objects
dtrain_cla <- xgb.DMatrix(data = train_x_cla, label = train_y_cla)
dtest_cla <- xgb.DMatrix(data = test_x_cla, label = test_y_cla)

# Set parameters
params_cla <- list(
  objective = "binary:logistic", # 0/1 classification
  eval_metric = "error",
  lambda = 0,
  alpha = 0, # turn off the regularizations
  max_depth = 1, # complexity
  eta = 0.1, # Learning rate
  subsample = 1,
  colsample_bytree = 1 # use all the observations
)

# Set random seed
set.seed(4)

# Train the model
xgb_cla <- xgb.train(
  params = params_cla,
  data = dtrain_cla,
  nrounds = 200, # Number of trees
  evals = list(train = dtrain_cla, test = dtest_cla),
  print_every_n = 10
)

# Predict on the test set
pred_cla <- predict(xgb_cla, dtest_cla)

# Set cutoff point for 0/1 binary outcome
prediction_binary <- as.numeric(pred_cla > 0.5)

# Output the confusion matrix
cla_boo_matrix <- table(Actual = test_y_cla, Prdicted = prediction_binary)
print(cla_boo_matrix)

# Calculate the accuracy
accuracy.cla.boo <- sum(diag(cla_boo_matrix))/sum(cla_boo_matrix)
print(accuracy.cla.boo) # 2 PollEvs

##-------------------------------------
## 9. Boosting for regression
##-------------------------------------

# Isolate the outcome variable from the features
train_y_reg <- data.train$Sales
test_y_reg <- data.test$Sales

# Convert factor variables into dummies
train_x_reg <- model.matrix(Sales ~. - Sales - High, data = data.train)[, -1]
test_x_reg <- model.matrix(Sales ~. - Sales - High, data = data.test)[, -1]

# Convert to DMatrix objects
dtrain_reg <- xgb.DMatrix(data = train_x_reg, label = train_y_reg)
dtest_reg <- xgb.DMatrix(data = test_x_reg, label = test_y_reg)

# Set parameters
params_reg <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  lambda = 0,
  alpha = 0, # turn off the regularizations
  max_depth = 2, # complexity
  eta = 0.01, # Learning rate
  subsample = 1,
  colsample_bytree = 1 # use all the observations
)

# Set random seed
set.seed(5)

# Train the model
xgb_reg <- xgb.train(
  params = params_reg,
  data = dtrain_reg,
  nrounds = 1000, # Number of trees
  evals = list(train = dtrain_reg, test = dtest_reg),
  print_every_n = 10
)

# Extract the full history into a data frame
eval_history <- attr(xgb_reg, "evaluation_log")

# Store the MSE values 
mse.reg.boo <- (tail(eval_history$test_rmse, 1))^2
print(mse.reg.boo) # 2 PollEvs

# Output the table of importance measures
importance_matrix <- xgb.importance(model = xgb_reg)

# Plot the importance of features
xgb.plot.importance(importance_matrix) # PollEv


##-------------------------------------
## 10. Conclusion Table
##-------------------------------------

# Create a vector of model names
model_names <- c("Unpruned Tree", "Pruned Tree", "Bagging", 
                 "Random Forest", "Boosting")

# Create vectors for stored metrics
accuracy_values <- c(accuracy.unpruned, accuracy.pruned, accuracy.cla.bag,
                     accuracy.cla.rf, accuracy.cla.boo)
mse_values <- c(test_mse_unpruned, test_mse_pruned, mse.reg.bag,
                mse.reg.rf, mse.reg.boo)

# Combine into table
performance_table <- data.frame(
  Model = model_names,
  Classification_Accuracy = accuracy_values,
  Regression_MSE = mse_values
)
View(performance_table)
