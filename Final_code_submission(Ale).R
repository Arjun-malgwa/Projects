library(xgboost)
library(pROC)
library(ggplot2)

set.seed(42)

data_train <- read.csv("train.csv", stringsAsFactors = FALSE)
data_test <- read.csv("test.csv", stringsAsFactors = FALSE)

# target class distribution
target_distribution <- table(data_train$churn)
print("Target Class Distribution:")
print(target_distribution)

# class frequencies
class_names <- c("No Churn", "Churn")
class_frequencies <- as.character(target_distribution)

# colors for churn and no churn
churn_color <- "lightcoral"
no_churn_color <- "lightgreen"

# target class distribution 
barplot(target_distribution, 
        main = "Target Class Distribution", 
        xlab = "Churn", 
        ylab = "Frequency", 
        col = ifelse(names(target_distribution) == "1", churn_color, no_churn_color),
        names.arg = class_names,
        ylim = c(0, max(target_distribution) + 100),
        cex.names = 0.8)

legend("topleft", 
       legend = paste(class_names, ": ", class_frequencies), 
       fill = c(no_churn_color, churn_color), 
       title = "Target Class", 
       bty = "n", # Turn off legend box
       cex = 0.8,  # Adjust font size
       x.intersp = 1.5, # Adjust horizontal spacing between legend items
       y.intersp = 1, # Adjust vertical spacing between legend items
       inset = c(0.02, 0.02))  # Adjust inset from plot edges

# churn to binary
data_train$churn <- as.factor(ifelse(data_train$churn == 1, "yes", "no"))

# features excluding id & churn
features <- setdiff(names(data_train), c('churn', 'id'))

# Convert factors to numeric and handle missing values
data_train[features] <- lapply(data_train[features], function(x) {
  if(is.numeric(x)) {
    ifelse(is.na(x), median(x, na.rm = TRUE), x)  # Replace NA with median for numerical
  } else {
    ifelse(is.na(x), 'Missing', x)  # Replace NA with 'Missing' for categorical
  }
})

for (feature_name in features) {
  if(is.numeric(data_train[[feature_name]])) {
    # training median
    median_value <- median(data_train[[feature_name]], na.rm = TRUE)
    # median imputation to test
    data_test[[feature_name]][is.na(data_test[[feature_name]])] <- median_value
  } else {
    # replace NA with missing
    data_test[[feature_name]][is.na(data_test[[feature_name]])] <- 'Missing'
  }
}

# correlation between each independent variable and churn
correlation <- sapply(data_train[features], function(x) {
  cor.test(as.numeric(x), as.numeric(data_train$churn))$estimate
})

# Create a dataframe with variable names and their correlation coefficients
correlation_data <- data.frame(
  Feature = names(correlation),
  Correlation = correlation
)

# Sort df by correlation coefficient
correlation_data <- correlation_data[order(correlation_data$Correlation, decreasing = TRUE), ]

# Reverse the order of independent variables
correlation_data$Feature <- factor(correlation_data$Feature, levels = correlation_data$Feature)

ggplot(correlation_data, aes(x = Feature, y = Correlation)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  ggtitle("Correlation of Independent Variables with Churn") +
  xlab("Independent Variables") +
  ylab("Correlation Coefficient") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(plot.title = element_text(hjust = 0.5))

# Prepare data for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(data_train[, features]), label = as.numeric(data_train$churn)-1)
dtest <- xgb.DMatrix(data = as.matrix(data_test[, features]))

# Check for class imbalance
class_imbalance <- prop.table(table(data_train$churn))
print("Class Imbalance:")
print(class_imbalance)


# Define XGBoost parameters
params <- list(
  objective = "binary:logistic",       
  booster = "gbtree",                  
  eval_metric = "auc",                 
  eta = 0.01,                          
  max_depth = 4,                      
  subsample = 0.8,                     
  colsample_bytree = 0.8,              
  min_child_weight = 5,                
  lambda = 3,                         
  alpha = 2,                           
  colsample_bylevel = 0.8,             
  colsample_bynode = 0.8               
)

# Placeholder for the best model's AUC and parameters
best_auc <- 0
best_params <- params

# Grid search for hyperparameter tuning
for (eta in c(0.01, 0.1)) {
  for (max_depth in c(4, 6)) {
    params$eta <- eta
    params$max_depth <- max_depth
    set.seed(420)  
    # Cross-validation
    cv <- xgb.cv(
      params = params,
      data = dtrain,
      nrounds = 100,
      nfold = 5,
      stratified = TRUE,
      print_every_n = 10,
      early_stopping_rounds = 10,
      maximize = TRUE
    )
    
    # Capture best AUC and parameters
    if (max(cv$evaluation_log$test_auc_mean) > best_auc) {
      best_auc <- max(cv$evaluation_log$test_auc_mean)
      best_params <- params
    }
  }
}

print(paste("Best AUC:", best_auc))

# AUC values from the cross-validation results
train_auc <- cv$evaluation_log$train_auc_mean
test_auc <- cv$evaluation_log$test_auc_mean
iteration <- seq_along(train_auc)

#df for plotting
auc_curve_data <- data.frame(
  Iteration = iteration,
  Train_AUC = train_auc,
  Test_AUC = test_auc
)

ggplot(auc_curve_data, aes(x = Iteration)) +
  geom_line(aes(y = Train_AUC, color = "Train AUC")) +
  geom_line(aes(y = Test_AUC, color = "Test AUC")) +
  scale_color_manual(values = c("Train AUC" = "blue", "Test AUC" = "red")) +
  ggtitle("Training and Validation AUC Curve") +
  ylab("AUC") +
  xlab("Iteration") +
  theme(plot.title = element_text(hjust = 0.5))


print("Best Parameters:")
print(best_params)

# Train  XGBoost model best parameters 
best_model <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = 500
)


# predicted probabilities for positive class
predictions <- predict(best_model, newdata = dtest)

# Feature importance analysis
model <- xgb.train(params = best_params, data = dtrain, nrounds = 100)
importance_matrix <- xgb.importance(feature_names = colnames(dtrain), model = model)
print(xgb.plot.importance(importance_matrix))

#submission file
submission <- data.frame(id = data_test$id, churn = predictions)
write.csv(submission, "submissionFINALXgb.csv", row.names = FALSE)








