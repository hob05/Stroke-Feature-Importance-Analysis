# data set created by user "fedesoriano" on Kaggle: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

# TO DO: FIX IMBALANCING SO THAT THE CALIBRATION AND MCNEMAR ARE BETTER. DO THIS VIA SMOTE AND WHATEVER THE FUCK CHATGPT MADE 

        #### Format Data ####
        
library(installr)
library(caret)
library(tidyverse)
library(tidymodels)
library(pROC)
library(PRROC)
library(themis)

db <- read.csv("healthcare-dataset-stroke-data.csv")

NA_num <- sum(db$bmi == "N/A", na.rm = TRUE)  # Count "N/A" values in BMI
NA_num / nrow(db) # only 3.93% of the data is missing so I will omit instead of using extra computational power to fill in the missing data

db <- db %>%
  mutate(bmi = as.numeric(replace(bmi, bmi == "N/A", NA))) %>%  # Replace "N/A" with NA and convert to numeric
  drop_na(bmi)  # Remove rows with NA in BMI   


set.seed(123)
db_shuf <- db[sample(1:nrow(db)),]
db_shuf <- subset(db_shuf, select = -id)

db_bound <- nrow(db_shuf) * 0.75

db_trn <- db_shuf[1:db_bound,]
db_tst <- db_shuf[(db_bound+1):nrow(db_shuf),]

table(db_trn$stroke)
table(db_tst$stroke) # the datasets are highly imbalanced, so must use balancing techniques

        #### Apply SMOTE for oversampling to make sure the model is not skewed towards negative stroke predictions in real testing ####
## Train set ##
db_trn_num <- db_trn %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), as.numeric))

db_trn_num$stroke <- as.factor(db_trn_num$stroke)

smote_recipe <- recipe(stroke ~ ., data = db_trn_num) %>%
  step_smote(stroke, over_ratio = 1)  # Balance classes (1:1 ratio)

# Apply SMOTE (prep & bake the data)
db_trn_smote <- smote_recipe %>%
  prep() %>%
  bake(new_data = NULL)

# Apply Tomek Links#
db_recipe <- recipe(stroke ~ ., data = db_trn_smote) %>%
  step_tomek(stroke)  

# Prepare and apply the preprocessing
db_trn_balanced <- prep(db_recipe) %>%
  bake(new_data = NULL)  # Get the transformed dataset

table(db_trn_balanced$stroke) # check to see if the stroke data is now balanced

sum(duplicated(db_trn_smote)) # check to make sure oversampling hasn't caused significantly duplicated data


## Test Set ##
db_tst_num <- db_tst %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), as.numeric))

db_tst_num$stroke <- as.factor(db_tst_num$stroke)

smote_recipe <- recipe(stroke ~ ., data = db_tst_num) %>%
  step_smote(stroke, over_ratio = 1)  # Balance classes (1:1 ratio)

# Apply SMOTE (prep & bake the data)
db_tst_smote <- smote_recipe %>%
  prep() %>%
  bake(new_data = NULL)

# Apply Tomek Links#
db_recipe <- recipe(stroke ~ ., data = db_tst_smote) %>%
  step_tomek(stroke)  

# Prepare and apply the preprocessing
db_tst_balanced <- prep(db_recipe) %>%
  bake(new_data = NULL)  # Get the transformed dataset

table(db_tst_balanced$stroke) # check to see if the stroke data is now balanced

sum(duplicated(db_tst_smote)) # check to make sure oversampling hasn't caused significantly duplicated data

        #### Train Model ####
train_control <- trainControl(method = "cv", 
                              number = 51) 

db_trn_rf <- train(as.factor(stroke) ~ .,
                   data = db_trn_smote, 
                   method = "rf",
                   trControl = train_control,
                   ntree = 250) 

saveRDS(db_trn_rf,"db_trn_rf.rds") # save the model so it will create the same output every time and so I don't have to wait for it to re-train every time
db_trn_rf <- readRDS("db_trn_rf.rds")

print(db_trn_rf)


    #### Test Model ####
db_pred <- predict(db_trn_rf, newdata = db_tst_smote)

db_tst_smote$stroke <- factor(db_tst_smote$stroke, levels = levels(db_pred))

stroke_cm <- confusionMatrix(db_pred, db_tst_smote$stroke)
stroke_cm
stroke_cm$byClass


    #### Feature Importance ####

var_imp <- varImp(db_trn_rf)  # Assuming db_trn_rf is your trained model
print(var_imp)

# Plot variable importance
plot(var_imp, main = "Variable Importance (Gini Index)")

cor(db_trn_num[, c("age", "avg_glucose_level", "hypertension", "bmi")]) # no redundancy 


    #### ROC Curve #### 

db_probs <- predict(db_trn_rf, db_tst_smote, type = "prob")

roc_curve <- roc(db_tst_smote$stroke, db_probs[, 2])
plot(roc_curve, main = "ROC Curve for Stroke Prediction")
auc(roc_curve) # 0.9849


    #### Precision-Recall Curve ####
db_tst_smote$stroke <- as.numeric(as.character(db_tst_smote$stroke))

pr_curve <- pr.curve(scores.class0 = db_probs[, 2], weights.class0 = db_tst_smote$stroke, curve = TRUE)
plot(pr_curve) # AUC = 0.9877343


    #### Model Calibration Curve ####
db_tst_smote$stroke <- as.factor(db_tst_smote$stroke)

cal_curve <- calibration(db_tst_smote$stroke ~ db_probs[, 2], bins = 10)
plot(cal_curve) # Model is highly poorly calibrated so I need to adjust this


    #### Stat Testing ####

mcnemar.test(stroke_cm$table) # Bad, likely due to imbalancing/poor calibration



