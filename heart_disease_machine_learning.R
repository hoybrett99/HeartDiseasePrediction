# Heart disease detection - machine learning

# Libraries
library(ggplot2)
library(GGally)
library(tidyverse)
library(randomForest)
library(caret)
library(glmnet)
library(purrr)



# Setting the working directory
setwd("C:\\Users\\hoybr\\Documents\\data_projects\\heart disease\\updated\\2022")

# Importing the data
heart <- read.csv("heart_2022_no_nans.csv")

# Theme for graph
theme.my.own <- function(){
  theme_bw()+
    theme(axis.text.x = element_text(size = 12, angle = 0, vjust = 1, hjust = 1),
          axis.text.y = element_text(size = 12, angle = 45),
          axis.title.x = element_text(size = 14, face = "plain"),             
          axis.title.y = element_text(size = 14, face = "plain"),             
          panel.grid.major.x = element_blank(),                                          
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.y = element_blank(),  
          plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), units = , "cm"),
          plot.title = element_text(size = 20, vjust = 1, hjust = 0.5),
          legend.text = element_text(size = 12, face = "italic"),          
          legend.title = element_text(size = 15, face = "bold.italic"),
          legend.background = element_rect(linetype = "solid", 
                                           colour = "black"))
}

# First few observations
head(heart)

# Format of the variables
str(heart)

# Number of variables
length(heart)

# Creating a new column heart disease
heart_disease <- heart %>% 
  mutate(HeartDisease = as.factor(case_when(HadHeartAttack == "Yes" ~ 1,
                                  HadAngina == "Yes" ~ 1,
                                  TRUE ~ 0))) %>% 
  dplyr::select(HeartDisease, -HadHeartAttack, -HadAngina, everything())


# Exploratory data analysis

# Extracting character data
char_var <- sapply(heart_disease, is.character)
heart_character <- heart_disease[char_var]

# Adding heart disease variable
heart_character <- cbind(HeartDisease = heart_disease$HeartDisease, heart_character) 

# Getting the total number of heart disease patients 
total_patients <- heart_disease %>%
  dplyr::select(HeartDisease) %>% 
  filter(HeartDisease == 1) %>% 
  nrow()


# State
heart_disease %>% 
  group_by(Sex, HeartDisease) %>% 
  summarise(Patients = n(), .groups = "drop") %>%
  filter(HeartDisease == 1) %>% 
  mutate(total = total_patients) %>% 
  mutate(percent = Patients/total * 100)


# Define a function to create a plot for a given variable
plot_heart_disease <- function(data, var) {
  data %>%
    group_by(!!sym(var), HeartDisease) %>%
    summarise(Patients = n(), .groups = "drop") %>%
    filter(HeartDisease == 1) %>% 
    mutate(total = total_patients) %>% 
    mutate(percent = Patients/total * 100) %>%
    ggplot(aes(x = !!sym(var), y = percent, fill = !!sym(var))) +
    geom_col(position = position_dodge(width = 0.8)) +
    theme.my.own() +
    labs(title = paste("HeartDisease vs", var),
         x = var,
         y = "Proportion of Patients with Heart Disease (%)") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# List of variables you want to plot against HeartDisease
variables_to_plot <- variables_to_plot <- setdiff(colnames(heart_character), "HeartDisease")


# Create plots for all specified variables
plots <- map(variables_to_plot, ~plot_heart_disease(heart_character, .x))

# View the plots
plots

# Numeric data
heart_numeric <- sapply(heart_disease, is.numeric)


# Extracting non character variables
heartnumeric <- heart_disease[heart_numeric]

# Ensure heartnumeric includes HeartDisease
heartnumeric <- cbind(HeartDisease = heart_disease$HeartDisease, heartnumeric) %>%
  mutate(HeartDisease = as.factor(HeartDisease))

# Determine the number of samples per group
n_per_group <- 500

# Sample equal numbers from each group in heartnumeric
heartnumeric_sampled <- heartnumeric %>%
  slice_sample(n = n_per_group) %>%
  mutate(HeartDisease = as.numeric(HeartDisease))

# Getting the correlation between the variables
ggpairs(heartnumeric_sampled,
        lower = list(continuous = "smooth"),
        diag = list(continuous = "densityDiag")) +
  theme(legend.position = "right",
        legend.title = element_text(face = "bold"),
        legend.text = element_text(size = 8))



# Extracting character varaiables
heart_char <- sapply(heart, is.character) 
heartcharacter <- heart[heart_char]



# Using a loop to find the unique values from each variables
for( i in colnames(heartcharacter)){
  unq <- unique(heartcharacter[[i]])
  
  # Printing the variables and its own unique values
  print("")
  print(i)
  print(unq)
}


# Removing state and weight variable in the dataset
heart1 <- heart %>% 
  dplyr::select(-State) %>% 
  mutate(across(where(~ all(. %in% c("Yes", "No"))),
                ~ case_when(
                  . == "Yes" ~ 1,
                  . == "No" ~ 0
                )))


# One hot encode some variables in the dataset
dummy <- dummyVars(" ~ . -Sex", data = heart1)
encoded_heart <- data.frame(predict(dummy, newdata = heart1)) 

# Names of the columns
colnames(encoded_heart)

# Renaming the variables
encoded_heart_renamed <- encoded_heart %>% 
  dplyr::rename(Nodiabetes = HadDiabetesNo,
                Noprediabetes = HadDiabetesNo..pre.diabetes.or.borderline.diabetes,
                Diabetes = HadDiabetesYes,
                DiabetesPregnant = HadDiabetesYes..but.only.during.pregnancy..female.,
                Age18to24 = AgeCategoryAge.18.to.24,
                Age25to29 = AgeCategoryAge.25.to.29,
                Age30to34 = AgeCategoryAge.30.to.34,
                Age35to39 = AgeCategoryAge.35.to.39,
                Age40to44 = AgeCategoryAge.40.to.44,
                Age45to49 = AgeCategoryAge.45.to.49,
                Age50to54 = AgeCategoryAge.50.to.54,
                Age55to59 = AgeCategoryAge.55.to.59,
                Age60to64 = AgeCategoryAge.60.to.64,
                Age65to69 = AgeCategoryAge.65.to.69,
                Age70to74 = AgeCategoryAge.70.to.74,
                Age75to79 = AgeCategoryAge.75.to.79,
                Ageplus80 = AgeCategoryAge.80.or.older
                )

# Joining the data together and removing the weight variable
heart_encoded_complete <- cbind(HeartDisease = heartnumeric$HeartDisease, Sex = heart$Sex, encoded_heart_renamed) %>% 
  mutate(Sex = as.numeric(as.factor(Sex))) %>% 
  select(-WeightInKilograms)

str(heart_encoded_complete)

# Checking the number of variables
length(heart_encoded_complete)
dim(heart_encoded_complete)



# Sampling 2000 datapoints from the dataset
# Set seed for reproducibility
set.seed(123)


# Determine the number of samples per group
number_per_group <- 1000

# Sample equal numbers from each group of heart disease
heart_sample_ml <- heart_encoded_complete %>% 
  dplyr::group_by(HeartDisease) %>% 
  slice_sample(n = number_per_group) %>% 
  ungroup() %>% 
  select(-HadHeartAttack, -HadAngina)

# Checking the numbers from each group
heart_sample_ml %>% 
  group_by(HeartDisease) %>% 
  summarise(n = n())



# Create indices for the training set
train_indices <- createDataPartition(heart_sample_ml$HeartDisease, p = 0.7,
                                     list = FALSE)

# Splitting the data to training and testing
train_set <- heart_sample_ml[train_indices,]
test_set <- heart_sample_ml[-train_indices,] %>% 
  dplyr::select(-HeartDisease)

test_outcome <- heart_sample_ml[-train_indices,] %>% 
  dplyr::select(HeartDisease)


# Initialize the main dataframe
rf_df <- data.frame(i = integer(), 
                    j = integer(), 
                    overall_accuracy = numeric(), 
                    accuracy_yes = numeric())

# Define the range for the number of trees
forest <- c(10, 50, 100, 200, 400, 600, 800, 1000)

# Loop over the features (i) and number of trees (j)
for (feature in seq(1, 10)) {
  for (tree in forest) {
    
    # Random forest model
    rf.model <- randomForest(HeartDisease ~ ., 
                             data = train_set, importance = TRUE, 
                             mtry = feature, ntree = tree)
    
    # Making predictions using the test set
    rf.predict <- predict(rf.model, test_set)
    
    # Calculating the accuracy metrics
    cm <- confusionMatrix(rf.predict, test_outcome$HeartDisease)
    
    # Extracting metrics
    overall_accuracy <- cm$overall["Accuracy"]
    accuracy_yes <- cm$byClass["Specificity"]
    
    # Storing the data
    random_forest_values <- data.frame(feature, tree, overall_accuracy, accuracy_yes)
    
    # Binding to the main dataframe
    rf_df <- rbind(rf_df, random_forest_values)
    
    # Indicating the progress
    print(paste("Number of features:", feature))
    print(paste("Number of trees", tree))
  }
}

# Finding the maximum value for accuracy yes
max_accuracy_yes <- max(rf_df$accuracy_yes, na.rm = TRUE)
max_overall_accuracy <- max(rf_df$overall_accuracy, na.rm = TRUE)

rf_df %>% 
  dplyr::filter(accuracy_yes == max_accuracy_yes)

# Random forest
rf.model <- randomForest(HeartDisease ~ ., 
                         data = train_set, important = TRUE, 
                         mtry = 8, ntree = 800)

# Making predictions using the test set
rf.predict <- predict(rf.model, test_set)

# Number of correct predictions
mean(rf.predict == test_outcome$HeartDisease)

# Table to show the prediction vs actual outcome
table(Predicted = rf.predict, Actual = test_outcome$HeartDisease)

#Plotting to show the importance of variables
varImpPlot(rf.model)

# Storing the importance scores
importance_scores <- importance(rf.model)
importance_df <- as.data.frame(importance_scores) %>% 
  filter(MeanDecreaseGini > 10)

# Extracting variables with gini score of over 1
important_vars <- importance_df %>% 
  dplyr::filter(MeanDecreaseGini > 3)

# Calculating the accuracy metrics
cm <- confusionMatrix(rf.predict, test_outcome$HeartDisease)

# Sensitivity is the accuracy for "Yes"
overall_accuracy <- cm$overall["Accuracy"]
accuracy_yes <- cm$byClass["Specificity"]
print(paste("Overall Accuracy:", overall_accuracy*100,"%"))
print(paste("Accuracy for Yes (Specificity ):", accuracy_yes*100,"%"))



# Saving the random forest model
saveRDS(rf.model, "rf_model.rds")









# Using lasso regression to find the best subset
x <- heart_sample_ml[,c(-1)]
y <- as.numeric(heart_sample_ml$HeartDisease)

# Check the structure
str(x)
dim(x)

# Confirm all columns are numeric
all(apply(x, 2, is.numeric))

# Converting to matrix
x_matrix <- as.matrix(x)

#creating a grid of values ranging from λ = 10^10 to λ = 10^−2
grid <- 10^ seq (10 , -2, length = 100)
grid

#Lasso Model
lasso.mod <- glmnet (x_matrix[train_indices,], y[train_indices], alpha = 1,
                     lambda = grid )

#plotting the lasso model graph
plot(lasso.mod)

plot(lasso.mod, xvar = "lambda", label = TRUE)


#Performing cross-validation
set.seed(1)
cv.out <- cv.glmnet(x_matrix[train_indices,], y[train_indices], alpha = 1)
plot(cv.out)

#choosing the best lambda
(bestlam <- cv.out$lambda.min)


#testing the best lambda on the prediction
lasso.pred <- predict (lasso.mod, s = bestlam ,
                        newx = x_matrix[-train_indices, ])
lasso.pred
#MSE for lasso
mean (( lasso.pred - y[-train_indices] ) ^2)

#chooses the best model, which is model with 11 variables
out <- glmnet (x_matrix , y , alpha = 1, lambda = grid )
lasso.coef <- predict ( out , type = "coefficients",
                        s = bestlam )
# Convert coefficients to a named vector
lasso.coef <- as.matrix(lasso.coef)

# Identify non-zero coefficients
non_zero_coefs <- lasso.coef[(lasso.coef)>abs(0.05), , drop = FALSE]

# Convert to dataframe
non_zero_df <- data.frame(
  Variables = rownames(non_zero_coefs),
  Coefficients = non_zero_coefs[,1],
  row.names = NULL
)

# Remove the intercept (if needed)
non_zero_df <- non_zero_df %>% 
  dplyr::filter(Variables != "(Intercept)")

# Print the result
print(non_zero_df)

# Convert the first column to a vector (not a list)
variable_list <- non_zero_df$Variables



# Logistic regression
# Subset the heart_sample_ml data frame using the list of variables
subset_df <- heart_sample_ml %>% 
  dplyr::select(HeartDisease, all_of(variable_list))

# Checking the dimensions
dim(subset_df)

# Getting the outcome
y_sub <- subset_df$HeartDisease

# Fitting the logistic regression model
glm.fits <- glm(HeartDisease ~ . - GeneralHealthPoor, data = subset_df ,
                family = binomial , subset = train_indices)

# Using the model to make predictions
glm.probs <- predict(glm.fits, subset_df[-train_indices,], type = "response")


# Assigning classes
glm.pred <- ifelse(glm.probs > 0.5, "1", "0")

# Table 
table(Prediction = glm.pred, Actual = y_sub[-train_indices])

# Overall accuracy
mean(glm.pred == y_sub[-train_indices])


# Creating confusion matrix
cm_br <- confusionMatrix(as.factor(glm.pred), y_sub[-train_indices])

# Overall accuracy
cm_br$overall["Accuracy"]

# True Positive Accuracy
cm_br$byClass["Specificity"]


# Saving the logistic regression model
saveRDS(glm.fits, "logistic_model.rds")






# Getting 5000 random samples
random_samples <- heart_encoded_complete %>% 
  slice_sample(n = 100000) %>% 
  dplyr::select(HeartDisease, all_of(variable_list))

# Checking the dimensions of the data
dim(random_samples)

# Making predictions
random.probs <- predict(glm.fits, random_samples, type = "response")

# Classifying the predictions
random.pred <- ifelse(random.probs > 0.5, "1", "0")

# Table to show the accuracy of the model
table(Prediction = random.pred, Actual = random_samples$HeartDisease)

# Confusion matrix
cm_random <- confusionMatrix(as.factor(random.pred), random_samples$HeartDisease)

# Accuracy
cm_random$overall["Accuracy"]

# True positive
cm_random$byClass["Specificity"]



# Exporting training set
write.csv(heart1, "training.csv")





heart1
