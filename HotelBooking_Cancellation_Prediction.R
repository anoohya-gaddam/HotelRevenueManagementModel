hotel_data <- read.csv("hotel_bookings.csv")
View(hotel_data)
# Next use the function summary to inspect the data
summary(hotel_data)
missing_counts <- colSums(is.na(hotel_data))

# Print the missing value counts
print(missing_counts)

# Impute the missing data.
hotel_data$children[is.na(hotel_data$children)] <- 0
# Run the summary function again. Now you see that no demographic/county columns have NA entries.
summary(hotel_data)

# Check the NULL values in the dataset
count_null <- function(x) sum(x == "NULL")

# Apply the function to each column of hotel_data
null_counts <- sapply(hotel_data, count_null)
null_counts

# Replace NULL value in country column with "UKNWN"
hotel_data$country <- ifelse(hotel_data$country == "NULL", "UKNWN", hotel_data$country)

drop <- c("company", "reservation_status", "arrival_date_year",
          "reservation_status_date" , "arrival_date_week_number", "stays_in_weekend_nights", "agent")
# Drop countries since there are too many NULLs
# Drop columns that are not relevant or to prevent data leakage
model_data <- hotel_data[,!(names(hotel_data) %in% drop)]
model_data$is_agent <- ifelse(hotel_data$agent == 'NULL', 0, 1)
model_data$is_canceled <- factor(model_data$is_canceled)
View(model_data)
#-------- End of data cleaning -----------

#-------- Data Visualization -----------

# Load required libraries
library(ggplot2)
library(dplyr)

## Calculate the booking count and percentage of cancellations by hotel type
cancellation_summary <- model_data %>%
  group_by(hotel, is_canceled) %>%
  summarise(count = n()) %>%
  group_by(hotel) %>%
  mutate(percentage = round(count / sum(count) * 100, 2))

# Create a grouped bar plot
ggplot(cancellation_summary, aes(x = hotel, fill = factor(is_canceled))) +
  geom_bar(aes(y = count), position = "dodge", stat = "identity") +
  geom_text(aes(y = count, label = paste0(percentage, "%")), vjust = -0.5, position = position_dodge(0.9)) +
  labs(title = "Booking Count and Cancellation Percentage by Hotel Type",
       x = "Hotel Type",
       y = "Count of Hotel Bookings",
       fill = "Cancellation Status") +
  scale_fill_manual(values = c("0" = "blue", "1" = "light blue")) +
  theme_minimal()

## Create a bar plot to visualize the relationship between is_canceled and is_repeated_guest
ggplot(model_data, aes(x = factor(is_repeated_guest), fill = factor(is_canceled))) +
  geom_bar(position = "dodge") +
  labs(
    title = "Cancellation Count by Repeated Guest Status",
    x = "Is Repeated Guest",
    y = "Count of Hotel Bookings",
    fill = "Cancellation Status"
  ) +
  scale_fill_manual(values = c("0" = "blue", "1" = "light blue")) +
  theme_minimal()

library(ggplot2)
library(dplyr)

month_order <- c(
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"
)

city_hotel_data <- model_data %>%
  filter(hotel == "City Hotel")

city_hotel_data <- model_data %>%
  mutate(arrival_date_month = factor(arrival_date_month, levels = month_order))

# Create the bar plot with the ordered months
city_plot <- ggplot(city_hotel_data, aes(x = arrival_date_month, y = after_stat(count), fill = factor(is_canceled))) +
  geom_bar() +
  labs(
    title = "City Hotel",
    x = "Month",
    y = "Count of Hotel Bookings",
    fill = "Cancellation Status"
  ) +
  scale_fill_manual(values = c("0" = "pink", "1" = "red")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)
  )


resort_hotel_data <- model_data %>%
  filter(hotel == "Resort Hotel")

resort_hotel_data <- resort_hotel_data %>%
  mutate(arrival_date_month = factor(arrival_date_month, levels = month_order))

# Create the bar plot with the ordered months
resort_plot <- ggplot(resort_hotel_data, aes(x = arrival_date_month, y = after_stat(count), fill = factor(is_canceled))) +
  geom_bar() +
  labs(
    title = "Resort Hotel",
    x = "Month",
    y = "Count of Hotel Bookings",
    fill = "Cancellation Status"
  ) +
  scale_fill_manual(values = c("0" = "pink", "1" = "red")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Display the plot
print(city_plot/resort_plot)


## Create the box plot between month and ADR
# Filter the data
data <- model_data[model_data$is_canceled == 0,]
# Load the necessary library
library(ggplot2)

month_order <- c(
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"
)

# Convert "arrival_date_month" to a factor with the specified order
model_data$arrival_date_month <- factor(model_data$arrival_date_month, levels = month_order)

# Create the box plot
p <- ggplot(data, aes(x = arrival_date_month, y = adr, fill = hotel)) +
  geom_boxplot() +
  labs(title = "Box Plot of ADR by Month",
       x = "month",
       y = "ADR") +
  theme_minimal() +
  theme(legend.title = element_blank())
options(repr.plot.width=9, repr.plot.height=6, repr.plot.res=100)
print(p)



## Create box plot between is_cancelled and ADR
ggplot(model_data, aes(x = is_canceled, y = adr)) +
  geom_boxplot() +
  ylim(0, 600)

ggplot(model_data, aes(x = market_segment)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "market_segment", y = "Count")
#-------- End of data visualization -----------



#install packages
install.packages("caret")
install.packages("e1071")
install.packages("kernlab")

#-------- Prediction in Binary Case (Cancel = 0 or 1) -----------

## Logistic Regression
library(caret)

#specify the cross-validation method
ctrl <- trainControl(method = "cv", number = 10)

#fit a regression model and use k-fold CV to evaluate performance
reg_model <- train(is_canceled~., data = model_data, method = "glm", family = "binomial", trControl = ctrl)

#view summary of k-fold CV
print(reg_model)

##SVM

library(kernlab)
ctrl <- trainControl(method = "cv", number = 10)
svm_cv_model <- train(is_canceled ~ ., data = model_data, method = "svmRadial", trControl = ctrl, tuneGrid = data.frame(C = 10, sigma = 0.001))
print(svm_cv_model)


## Classification Tree
install.packages("rpart.plot")
levels(model_data$is_canceled)
model_data$is_canceled <- make.names(model_data$is_canceled)
library(caret)
library(rpart)

# Check and rename levels if needed
model_data$is_canceled <- make.names(model_data$is_canceled)

# Define the control parameters
ctrl <- trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE)

# Create and train the rpart model
tree.model <- train(is_canceled ~ ., data = model_data, method = "rpart", trControl = ctrl)

# Print the results
print(tree.model)

## XGBoost
#install.packages("xgboost")
library(caret)
ctrl <- trainControl(method = "cv", number = 10)


# Fit a random forest model and use k-fold CV to evaluate performance
param <-  data.frame(nrounds=c(100), max_depth = c(2),eta =c(0.3),gamma=c(0),
                     colsample_bytree=c(0.8),min_child_weight=c(1),subsample=c(1))
fit.xgbTree <- train(is_canceled ~ ., data = model_data, method="xgbTree",
                     metric="Accuracy", trControl=ctrl,tuneGrid=param)
# View summary of k-fold CV for the random forest model
print(fit.xgbTree$results$Accuracy)
print(fit.xgbTree)


### Prediction using XGBoost
predictions <- predict(fit.xgbTree, newdata = model_data)

# Calculate accuracy
accuracy <- mean(predictions == model_data$is_canceled)
print(paste("Accuracy:", accuracy, "%"))


#-------- Clustering -----------


#K-means

hotel_numeric <- model_data[,c("is_canceled","lead_time","arrival_date_day_of_month","adults","children","babies","is_repeated_guest","adr", "is_agent")]
hotel_kmeans <- kmeans(hotel_numeric,4,nstart=10)
### This is using k-means on the data "simple"
### and setting k=4 to create 4 clusters
### more on the number of clusters later
colorcluster <- 1+hotel_kmeans$cluster
### hotel_kmeans$cluster is a vector
### telling the cluster each observation belongs to.
### we can associate a color for each cluster (and therefore for each observartion)
### in the vector colorcluster
###
plot(hotel_numeric, col = colorcluster)
points(hotel_kmeans$centers, col = 1, pch = 24, cex = 1.5, lwd=1, bg = 2:5)
### The command
hotel_kmeans$centers
## displays the k centers (good to interpret)
hotel_kmeans$size

