## crop-production-predictor
A model to predict crop production using machine learning and agricultural data with three different models.

## 🛠️ Tech Stack
. Language: Python

. Libraries: pandas, scikit-learn, matplotlib, seaborn, streamlit, mysql-connector-python

. Database: MySQL

. Visualization: Matplotlib, Seaborn, Streamlit

## Features
-- Data Preprocessing: Removes unnecessary columns, handles missing values, and cleans numerical data.

![image](https://github.com/user-attachments/assets/5c84fe9c-8e3b-43a6-b144-13aae2cd82e9)


## 📊 Database Integration

-- Stored processed data in a MySQL database (crop_data table).

## 📈 Evaluation Metrics and Outlier Removal 

-- Compared models using RMSE, MAE, and R².

Filtered out invalid entries like nulls and non-positive values to ensure clean data for modeling.

MAE (Mean Absolute Error): Measures the average magnitude of errors in predictions.

MSE (Mean Squared Error): Measures the average of the squared differences between predicted and actual values.

RMSE (Root Mean Squared Error): The square root of MSE, representing the error in the same units as the target variable.

R² (Coefficient of Determination): Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

![image](https://github.com/user-attachments/assets/c2b1b7bb-dda1-403f-a79c-b5ed355d0bdb)

## 🤖 ML Models: Trained and evaluated three models:

![image](https://github.com/user-attachments/assets/08b32746-121c-4932-8294-b81360fc9138)

Linear Regression: A simple and interpretable model used to predict crop production based on linear relationships between features.

Decision Tree Regressor: A non-linear model that splits data into smaller subsets based on feature values.

Random Forest Regressor: An ensemble model that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Machine Learning Models: Implements three models for crop production prediction:

-- Prediction: Estimates crop production based on user inputs for area harvested and yield.

-- Model Evaluation: Provides evaluation metrics such as MAE, MSE, RMSE, and R² to assess model performance.

## 📄 Prediction History 

Prediction History: Allows users to track and download the history of predictions.

![image](https://github.com/user-attachments/assets/aa8555d6-ede2-44f8-9381-e8144d33b836)


## 🌐 Interactive Dashboard

-- Streamlit app to filters 

![image](https://github.com/user-attachments/assets/797aaf67-e80d-4a5f-a6ab-6500ede123e5)

1. Market Price Forecasting and Precision Farming

2. Agro-Technology Solutions and Evaluation & Insights Section


