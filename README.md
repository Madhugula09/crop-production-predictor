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
-- EDA Visualization.

![image](https://github.com/user-attachments/assets/f57e420a-aca6-4c96-8b2e-7bc66ec93f29)


## 📈 Evaluation Metrics and Outlier Removal 

-- Compared models using RMSE, MAE, and R².

Filtered out invalid entries like nulls and non-positive values to ensure clean data for modeling.

MAE (Mean Absolute Error): Measures the average magnitude of errors in predictions.

MSE (Mean Squared Error): Measures the average of the squared differences between predicted and actual values.

RMSE (Root Mean Squared Error): The square root of MSE, representing the error in the same units as the target variable.

R² (Coefficient of Determination): Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

![image](https://github.com/user-attachments/assets/c2b1b7bb-dda1-403f-a79c-b5ed355d0bdb)

## 🤖 ML Models: Trained and evaluated three models:

Linear Regression: A simple and interpretable model used to predict crop production based on linear relationships between features.

Decision Tree Regressor: A non-linear model that splits data into smaller subsets based on feature values.

Random Forest Regressor: An ensemble model that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Machine Learning Models: Implements three models for crop production prediction:

-- Prediction: Estimates crop production based on user inputs for area harvested and yield.

-- Model Evaluation: Provides evaluation metrics such as MAE, MSE, RMSE, and R² to assess model performance.

## 🌐 Interactive Dashboard

-- Streamlit app to filters 

-- Future predictions

![image](https://github.com/user-attachments/assets/09e23d68-3ba6-4295-b533-7b3859738e94)

-- Past predictions

![image](https://github.com/user-attachments/assets/4f2ccbed-9a27-4267-bfc5-59de19e6cbaf)




