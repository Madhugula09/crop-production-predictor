## crop-production-predictor
A model to predict crop production using machine learning and agricultural data with three different models.
## Features
Data Preprocessing: Removes unnecessary columns, handles missing values, and cleans numerical data.

Outlier Detection: Uses the IQR method to identify and remove outliers.

Machine Learning Models: Implements three models for crop production prediction:

-- Linear Regression

-- Decision Tree Regressor

-- Random Forest Regressor

Prediction: Estimates crop production based on user inputs for area harvested and yield.

Model Evaluation: Provides evaluation metrics such as MAE, MSE, RMSE, and R² to assess model performance.

Prediction History: Allows users to track and download the history of predictions.
Models Used
Linear Regression: A simple and interpretable model used to predict crop production based on linear relationships between features.

Decision Tree Regressor: A non-linear model that splits data into smaller subsets based on feature values.

Random Forest Regressor: An ensemble model that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

## Evaluation Metrics

MAE (Mean Absolute Error): Measures the average magnitude of errors in predictions.

MSE (Mean Squared Error): Measures the average of the squared differences between predicted and actual values.

RMSE (Root Mean Squared Error): The square root of MSE, representing the error in the same units as the target variable.

R² (Coefficient of Determination): Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
