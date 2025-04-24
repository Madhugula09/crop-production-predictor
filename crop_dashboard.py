import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings
import random

warnings.filterwarnings("ignore")

# --- Load Data from MySQL ---
def load_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Padmavathi@09",
        database="crop_production_db"
    )
    query = "SELECT * FROM crop_data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- Preprocessing ---
def preprocess_data(df):
    df = df.copy()
    df = df.drop(columns=['id', 'unit', 'flag', 'flag_description'], errors='ignore')
    df = df.rename(columns={'yied': 'yield'})
    for col in ['area_harvested', 'yield', 'production']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['area_harvested', 'yield', 'production'])
    df = df[(df['area_harvested'] > 0) & (df['yield'] > 0) & (df['production'] > 0)]
    for col in ['area_harvested', 'yield', 'production']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

# --- Outlier Detection ---
def detect_outliers(df):
    outliers_summary = {}
    for col in ['area_harvested', 'yield', 'production']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outliers_summary[col] = len(outliers)
    return outliers_summary

# --- Train Model with Cross-validation ---
def train_model(df, model_type):
    if len(df) < 2:
        return None, None, None, None, None, None, None, None, None

    X = df[['area_harvested', 'yield']]
    y = df['production']

    model = None
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Decision Tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    cv_value = min(5, len(X))
    if cv_value < 2:
        st.warning("Not enough data for cross-validation. At least 2 samples required.")
    else:
        cv_scores = cross_val_score(model, X, y, cv=cv_value, scoring='neg_mean_squared_error')
        mean_cv_mse = -cv_scores.mean()

    return model, preds, y_test, mae, mse, rmse, r2, model_type, mean_cv_mse

# --- Streamlit App ---
st.set_page_config(page_title="Crop Production Prediction", layout="wide")
st.title("🌾 Crop Production Prediction Dashboard")

data = load_data()
st.sidebar.header("🔍 Filter the Data")
countries = st.sidebar.multiselect("🌍 Country", sorted(data['country'].dropna().unique()))
crops = st.sidebar.multiselect("🌽 Crop", sorted(data['crop'].dropna().unique()))
years = st.sidebar.multiselect("📅 Year", sorted(data['year'].dropna().unique()))
model_option = st.sidebar.selectbox("🧠 Select Model", ["Linear Regression", "Decision Tree", "Random Forest"])

filtered_df = data.copy()
if countries:
    filtered_df = filtered_df[filtered_df['country'].isin(countries)]
if crops:
    filtered_df = filtered_df[filtered_df['crop'].isin(crops)]
if years:
    filtered_df = filtered_df[filtered_df['year'].isin(years)]

filtered_df_cleaned = preprocess_data(filtered_df)
st.subheader("🧹 Cleaned Data")
st.dataframe(filtered_df_cleaned)

st.subheader("📛 Outlier Summary")
st.write(detect_outliers(filtered_df_cleaned))

model, predictions, y_test, mae, mse, rmse, r2, model_used, mean_cv_mse = train_model(filtered_df_cleaned, model_option)

if model and predictions is not None:
    st.success(f"✅ {model_used} model trained successfully!")
    st.metric("📏 MAE", f"{mae:,.2f}")
    st.metric("📐 MSE", f"{mse:,.2f}")
    st.metric("🔢 RMSE", f"{rmse:,.2f}")
    st.metric("📊 R²", f"{r2:.4f}")
    st.metric("📅 Cross-validated MSE", f"{mean_cv_mse:,.2f}")

    st.subheader("📊 Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(6, 4))  # You can change width and height here
    ax.scatter(y_test, predictions, alpha=0.6, color='blue', label='Predictions')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
    ax.set_xlabel("Actual Production")
    ax.set_ylabel("Predicted Production")
    ax.set_title(f"{model_used} - Actual vs Predicted")
    ax.legend(fontsize=3)
    st.pyplot(fig)

st.subheader("🏆 Compare All Models on This Data")
results = []
for m in ["Linear Regression", "Decision Tree", "Random Forest"]:
    model, preds, y_test, mae, mse, rmse, r2, name, mean_cv_mse = train_model(filtered_df_cleaned, m)
    if model:
        results.append({"Model": name, "R²": r2, "MAE": mae, "MSE": mse, "RMSE": rmse, "Cross-validated MSE": mean_cv_mse})
if results:
    results_df = pd.DataFrame(results).sort_values(by=["R²", "MAE", "MSE", "RMSE"], ascending=[False, True, True, True])
    st.dataframe(results_df)
    best_model = results_df.iloc[0]
    st.success(f"🥇 Best model based on all values (R², MAE, MSE, RMSE) is: **{best_model['Model']}**")
    st.write(f"📊 **R²**: {best_model['R²']:.4f}, 📏 **MAE**: {best_model['MAE']:.2f}, 📐 **MSE**: {best_model['MSE']:.2f}, 🔢 **RMSE**: {best_model['RMSE']:.2f}")

st.subheader("🔮 Predict Crop Production")
area_input = st.number_input("📐 Area Harvested (hectares)", min_value=1.0, value=1.0)
yield_input = st.number_input("🧪 Yield (kg/hectare)", min_value=1.0, value=1.0)
if model:
    prediction = model.predict([[area_input, yield_input]])[0]
    st.success(f"🌟 Predicted Production using **{model_used}**: **{prediction:,.2f} tons**")
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({"Model": model_used, "Area Harvested": area_input, "Yield": yield_input, "Predicted Production": prediction})

if "history" in st.session_state and st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.subheader("📄 Prediction History")
    st.dataframe(history_df)
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Prediction History", csv, "prediction_history.csv", "text/csv")
    st.subheader("📈 Prediction Trends")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=history_df, x="Model", y="Predicted Production", ax=ax2, ci=None)
    ax2.set_title("Model-wise Predicted Production")
    st.pyplot(fig2)

# ------------------- New Sections -------------------
st.header("🧠 Advanced Features")

# Market Price Forecasting
st.subheader("📈 Market Price Forecasting")
if crops and years:
    for crop in crops:
        for year in years:
            st.write(f"**{crop} ({year})** - Estimated Price: ₹{random.randint(1000, 5000)} per ton")
else:
    st.info("Select Crop and Year to see price forecasts.")

# Precision Farming Tips
st.subheader("🌾 Precision Farming")
if crops:
    for crop in crops:
        st.markdown(f"**{crop}**: Use soil sensors, drone imagery, and precision irrigation for yield boost.")
else:
    st.info("Select crop(s) to get precision farming tips.")

# Agro-Technology Solutions
st.subheader("🤖 Agro-Technology Solutions")
if countries:
    for country in countries:
        st.write(f"**{country}**: Recommend IoT sensors, satellite monitoring, and AI-based pest detection.")
else:
    st.info("Select country to view relevant agri-tech solutions.")


# Evaluation & Insights
st.subheader("📊 Evaluation & Insights")

# Ensure your DataFrame isn't empty and has required columns
required_columns = {'crop', 'country', 'production'}

if (
    isinstance(filtered_df_cleaned, pd.DataFrame)
    and not filtered_df_cleaned.empty
    and required_columns.issubset(filtered_df_cleaned.columns)
):
    try:
        # Drop missing values in production to avoid NaN issues
        df = filtered_df_cleaned.dropna(subset=['production'])

        best_crop = df.groupby('crop')['production'].mean().idxmax()
        top_country = df.groupby('country')['production'].mean().idxmax()

        st.write(f"🌟 **Highest average production crop**: {best_crop}")
        st.write(f"🌍 **Top producing country (avg)**: {top_country}")
    except Exception as e:
        st.error(f"⚠️ An error occurred while processing insights: {e}")
else:
    st.warning("⚠️ No data available for insights please select check with filters ")
