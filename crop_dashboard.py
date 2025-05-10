import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import numpy as np

# ------------------- DB Connection -------------------
def load_data():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Padmavathi@09",
        database="crop_production_db"
    )
    query = "SELECT * FROM crop_data"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

# ------------------- App UI -------------------
st.title("🌾 Crop Production Predictor")

# Load data
df = load_data()

# Drop rows with missing critical values
df = df.dropna(subset=["country", "crop", "year", "area_harvested", "yield", "production"])
df = df[df["yield"].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
df["yield"] = df["yield"].astype(float)

st.subheader("🧹 Full Cleaned Dataset from MySQL")
st.dataframe(df, use_container_width=True)

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
selected_crop = st.sidebar.selectbox("Crop", sorted(df["crop"].unique()))
selected_country = st.sidebar.selectbox("Country", sorted(df["country"].unique()))
year_range = st.sidebar.slider("Year Range", int(df["year"].min()), int(df["year"].max()), (2019, 2023))

# Filtered data
filtered_df = df[
    (df["crop"] == selected_crop) &
    (df["country"] == selected_country) &
    (df["year"].between(year_range[0], year_range[1]))
]

st.subheader("📊 Filtered Data")
st.write(filtered_df)

# ----- EDA -----------

st.subheader("📊 Exploratory Data Analysis (EDA)")

# Ensure essential columns exist
if not {'country', 'crop', 'year', 'area_harvested', 'yield', 'production'}.issubset(filtered_df.columns):
    st.warning("Essential columns missing for full EDA.")
else:
    # 1. Crop Types Distribution
    st.markdown("### 🌱 Crop Distribution")
    crop_counts = filtered_df['crop'].value_counts().reset_index()
    crop_counts.columns = ['Crop', 'Count']
    st.plotly_chart(px.bar(crop_counts, x='Crop', y='Count', title='Crop Type Distribution', color='Count'))

    # 2. Geographical Crop Distribution
    st.markdown("### 🌍 Geographical Crop Distribution")
    region_crop = filtered_df.groupby(['country', 'crop'])['production'].sum().reset_index()
    fig = px.sunburst(region_crop, path=['country', 'crop'], values='production',
    title='Production by Region and Crop')
    st.plotly_chart(fig)

    # 3. Yearly Trends in Area, Yield, and Production
    st.markdown("### 📅 Yearly Trends")
    yearly = filtered_df.groupby('year')[['area_harvested', 'yield', 'production']].mean().reset_index()
    fig = px.line(yearly, x='year', y=['area_harvested', 'yield', 'production'],
    markers=True, title='Yearly Trends: Area, Yield, Production')
    st.plotly_chart(fig)

    # 4. Growth Analysis per Crop
    st.markdown("### 📈 Crop-wise Growth Trends")
    crop_yearly = filtered_df.groupby(['year', 'crop'])['production'].sum().reset_index()
    fig = px.line(crop_yearly, x='year', y='production', color='crop',
    title='Production Growth per Crop')
    st.plotly_chart(fig)

    # 5. Area vs Yield Relationship
    st.markdown("### 🌾 Area Harvested vs Yield")
    fig = px.scatter(filtered_df, x='area_harvested', y='yield', color='crop',
    size='production', hover_data=['country'],
    title='Area Harvested vs Yield')
    st.plotly_chart(fig)

    # 6. Correlation Heatmap
    st.markdown("### 🔗 Input-Output Correlation")
    corr_df = filtered_df[['area_harvested', 'yield', 'production']].corr()
    st.dataframe(corr_df.style.background_gradient(cmap='YlGnBu'))

    # 7. Yield Comparison Across Crops
    st.markdown("### ⚖️ Yield per Crop")
    yield_crop = filtered_df.groupby('crop')['yield'].mean().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(yield_crop, x='crop', y='yield',
    title='Average Yield by Crop', color='yield'))

    # 8. Production Comparison Across Regions
    st.markdown("### 🏭 Production Across Regions")
    prod_region = filtered_df.groupby('country')['production'].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(prod_region, x='country', y='production',
    title='Total Production by Region', color='production'))

    # 9. Productivity Ratio (Production / Area)
    st.markdown("### 📊 Productivity Ratio")
    filtered_df['productivity_ratio'] = filtered_df['production'] / filtered_df['area_harvested']
    prod_ratio = filtered_df.groupby('crop')['productivity_ratio'].mean().reset_index()
    st.plotly_chart(px.bar(prod_ratio.sort_values('productivity_ratio', ascending=False),
    x='crop', y='productivity_ratio',
    title='Productivity Ratio per Crop'))

    # 10. Outlier Detection (Boxplot)
    st.markdown("### ❗ Outlier Detection in Yield")
    fig = px.box(filtered_df, x='crop', y='yield', title='Yield Distribution by Crop (Box Plot)')
    st.plotly_chart(fig)

# ------------------- Model Training -------------------
if len(filtered_df) < 10:
    st.warning("Not enough data for training. Please adjust filters.")
else:
    X = filtered_df[["year", "area_harvested", "yield"]]
    y = filtered_df["production"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(random_state=42)
    dt_model = DecisionTreeRegressor(random_state=42)

    # Train models
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)

    # ------------------- Model Evaluation -------------------
    models = {
        "Linear Regression": lr_model,
        "Random Forest": rf_model,
        "Decision Tree": dt_model
    }

    st.subheader("📈 Model Evaluation on Test Data")
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        results.append({
            "Model": name,
            "R² Score": round(r2, 4),
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2)
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Best model based on R²
    best_model_row = results_df.sort_values("R² Score", ascending=False).iloc[0]
    best_model_name = best_model_row["Model"]

    st.success(f"✅ **Best model based on test data R² Score:** {best_model_name}")

    # ------------------- Future Prediction -------------------
    st.subheader("📅 Predict Future Production")

    future_year = st.number_input("Enter Future Year", min_value=2024, max_value=2100, value=2032)
    future_area = st.number_input("Enter Area Harvested (hectares)", min_value=1, value=100)
    future_yield = st.number_input("Enter Yield (tons/hectare)", min_value=0.0, value=2.5, step=0.1)

    # Estimated price for future prediction
    future_price_per_ton = st.number_input("Estimated Future Price per Ton (USD)", value=250.0, key="future_price_per_ton")
    if st.button("Predict Future Production"):
        future_input = [[future_year, future_area, future_yield]]
        st.markdown(f"### 📌 Predicting for year {future_year} and area harvested {future_area} hectares")

        for name, model in models.items():
            future_pred = model.predict(future_input)[0]
            future_pred = max(0, future_pred)  # Ensure non-negative prediction
            st.write(f"**{name} Prediction:** {future_pred:.2f} tons")
            estimated_value = future_pred * future_price_per_ton
            st.write(f"Estimated Value (in USD): ${estimated_value:.2f}")

        st.info(f"✅ Best model for training data: **{best_model_name}**")

    # ------------------- Past Prediction -------------------
    st.subheader("📅 Predict Past Production")

    past_year = st.number_input("Enter Past Year", min_value=2000, max_value=2023, value=2020)
    past_area = st.number_input("Enter Past Area Harvested (hectares)", min_value=1, value=100)
    past_yield = st.number_input("Enter Past Yield (tons/hectare)", min_value=0.0, value=2.5, step=0.1)

    # Estimated price for past prediction
    past_price_per_ton = st.number_input("Estimated Price per Ton (USD)", value=200.0, key="past_price_per_ton")

    if st.button("Predict Past Production"):
        past_input = [[past_year, past_area, past_yield]]
        st.markdown(f"### 📌 Predicting for year {past_year} and area harvested {past_area} hectares")

        for name, model in models.items():
            past_pred = model.predict(past_input)[0]
            st.write(f"**{name} Prediction:** {past_pred:.2f} tons")
            estimated_value = past_pred * past_price_per_ton
            st.write(f"Estimated Value (in USD): ${estimated_value:.2f}")

        st.info(f"✅ Best model for past prediction: **{best_model_name}**")

crops = sorted(filtered_df['crop'].dropna().unique().tolist())
years = sorted(filtered_df['year'].dropna().unique().tolist())
countries = sorted(filtered_df['country'].dropna().unique().tolist())

# Precision Farming Tips
st.subheader("🌿 Precision Farming Tips")
if crops:
    for crop in crops:
        st.markdown(f"**{crop}**: Use drone mapping, soil sensors, and AI irrigation to boost {crop} yield.")
else:
    st.info("Select crop(s) to get tips.")

# Agro-Tech Solutions
st.subheader("🤖 Agro-Tech Solutions by Country")
if countries:
    for country in countries:
        st.write(f"**{country}**: Use IoT sensors, weather analytics & AI-driven pest monitoring.")
else:
    st.info("Select country to view agri-tech recommendations.")
