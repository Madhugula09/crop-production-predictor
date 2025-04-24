import pandas as pd
import mysql.connector
import numpy as np

# Set file path
file_path = "C:/Users/madhugula padmavathi/Downloads/FAOSTAT_data - FAOSTAT_data_en_12-29-2024.csv"

# Load CSV
df = pd.read_csv(file_path)

# Keep relevant columns
df = df[["Area", "Item", "Element", "Year", "Unit", "Value", "Flag", "Flag Description"]]

# Rename columns
df.rename(columns={
    "Area": "country",
    "Item": "crop",
    "Element": "element",
    "Year": "year",
    "Unit": "unit",
    "Value": "value",
    "Flag": "flag",
    "Flag Description": "flag_description"
}, inplace=True)

# Pivot table
df_pivot = df.pivot_table(
    index=["country", "crop", "year", "unit", "flag", "flag_description"],
    columns="element",
    values="value",
    aggfunc="first"
).reset_index()

# Rename pivoted columns to match MySQL
df_pivot.rename(columns={
    "Area harvested": "area_harvested",
    "Yield": "yield",
    "Production": "production"
}, inplace=True)

# Fill NaN values in critical columns with the median
df_pivot['area_harvested'] = df_pivot['area_harvested'].fillna(df_pivot['area_harvested'].median())
df_pivot['yield'] = df_pivot['yield'].fillna(df_pivot['yield'].median())
df_pivot['production'] = df_pivot['production'].fillna(df_pivot['production'].median())

# Now let's print the count of NaN values again to ensure the NaNs were handled
nan_count = df_pivot.isnull().sum()
print("NaN Values Count Per Column After Filling:")
print(nan_count)

# Check total NaNs in the DataFrame after filling
total_nan_after_fill = df_pivot.isnull().sum().sum()
print(f"\nTotal NaN values in the DataFrame after filling: {total_nan_after_fill}")

# Optionally, you can drop rows with NaN values in the 'area_harvested', 'production', 'yield' columns if they are still problematic
df_pivot.dropna(subset=["area_harvested", "production", "yield"], how="any", inplace=True)

# ✅ Save cleaned data
df_pivot.to_csv("cleaned_crop_data_final.csv", index=False)
print("📁 Cleaned data saved to 'cleaned_crop_data_final.csv'")

# ✅ Connect to MySQL and insert data
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",        
        password="Padmavathi@09",    
        database="crop_production_db"     
    )
    cursor = conn.cursor()
    print("🔌 Connected to MySQL!")

    # ✅ Create table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS crop_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        country VARCHAR(255),
        crop VARCHAR(255),
        year INT,
        area_harvested FLOAT,
        yield FLOAT,
        production FLOAT,
        unit VARCHAR(100),
        flag VARCHAR(10),
        flag_description VARCHAR(255)
    )
    """
    cursor.execute(create_table_query)
    print("📦 Table 'crop_data' checked/created.")

    # ✅ Insert query
    insert_query = """
        INSERT INTO crop_data (
            country, crop, year,
            area_harvested, yield, production,
            unit, flag, flag_description
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # Prepare data for insertion
    data_to_insert = df_pivot[[
        "country", "crop", "year",
        "area_harvested", "yield", "production",
        "unit", "flag", "flag_description"
    ]].values.tolist()

    # Insert the data into MySQL
    cursor.executemany(insert_query, data_to_insert)
    conn.commit()
    print(f"✅ {cursor.rowcount} rows inserted into 'crop_data'.")

except mysql.connector.Error as err:
    print(f"❌ MySQL Error: {err}")
finally:
    if 'cursor' in locals(): cursor.close()
    if 'conn' in locals(): conn.close()
