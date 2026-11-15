'''
1. Algorithm
1. Import the required Python libraries.
2. Load the Uber fare dataset using Pandas.
3. Perform data preprocessing:
   - Remove missing values.
   - Remove duplicate records.
   - Convert pickup datetime into useful features.
   - Remove invalid fare amounts and geographical coordinates.
4. Detect and remove outliers using the IQR method.
5. Perform correlation analysis and visualize the correlation matrix.
6. Divide the dataset into training and testing sets.
7. Train a Linear Regression model.
8. Train a Random Forest Regression model.
9. Predict fare amounts using both models.
10. Evaluate and compare models using R² Score and RMSE.  '''


# Uber Fare Prediction using Machine Learning

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 2. Load Dataset
df = pd.read_csv("uber.csv")

print("Original Dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# 3. Data Preprocessing

# Remove unnecessary column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Convert pickup_datetime into datetime format
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], errors="coerce", utc=True
)

df.dropna(subset=["pickup_datetime"], inplace=True)

# Extract date and time features
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

# Remove original datetime column
df.drop("pickup_datetime", axis=1, inplace=True)

# Remove invalid fare amounts
df = df[
    (df["fare_amount"] > 0) &
    (df["passenger_count"] > 0) &
    (df["passenger_count"] <= 6)
]

# Validate geographical coordinates
df = df[
    df["pickup_latitude"].between(-90, 90) &
    df["dropoff_latitude"].between(-90, 90) &
    df["pickup_longitude"].between(-180, 180) &
    df["dropoff_longitude"].between(-180, 180)
]

# Remove invalid or zero coordinates
df = df[
    (df["pickup_latitude"] != 0) &
    (df["dropoff_latitude"] != 0) &
    (df["pickup_longitude"] != 0) &
    (df["dropoff_longitude"] != 0)
]

# 4. Calculate Distance using Haversine Formula

def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) *
         np.sin(dlon / 2) ** 2)

    c = 2 * np.arcsin(np.sqrt(a))

    return 6371 * c  # Earth radius in KM

df["distance_km"] = calculate_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

# Remove zero-distance rides
df = df[df["distance_km"] > 0]

# 5. Outlier Detection using IQR Method

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\nLower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# Remove fare outliers
df = df[
    (df["fare_amount"] >= lower_limit) &
    (df["fare_amount"] <= upper_limit)
]

print("\nDataset after preprocessing:")
print(df.head())

# 6. Correlation Analysis

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Uber Fare Correlation Matrix")
plt.tight_layout()
plt.show()

# 7. Feature Selection

X = df.drop(
    ["fare_amount", "key",
     "pickup_longitude", "pickup_latitude",
     "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    errors="ignore"
)

y = df["fare_amount"]

# 8. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 9. Linear Regression Model

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_r2 = r2_score(y_test, lr_predictions)

lr_rmse = np.sqrt(
    mean_squared_error(y_test, lr_predictions)
)

# 10. Random Forest Regression Model

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_predictions)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_predictions)
)

# 11. Compare Model Performance

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [lr_r2, rf_r2],
    "RMSE": [lr_rmse, rf_rmse]
})

print("\nModel Performance Comparison:")
print(results.to_string(index=False))

# 12. Visualize Model Comparison

results.set_index("Model")[["R2 Score", "RMSE"]].plot(
    kind="bar",
    figsize=(9, 5),
    subplots=True,
    legend=False
)

plt.tight_layout()
plt.show()

# 13. Final Result

if rf_r2 > lr_r2:
    print("\nRandom Forest Regression performs better.")
else:
    print("\nLinear Regression performs better.")


'''
Important Concepts

Outlier detection using IQR:

IQR=Q3-Q1

- Lower Limit = Q1 - 1.5 x IQR
- Upper Limit = Q3 + 1.5 x IQR

Linear Regression: Finds a linear relationship between input features and fare amount.



Random Forest Regression: Combines multiple decision trees to predict fare amounts and 
capture nonlinear relationships.
'''