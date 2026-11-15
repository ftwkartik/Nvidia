'''
1. Algorithm
1. Import the required Python libraries.
2. Load the sales_data_sample.csv dataset.
3. Perform data preprocessing:
   - Remove duplicate and missing records.
   - Select relevant numerical features.
   - Calculate customer-level purchasing information.
4. Normalize the selected features using StandardScaler.
5. Apply K-Means clustering for different values of K.
6. Calculate Within-Cluster Sum of Squares (WCSS) for each K.
7. Plot the Elbow Curve to identify the optimal number of clusters.
8. Train the K-Means model using the selected K value.
9. Assign cluster labels to each customer.
10. Visualize the customer segments using a scatter plot.
11. Analyze the characteristics of each customer segment. '''


# Customer Segmentation using K-Means Clustering

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 2. Load Dataset
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

print("Original Dataset:")
print(df.head())

# 3. Data Preprocessing

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Remove records without customer information
df.dropna(subset=["CUSTOMERNAME"], inplace=True)

# Convert ORDERDATE into datetime format
df["ORDERDATE"] = pd.to_datetime(
    df["ORDERDATE"], errors="coerce"
)

df.dropna(subset=["ORDERDATE"], inplace=True)

# 4. Customer-Level Feature Engineering

customer_data = df.groupby("CUSTOMERNAME").agg({
    "SALES": "sum",
    "ORDERNUMBER": "nunique",
    "QUANTITYORDERED": "sum",
    "PRICEEACH": "mean"
}).reset_index()

# Rename columns
customer_data.columns = [
    "Customer",
    "Total_Sales",
    "Total_Orders",
    "Total_Quantity",
    "Average_Price"
]

print("\nCustomer Data:")
print(customer_data.head())

# 5. Select Features for Clustering
X = customer_data[
    ["Total_Sales", "Total_Orders",
     "Total_Quantity", "Average_Price"]
]

# Apply log transformation to reduce skewness
X = np.log1p(X)

# 6. Normalize Features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# 7. Find Optimal Clusters using Elbow Method

wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# 8. Apply K-Means Clustering
# Select K based on the Elbow Curve
optimal_k = 3

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

customer_data["Cluster"] = kmeans.fit_predict(X_scaled)

# 9. Display Customer Segments
print("\nCustomer Segmentation Results:")
print(customer_data.head(10))

# 10. Analyze Each Cluster
print("\nCluster Analysis:")

print(
    customer_data.groupby("Cluster")[
        ["Total_Sales", "Total_Orders",
         "Total_Quantity", "Average_Price"]
    ].mean().round(2)
)

# 11. Visualize Customer Segments
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=customer_data,
    x="Total_Sales",
    y="Total_Orders",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Total Sales")
plt.ylabel("Total Orders")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# 12. Final Result
print("\nOptimal Number of Clusters:", optimal_k)
print("Customer Segmentation Completed Successfully!")
