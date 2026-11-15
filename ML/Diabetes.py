'''
Algorithm
1. Import the required Python libraries.
2. Load the diabetes.csv dataset using Pandas.
3. Perform data preprocessing:
   - Check and remove duplicate records.
   - Handle missing and invalid values.
   - Separate input features and target variable.
4. Split the dataset into training and testing sets (80:20).
5. Normalize the features using StandardScaler.
6. Create a KNN classifier with K = 5.
7. Train the KNN model using training data.
8. Predict diabetes outcomes using testing data.
9. Generate the confusion matrix.
10. Calculate Accuracy, Error Rate, Precision and Recall.
11. Display and analyze the model performance. '''


# Diabetes Prediction using KNN

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)

# 2. Load Dataset
df = pd.read_csv("diabetes.csv")

print("Original Dataset:")
print(df.head())

# 3. Data Preprocessing

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Replace invalid zero values with NaN
columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in columns:
    df[col] = df[col].replace(0, np.nan)

# Handle missing values using median
for col in columns:
    df[col] = df[col].fillna(df[col].median())

print("\nPreprocessed Dataset:")
print(df.head())

# 4. Separate Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 5. Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Normalize Features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

# 8. Train Model
knn.fit(X_train, y_train)

# 9. Predict Results
y_pred = knn.predict(X_test)

# 10. Calculate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 11. Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

print("\n----- MODEL PERFORMANCE -----")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("Error Rate:", round(error_rate * 100, 2), "%")

print("Precision:", round(precision * 100, 2), "%")

print("Recall:", round(recall * 100, 2), "%")

# 12. Visualize Confusion Matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

plt.title("Diabetes Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 13. Display Final Result
print("\nDiabetes Prediction Completed Successfully!")


'''
3. Important Concepts
1. K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm used for classification. 
It classifies a new data point based on the majority class of its K nearest neighbors.

For example, when K = 5:
- If 3 neighbors have diabetes.
- If 2 neighbors do not have diabetes.
- The predicted result is Diabetes.

2. Data Normalization

StandardScaler converts features into a common scale so that features 
with larger numerical values do not dominate distance calculations.

Z={X-mu}/{sigma}

3. Confusion Matrix
	         Predicted No	Predicted Yes
Actual No	      TN	      FP
Actual Yes	      FN	      TP

4. Evaluation Metrics
1. Accuracy

Measures the percentage of correctly predicted outcomes.
Accuracy={TP+TN}/{TP+TN+FP+FN}

2. Error Rate

Measures the percentage of incorrect predictions.

Error
Rate=1-Accuracy

3. Precision

Measures how many patients predicted as diabetic actually have diabetes.

Precision={TP}/{TP+FP}

4. Recall

Measures how many actual diabetic patients are correctly identified.

Recall= {TP}/{TP+FN} '''