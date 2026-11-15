'''
1. Algorithm
1. Import the required Python libraries.
2. Load the bank customer churn dataset.
3. Perform data preprocessing:
   - Remove missing and duplicate values.
   - Drop unnecessary columns such as Customer ID and Surname.
   - Convert categorical variables into numerical values using One-Hot Encoding.
4. Separate input features (X) and target variable (y).
5. Split the dataset into training and testing sets (80:20).
6. Normalize the input features using StandardScaler.
7. Build a Neural Network using Keras Sequential API.
8. Compile the model using:
   - Optimizer: Adam
   - Loss Function: Binary Crossentropy
   - Metric: Accuracy
9. Train the neural network using training data.
10. Predict customer churn on testing data.
11. Optimize the model using Early Stopping.
12. Evaluate performance using Accuracy and Confusion Matrix. '''


# Bank Customer Churn Prediction using Neural Network

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 2. Load Dataset
df = pd.read_csv("Churn_Modelling.csv")

print("Original Dataset:")
print(df.head())

# 3. Data Preprocessing

# Remove duplicate and missing records
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Drop unnecessary columns
df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1,
    inplace=True,
    errors="ignore"
)

# Convert categorical columns into numerical values
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True,
    dtype=int
)

print("\nPreprocessed Dataset:")
print(df.head())

# 4. Separate Features and Target
X = df.drop("Exited", axis=1)
y = df["Exited"]

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

# 7. Build Neural Network Model
model = Sequential()

# Input Layer and First Hidden Layer
model.add(Dense(
    64,
    activation="relu",
    input_shape=(X_train.shape[1],)
))

# Dropout Layer
model.add(Dropout(0.3))

# Second Hidden Layer
model.add(Dense(32, activation="relu"))

# Output Layer
model.add(Dense(1, activation="sigmoid"))

# 8. Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 9. Optimize Model using Early Stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 10. Train Neural Network
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 11. Predict Customer Churn
y_prob = model.predict(X_test)

y_pred = (y_prob > 0.5).astype(int).ravel()

# 12. Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# 13. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 14. Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Churned", "Churned"]
))

# 15. Visualize Confusion Matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Churned", "Churned"],
    yticklabels=["Not Churned", "Churned"]
)

plt.title("Bank Customer Churn Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 16. Plot Training and Validation Accuracy
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# 17. Final Result
print("\nFinal Test Accuracy:", round(accuracy * 100, 2), "%")
print("Neural Network Churn Prediction Completed Successfully!")



'''
2. Data Normalization

StandardScaler transforms features to a common scale.

Z = X-mu /sigma

Where:
- (X) = Original value
- (mu) = Mean
- (sigma) = Standard deviation
3. Activation Functions
- ReLU: Used in hidden layers to introduce non-linearity.
- Sigmoid: Produces an output between 0 and 1 for binary classification.
4. Model Optimization

Adam optimizer updates model weights to minimize prediction error. 
Early Stopping prevents unnecessary training and helps reduce overfitting. 

4. Evaluation Metrics

Confusion Matrix
                     Predicted: No Churn	Predicted:Churn
Actual: No Churn	 True Negative (TN)     False Positive (FP)
Actual: Churn	     False Negative (FN)	True Positive (TP)
 
Accuracy formula:

Accuracy= TP+TN / TP+TN+FP+FN

'''