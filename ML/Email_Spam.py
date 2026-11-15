'''
1. Algorithm
1. Import the required Python libraries.
2. Load the email spam dataset.
3. Preprocess the dataset:
   - Remove missing and duplicate records.
   - Convert email labels into numerical values.
   - Clean unnecessary text and special characters.
4. Convert email text into numerical features using TF-IDF Vectorization.
5. Split the dataset into training and testing sets.
6. Train the KNN classifier.
7. Train the SVM classifier.
8. Predict spam and non-spam emails using both classifiers.
9. Calculate Accuracy, Precision, Recall and F1-Score.
10. Compare the performance of both classifiers using a classification report and confusion matrix. '''



# Email Spam Classification using KNN and SVM

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 2. Load Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Select message and label columns
if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
else:
    df = df[["Category", "Message"]]
    df.columns = ["label", "message"]

print("Original Dataset:")
print(df.head())

# 3. Data Preprocessing

# Remove missing and duplicate values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Convert labels into numerical values
df["label"] = df["label"].str.lower().map({
    "ham": 0,
    "spam": 1
})

df.dropna(subset=["label"], inplace=True)

df["label"] = df["label"].astype(int)

# Remove empty messages
df = df[df["message"].str.strip() != ""]

print("\nDataset after preprocessing:")
print(df.head())

# 4. Separate Features and Labels
X = df["message"]
y = df["label"]

# 5. Convert Text into Numerical Features using TF-IDF
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X = tfidf.fit_transform(X)

# 6. Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Train KNN Classifier
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="cosine",
    algorithm="brute"
)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

# 8. Train SVM Classifier
svm = LinearSVC(random_state=42)

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

# 9. Evaluate KNN Classifier
print("\n----- KNN CLASSIFIER -----")

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)

print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)

print("\nClassification Report:")
print(classification_report(
    y_test, knn_pred,
    target_names=["Ham", "Spam"]
))

# 10. Evaluate SVM Classifier
print("\n----- SVM CLASSIFIER -----")

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)

print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)

print("\nClassification Report:")
print(classification_report(
    y_test, svm_pred,
    target_names=["Ham", "Spam"]
))

# 11. Compare Both Classifiers
results = pd.DataFrame({
    "Classifier": ["KNN", "SVM"],
    "Accuracy": [knn_accuracy, svm_accuracy],
    "Precision": [knn_precision, svm_precision],
    "Recall": [knn_recall, svm_recall],
    "F1 Score": [knn_f1, svm_f1]
})

print("\n----- PERFORMANCE COMPARISON -----")
print(results.to_string(index=False))

# 12. Visualize Performance Comparison
results.set_index("Classifier").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("KNN vs SVM Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 13. Confusion Matrix for KNN
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(
    confusion_matrix(y_test, knn_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
    ax=axes[0]
)

axes[0].set_title("KNN Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# 14. Confusion Matrix for SVM
sns.heatmap(
    confusion_matrix(y_test, svm_pred),
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
    ax=axes[1]
)

axes[1].set_title("SVM Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 15. Identify Best Classifier
if svm_f1 > knn_f1:
    print("\nSVM performs better based on F1 Score.")
elif knn_f1 > svm_f1:
    print("\nKNN performs better based on F1 Score.")
else:
    print("\nBoth classifiers have equal F1 Scores.")


'''
4. Evaluation Metrics
Metric	Meaning

Accuracy	Overall correct predictions
Precision	Correctly predicted spam out of all predicted spam
Recall	    Correctly identified spam out of all actual spam
F1-Score	Harmonic mean of precision and recall  

5. Performance Comparison
Parameter	              KNN           	SVM
Working	           Nearest neighbours	 Optimal hyperplane
Training	       Usually faster        Depends on dataset
Prediction	       Can be slower	     Usually faster
High-dimensional text	Less suitable	 Highly suitable
Memory usage	   Stores training data	 Stores learned model
Performance	       Dataset-dependent    Often better for text

6. Result
Email Spam Classification was successfully implemented using KNN and SVM classifiers. 
Text preprocessing and TF-IDF vectorization were performed. 
Both classifiers were evaluated using Accuracy, Precision, Recall and F1-Score. 
Their performance was compared using classification reports, confusion matrices and graphical visualization.

'''
