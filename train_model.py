import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -----------------------------------
# Load Dataset
# -----------------------------------
data = pd.read_excel(
    "dataset/karnataka_accident_dataset_4000.xlsx"
)

# -----------------------------------
# Select Features
# -----------------------------------
selected_features = [
    "Driver_Age",
    "Speed_Limit",
    "Weather",
    "Road_Type",
    "Traffic",
    "Alcohol_Involved"
]

target_column = "Accident_Severity"

# -----------------------------------
# Keep Required Columns
# -----------------------------------
data = data[
    selected_features + [target_column]
]

# -----------------------------------
# Handle Missing Values
# -----------------------------------

# Numerical Columns
num_cols = data.select_dtypes(
    include=['number']
).columns

num_imputer = SimpleImputer(
    strategy='mean'
)

data[num_cols] = num_imputer.fit_transform(
    data[num_cols]
)

# Categorical Columns
cat_cols = data.select_dtypes(
    include=['object', 'string']
).columns

for col in cat_cols:

    data[col] = data[col].fillna(
        data[col].mode()[0]
    )

# -----------------------------------
# Convert Target Variable
# -----------------------------------
data[target_column] = (
    data[target_column]
    .round()
    .astype(int)
)

# Convert:
# 1,2,3 --> 0,1,2
data[target_column] = (
    data[target_column] - 1
)

# -----------------------------------
# Encode Categorical Columns
# -----------------------------------
label_encoders = {}

for col in cat_cols:

    le = LabelEncoder()

    data[col] = le.fit_transform(
        data[col].astype(str)
    )

    label_encoders[col] = le

# -----------------------------------
# Features and Target
# -----------------------------------
X = data[selected_features]

y = data[target_column]

# -----------------------------------
# Train Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Models
# -----------------------------------

# Decision Tree
dt = DecisionTreeClassifier(
    random_state=42
)

# Logistic Regression
lr = LogisticRegression(
    max_iter=1000
)

# KNN
knn = KNeighborsClassifier(
    n_neighbors=5
)

# -----------------------------------
# Train Models
# -----------------------------------
dt.fit(X_train, y_train)

lr.fit(X_train, y_train)

knn.fit(X_train, y_train)

# -----------------------------------
# Predictions
# -----------------------------------
dt_pred = dt.predict(X_test)

lr_pred = lr.predict(X_test)

knn_pred = knn.predict(X_test)

# -----------------------------------
# Decision Tree Metrics
# -----------------------------------
dt_acc = accuracy_score(
    y_test,
    dt_pred
) * 100

dt_precision = precision_score(
    y_test,
    dt_pred,
    average='weighted'
)

dt_recall = recall_score(
    y_test,
    dt_pred,
    average='weighted'
)

dt_f1 = f1_score(
    y_test,
    dt_pred,
    average='weighted'
)

# -----------------------------------
# Logistic Regression Metrics
# -----------------------------------
lr_acc = accuracy_score(
    y_test,
    lr_pred
) * 100

lr_precision = precision_score(
    y_test,
    lr_pred,
    average='weighted'
)

lr_recall = recall_score(
    y_test,
    lr_pred,
    average='weighted'
)

lr_f1 = f1_score(
    y_test,
    lr_pred,
    average='weighted'
)

# -----------------------------------
# KNN Metrics
# -----------------------------------
knn_acc = accuracy_score(
    y_test,
    knn_pred
) * 100

knn_precision = precision_score(
    y_test,
    knn_pred,
    average='weighted'
)

knn_recall = recall_score(
    y_test,
    knn_pred,
    average='weighted'
)

knn_f1 = f1_score(
    y_test,
    knn_pred,
    average='weighted'
)

# -----------------------------------
# Print Results
# -----------------------------------
print("\nModel Comparison")
print("----------------------------------------------------------------")

print(
    f"{'Model':<25}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1 Score':<12}"
)

print("----------------------------------------------------------------")

print(
    f"{'Decision Tree':<25}"
    f"{dt_acc:.2f}%     "
    f"{dt_precision:.2f}        "
    f"{dt_recall:.2f}      "
    f"{dt_f1:.2f}"
)

print(
    f"{'Logistic Regression':<25}"
    f"{lr_acc:.2f}%     "
    f"{lr_precision:.2f}        "
    f"{lr_recall:.2f}      "
    f"{lr_f1:.2f}"
)

print(
    f"{'KNN':<25}"
    f"{knn_acc:.2f}%     "
    f"{knn_precision:.2f}        "
    f"{knn_recall:.2f}      "
    f"{knn_f1:.2f}"
)

# -----------------------------------
# Best Model
# -----------------------------------
scores = [
    dt_acc,
    lr_acc,
    knn_acc
]

models = [
    "Decision Tree",
    "Logistic Regression",
    "KNN"
]

best_model = models[np.argmax(scores)]

print("\n-----------------------------------")
print(f"Best Model: {best_model}")

# -----------------------------------
# Save Best Model
# -----------------------------------
if best_model == "Decision Tree":

    final_model = dt

elif best_model == "Logistic Regression":

    final_model = lr

else:

    final_model = knn

# Save Model
joblib.dump(
    final_model,
    "models/trained_model.pkl"
)

# Save Encoders
joblib.dump(
    label_encoders,
    "models/label_encoders.pkl"
)

# -----------------------------------
# Accuracy Comparison Graph
# -----------------------------------
plt.figure(figsize=(8, 5))

plt.bar(models, scores)

plt.xlabel("Models")

plt.ylabel("Accuracy (%)")

plt.title("Model Accuracy Comparison")

for i, v in enumerate(scores):

    plt.text(
        i,
        v + 0.5,
        f"{v:.2f}%",
        ha='center'
    )

plt.savefig(
    "graphs/model_comparison.png"
)

plt.show()

# -----------------------------------
# Precision Recall F1 Graph
# -----------------------------------

metrics = ['Precision', 'Recall', 'F1 Score']

decision_tree_scores = [
    dt_precision,
    dt_recall,
    dt_f1
]

logistic_scores = [
    lr_precision,
    lr_recall,
    lr_f1
]

knn_scores = [
    knn_precision,
    knn_recall,
    knn_f1
]

x = np.arange(len(metrics))

width = 0.25

plt.figure(figsize=(10, 5))

plt.bar(
    x - width,
    decision_tree_scores,
    width,
    label='Decision Tree'
)

plt.bar(
    x,
    logistic_scores,
    width,
    label='Logistic Regression'
)

plt.bar(
    x + width,
    knn_scores,
    width,
    label='KNN'
)

plt.xticks(x, metrics)

plt.ylabel("Score")

plt.title("Precision, Recall and F1 Score Comparison")

plt.legend()

plt.savefig(
    "graphs/metrics_comparison.png"
)

plt.show()

# -----------------------------------
# Confusion Matrix
# -----------------------------------

cm = confusion_matrix(
    y_test,
    lr_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Low', 'Medium', 'High'],
    yticklabels=['Low', 'Medium', 'High']
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix - Logistic Regression")

plt.savefig(
    "graphs/confusion_matrix.png"
)

plt.show()

print("\nModel and graphs saved successfully.")
