# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load Dataset
df = pd.read_csv("heart_disease system_uci.csv")  # Change to your correct filename

# Basic info
print(df.head())
print(df.info())

# Drop unnecessary columns if any (like 'id')
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Handle missing values
df = df.dropna()  # or df.fillna(method='ffill') if you prefer filling

# Convert categorical columns to numeric
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Print column names to identify target
print(df.columns)

# Update this to the correct target column name from your dataset
target_col = 'num'  # OR 'target', whichever is in your dataset

# Convert the target variable to binary
# Assume 0 is no disease, and > 0 is disease
df[target_col] = df[target_col].apply(lambda x: 1 if x > 0 else 0)

X = df.drop(target_col, axis=1)
y = df[target_col]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y) # Added stratify

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

# Training and evaluating models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.2f}")
    # Print classification report and confusion matrix for binary classification
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Visualizing Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# ROC Curve for Random Forest
# Re-train or use the trained Random Forest model from the loop
# rf_model = RandomForestClassifier() # No need to re-instantiate if you want the same model from the loop
# rf_model.fit(X_train, y_train) # No need to re-fit if already done in the loop

# Assuming the Random Forest model trained in the loop is stored in 'model' when name is "Random Forest"
# Or just use the rf_model defined specifically for ROC if you prefer:
rf_model = RandomForestClassifier() # Instantiate again if you want a fresh one for ROC
rf_model.fit(X_train, y_train)      # Fit again

# Get probabilities for the positive class (which is now 1)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC for the binary problem
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve - Random Forest (Binary Classification)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
