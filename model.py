import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("reservations_cleaned.csv")

# ----- 1. PREPROCESSING -----

# Target variable
target = 'is_fraud'

# Drop ID and time-based columns
df = df.drop(columns=['reservation_id', 'user_id', 'reservation_time', 'reservation_date'])

# Detect categorical columns
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

# Remove target from categorical columns (safely)
if target in categorical_cols:
    categorical_cols.remove(target)

# Encode categorical features using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop(columns=[target])
y = df[target]

# ----- 2. SPLIT DATA -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----- 3. MODEL -----
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----- 4. EVALUATION -----
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # For ROC AUC

# Classification report
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC AUC
roc_auc = roc_auc_score(y_test, y_prob)
print(f"📈 ROC AUC Score: {roc_auc:.4f}")
import joblib
import os

# Create a folder to save model artifacts
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/fraud_model.pkl")

# Save label encoders
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("✅ Model and encoders saved successfully!")
