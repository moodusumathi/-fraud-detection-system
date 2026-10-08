import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

np.random.seed(42)

# ------------------------------------
# Generate Synthetic Data
# ------------------------------------
n = 10000

data = pd.DataFrame({
    "amount": np.random.exponential(scale=2000, size=n),
    "hour": np.random.randint(0, 24, n),
    "is_international": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    "is_online": np.random.choice([0, 1], n, p=[0.4, 0.6]),
    "device": np.random.choice(["Mobile", "Desktop"], n),
    "merchant_risk": np.random.choice(["Low", "Medium", "High"], n),
    "prev_failed_txn": np.random.poisson(0.5, n),
    "txn_frequency": np.random.randint(1, 10, n)
})

# Fraud logic (realistic)
data["fraud"] = (
    (data["amount"] > 5000).astype(int) |
    (data["is_international"] & data["is_online"]) |
    (data["merchant_risk"] == "High") |
    (data["prev_failed_txn"] > 2)
).astype(int)

# ------------------------------------
# Split
# ------------------------------------
X = data.drop("fraud", axis=1)
y = data["fraud"]

num_features = ["amount", "hour", "prev_failed_txn", "txn_frequency"]
cat_features = ["device", "merchant_risk", "is_international", "is_online"]

# ------------------------------------
# Pipeline
# ------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ------------------------------------
# Train
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------
# Save
# ------------------------------------
joblib.dump(model, "fraud_model.pkl")
print("✅ Model trained and saved as fraud_model.pkl")
