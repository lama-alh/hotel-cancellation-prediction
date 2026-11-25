# Run this in Google Colab after uploading hotel_clean.csv
# It trains a lighter model on user-friendly features and saves model.pkl for the Streamlit app.

import os
import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("hotel_clean.csv")

NUM = ["lead_time", "adr", "total_nights", "adults", "total_of_special_requests",
       "previous_cancellations", "booking_changes", "required_car_parking_spaces"]
CAT = ["hotel", "deposit_type", "market_segment", "customer_type"]

X = df[NUM + CAT]
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline([
    ("prep", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
        remainder="passthrough")),
    ("rf", RandomForestClassifier(
        n_estimators=150, max_depth=18, min_samples_leaf=5,
        random_state=42, n_jobs=-1)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
print(f"Accuracy = {accuracy_score(y_test, pred):.3f}")
print(f"F1       = {f1_score(y_test, pred):.3f}")
print(f"ROC-AUC  = {roc_auc_score(y_test, prob):.3f}")

joblib.dump(model, "model.pkl", compress=3)
print(f"model.pkl size: {os.path.getsize('model.pkl') / 1e6:.1f} MB")  # keep it under 25 MB
print("scikit-learn version:", sklearn.__version__)  # copy this into requirements.txt
