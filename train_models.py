"""
train_models.py
---------------
Downloads the loan dataset, preprocesses it, trains Decision Tree,
Naive Bayes, and ANN models, then serializes all artefacts to disk
so the Streamlit dashboard can load them instantly.

Run once:  python train_models.py
"""

import os
import sys
import json
import warnings
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# -- paths --
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "loan_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -- 1. Download dataset --
DATA_URL = (
    "https://raw.githubusercontent.com/dphi-official/Datasets/"
    "master/Loan_Data/loan_train.csv"
)

if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"  Saved -> {DATA_PATH}")
else:
    print(f"Dataset already exists at {DATA_PATH}")

# -- 2. Load & basic info --
df = pd.read_csv(DATA_PATH)
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)
print(f"\nDataset shape: {df.shape}")
print("Target distribution:\n", df["Loan_Status"].value_counts())

# -- 3. Preprocessing --
df_proc = df.copy()

if "Loan_ID" in df_proc.columns:
    df_proc = df_proc.drop("Loan_ID", axis=1)

X = df_proc.drop("Loan_Status", axis=1)
y = df_proc["Loan_Status"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols   = X.select_dtypes(include=[np.number]).columns.tolist()

# Impute missing values
cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="median")
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
X[numerical_cols]   = num_imputer.fit_transform(X[numerical_cols])

# Label-encode categoricals
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale
X_numeric = X.apply(pd.to_numeric, errors="coerce")
for col in X_numeric.columns:
    if X_numeric[col].isnull().any():
        X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())

scaler  = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X.columns)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape}  |  Test size: {X_test.shape}")

# -- 4. Decision Tree --
print("\n[1/3] Training Decision Tree...")
dt_model = DecisionTreeClassifier(
    max_depth=5, min_samples_split=10,
    min_samples_leaf=5, random_state=42, criterion="gini"
)
dt_model.fit(X_train, y_train)
dt_pred  = dt_model.predict(X_test)
dt_prob  = dt_model.predict_proba(X_test)[:, 1]
dt_train_pred = dt_model.predict(X_train)

dt_metrics = {
    "Accuracy":  accuracy_score(y_test, dt_pred),
    "Precision": precision_score(y_test, dt_pred),
    "Recall":    recall_score(y_test, dt_pred),
    "F1-Score":  f1_score(y_test, dt_pred),
    "AUC":       roc_auc_score(y_test, dt_prob),
    "Train_Accuracy": accuracy_score(y_train, dt_train_pred),
}
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
cm_dt = confusion_matrix(y_test, dt_pred).tolist()

# Feature importance
feature_importance = pd.DataFrame({
    "Feature":    X.columns.tolist(),
    "Importance": dt_model.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print(f"  DT Accuracy: {dt_metrics['Accuracy']:.4f}")

# -- 5. Naive Bayes --
print("[2/3] Training Naive Bayes...")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred  = nb_model.predict(X_test)
nb_prob  = nb_model.predict_proba(X_test)[:, 1]
nb_train_pred = nb_model.predict(X_train)

nb_metrics = {
    "Accuracy":  accuracy_score(y_test, nb_pred),
    "Precision": precision_score(y_test, nb_pred),
    "Recall":    recall_score(y_test, nb_pred),
    "F1-Score":  f1_score(y_test, nb_pred),
    "AUC":       roc_auc_score(y_test, nb_prob),
    "Train_Accuracy": accuracy_score(y_train, nb_train_pred),
}
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_prob)
cm_nb = confusion_matrix(y_test, nb_pred).tolist()
print(f"  NB Accuracy: {nb_metrics['Accuracy']:.4f}")

# -- 6. ANN --
print("[3/3] Training ANN...")
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    ann_model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    ann_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = ann_model.fit(
        X_train, y_train,
        epochs=100, batch_size=16,
        validation_split=0.2, verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )
        ],
    )

    ann_prob_test  = ann_model.predict(X_test,  verbose=0).flatten()
    ann_pred_test  = (ann_prob_test >= 0.5).astype(int)
    ann_prob_train = ann_model.predict(X_train, verbose=0).flatten()
    ann_pred_train = (ann_prob_train >= 0.5).astype(int)

    ann_metrics = {
        "Accuracy":  accuracy_score(y_test,  ann_pred_test),
        "Precision": precision_score(y_test, ann_pred_test),
        "Recall":    recall_score(y_test,    ann_pred_test),
        "F1-Score":  f1_score(y_test,        ann_pred_test),
        "AUC":       roc_auc_score(y_test,   ann_prob_test),
        "Train_Accuracy": accuracy_score(y_train, ann_pred_train),
    }
    fpr_ann, tpr_ann, _ = roc_curve(y_test, ann_prob_test)
    cm_ann = confusion_matrix(y_test, ann_pred_test).tolist()

    ann_history = {
        "loss":     history.history["loss"],
        "val_loss": history.history["val_loss"],
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
    }
    ann_available = True
    print(f"  ANN Accuracy: {ann_metrics['Accuracy']:.4f}")

    ann_model.save(os.path.join(MODEL_DIR, "ann_model.keras"))
    print("  ANN model saved.")

except Exception as e:
    print(f"  ANN training failed ({e}). Skipping.")
    ann_available   = False
    ann_metrics     = {}
    fpr_ann = tpr_ann = []
    cm_ann          = []
    ann_history     = {}

# -- 7. Serialize everything --
print("\nSaving artefacts...")

joblib.dump(dt_model,       os.path.join(MODEL_DIR, "dt_model.pkl"))
joblib.dump(nb_model,       os.path.join(MODEL_DIR, "nb_model.pkl"))
joblib.dump(scaler,         os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
joblib.dump(cat_imputer,    os.path.join(MODEL_DIR, "cat_imputer.pkl"))
joblib.dump(num_imputer,    os.path.join(MODEL_DIR, "num_imputer.pkl"))

feature_importance.to_csv(
    os.path.join(MODEL_DIR, "feature_importance.csv"), index=False
)

# Save metrics and curve data as JSON for the dashboard
results = {
    "feature_cols": X.columns.tolist(),
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "ann_available": ann_available,
    "dataset_shape": list(df.shape),
    "approval_rate": float(y.value_counts(normalize=True).get(1, 0)),
    "models": {
        "Decision Tree": {
            "metrics": dt_metrics,
            "cm": cm_dt,
            "fpr": fpr_dt.tolist(),
            "tpr": tpr_dt.tolist(),
        },
        "Naive Bayes": {
            "metrics": nb_metrics,
            "cm": cm_nb,
            "fpr": fpr_nb.tolist(),
            "tpr": tpr_nb.tolist(),
        },
    },
    "ann_history": ann_history,
}

if ann_available:
    results["models"]["ANN"] = {
        "metrics": ann_metrics,
        "cm": cm_ann,
        "fpr": fpr_ann.tolist(),
        "tpr": tpr_ann.tolist(),
    }

with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Save raw dataset (needed for EDA page)
df.to_csv(os.path.join(MODEL_DIR, "dataset.csv"), index=False)

print("\n[OK] All artefacts saved to:", MODEL_DIR)
print("   Run the dashboard with:  streamlit run dashboard.py")
