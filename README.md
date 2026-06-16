# 💳 Loan Approval Prediction — ML Dashboard

A professional, interactive dashboard that predicts whether a loan application will be **Approved** or **Rejected** using three Machine Learning models: **Decision Tree**, **Naive Bayes**, and an **Artificial Neural Network (ANN)**.

Built with Python, Scikit-Learn, TensorFlow/Keras, and Streamlit.

---

## 📁 Project Structure

```
Loan Approval Prediction/
│
├── README.md                        ← This file (complete project documentation)
├── Loan_Approval_Prediction.ipynb   ← Original Jupyter Notebook (research & analysis)
├── train_models.py                  ← Trains all 3 models and saves them to disk
├── dashboard.py                     ← The Streamlit web dashboard (main app)
├── notebook_code.py                 ← All notebook code in a standalone .py file
├── loan_data.csv                    ← The dataset (auto-downloaded if missing)
│
├── models/                          ← Created after running train_models.py
│   ├── dt_model.pkl                 ← Saved Decision Tree model
│   ├── nb_model.pkl                 ← Saved Naive Bayes model
│   ├── ann_model.keras              ← Saved ANN model (TensorFlow/Keras)
│   ├── ann_model.h5                 ← Saved ANN model (legacy HDF5 format)
│   ├── scaler.pkl                   ← Feature scaler (StandardScaler)
│   ├── label_encoders.pkl           ← Encoders for categorical features
│   ├── cat_imputer.pkl              ← Imputer for missing categorical values
│   ├── num_imputer.pkl              ← Imputer for missing numerical values
│   ├── feature_importance.csv       ← Feature importance scores from Decision Tree
│   ├── results.json                 ← All metrics, ROC data, confusion matrices
│   └── dataset.csv                  ← Copy of dataset used by the dashboard
│
└── .streamlit/
    └── config.toml                  ← Dashboard theme configuration (dark mode)
```

---

## 🖥️ System Requirements

| Requirement       | Minimum                              |
| ----------------- | ------------------------------------ |
| Operating System  | Windows 10 / 11                      |
| Python Version    | **Python 3.10+** (3.13 recommended)  |
| RAM               | 4 GB or more                         |
| Disk Space        | ~1 GB (including TensorFlow)         |
| Internet          | Required only for first-time setup   |

---

## 📦 Step 1 — Install Required Libraries

Open **Command Prompt** or **PowerShell**, navigate to the project folder, and run:

```bash
cd "D:\ML LAB FAHAD\Loan Approval Prediction"
```

Then install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib tensorflow
```

### Individual Package Details

| #  | Package        | What It Does                              |
| -- | -------------- | ----------------------------------------- |
| 1  | **pandas**     | Data loading, tables, manipulation        |
| 2  | **numpy**      | Math operations, arrays                   |
| 3  | **scikit-learn**| ML models (Decision Tree, Naive Bayes)   |
| 4  | **matplotlib** | Charts, graphs, visualizations            |
| 5  | **seaborn**    | Statistical data visualization            |
| 6  | **streamlit**  | The web dashboard framework               |
| 7  | **joblib**     | Saving/loading trained models             |
| 8  | **tensorflow** | Deep learning ANN model                   |

### Verify All Packages Are Installed

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, streamlit, joblib, tensorflow; print('All packages installed successfully!')"
```

---

## 🏋️ Step 2 — Train the Models

Before launching the dashboard, train the ML models once:

```bash
python train_models.py
```

**Expected output:**

```
Dataset already exists at ...\loan_data.csv

Dataset shape: (491, 13)
Target distribution:
 1    343
 0    148

Train size: (392, 11)  |  Test size: (99, 11)

[1/3] Training Decision Tree...
  DT Accuracy: 0.7879
[2/3] Training Naive Bayes...
  NB Accuracy: 0.8586
[3/3] Training ANN...
  ANN Accuracy: 0.8384
  ANN model saved.

Saving artefacts...

[OK] All artefacts saved to: ...\models
   Run the dashboard with:  streamlit run dashboard.py
```

> **You only need to run this once.** The trained models are saved to disk. Re-run anytime to retrain.

---

## 🚀 Step 3 — Launch the Dashboard

Start the interactive web dashboard:

```bash
streamlit run dashboard.py
```

Your browser will automatically open at:

```
http://localhost:8501
```

---

## 🧭 Step 4 — Using the Dashboard

The dashboard has **4 pages**, accessible from the sidebar:

### Page 1: 📊 Data Explorer
- View dataset statistics (total applications, approval rate, missing values)
- See 6 distribution charts (loan status, gender, education, income, credit history, property area)
- Explore the correlation heatmap
- Preview raw data

### Page 2: 🤖 Model Insights
- Compare all 3 models across 5 metrics (Accuracy, Precision, Recall, F1-Score, AUC)
- View visual metric breakdowns with bar charts
- Compare ROC curves for all models
- See confusion matrices side-by-side
- View ANN training history (loss & accuracy over epochs)
- Analyse feature importance from Decision Tree

### Page 3: 🔮 Predict Loan
- Fill in applicant details (income, education, credit history, etc.)
- Select which model to use (Decision Tree, Naive Bayes, or ANN)
- Get an instant **Approved / Rejected** prediction with confidence score
- View risk assessment and key influencing factors

### Page 4: 📝 Source Code
- View the complete source code of every project file directly in the dashboard
- Review `train_models.py`, `dashboard.py`, and `notebook_code.py` with syntax highlighting

---

## 🧠 Models Used

### 1. Decision Tree Classifier
- **Type:** Supervised classification using Gini impurity
- **Configuration:** `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5`
- **How it works:** Builds a tree-like flowchart by finding the best yes/no question at each step
- **Strengths:** Fast, interpretable, provides feature importance

### 2. Gaussian Naive Bayes
- **Type:** Probabilistic classifier using Bayes' theorem
- **Configuration:** Default (no hyperparameters)
- **How it works:** Calculates probability of approval given features using Bayes' formula, assuming feature independence
- **Strengths:** Simple, fast, handles small datasets well

### 3. Artificial Neural Network (ANN)
- **Type:** Deep learning model with Dense layers
- **Architecture:**
  ```
  Input (11 features)
    → Dense(64, ReLU) → Dropout(0.3)
    → Dense(32, ReLU) → Dropout(0.2)
    → Dense(16, ReLU)
    → Dense(1, Sigmoid) → Output probability
  ```
- **Optimizer:** Adam (learning_rate=0.001)
- **Loss:** Binary Cross-Entropy
- **Training:** Up to 100 epochs, batch size 16, 20% validation split
- **Early Stopping:** patience=15, restore_best_weights=True
- **Dropout:** 30% after first layer, 20% after second — prevents overfitting
- **Strengths:** Can learn complex non-linear patterns in data

#### ANN Key Concepts

| Concept                | Description                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| **ReLU activation**    | If input > 0, pass it through; if ≤ 0, output 0. Adds non-linearity for complex pattern learning.  |
| **Sigmoid activation** | Squishes output to 0–1 range. Perfect for probability output in binary classification.              |
| **Dropout**            | Randomly "turns off" neurons during training, forcing the network to not rely on any single neuron. |
| **Adam optimizer**     | Adaptive learning rate optimizer. Adjusts step size automatically — better than basic SGD.          |
| **Binary Crossentropy**| Measures how far predicted probability is from actual label (0 or 1). The loss function to minimize.|
| **Early Stopping**     | Monitors validation loss; stops training if no improvement for 15 epochs. Prevents overfitting.     |

#### ANN Training Process
1. Data flows forward through layers (forward propagation)
2. Loss is computed comparing prediction to actual label
3. Error flows backward adjusting weights (backpropagation)
4. Adam optimizer determines how much to adjust each weight
5. Process repeats for up to 100 epochs (or until early stopping triggers)
6. Best model weights (lowest validation loss) are restored

---

## 📊 Model Performance Summary

| Model           | Accuracy | Precision | Recall  | F1-Score | AUC    |
| --------------- | -------- | --------- | ------- | -------- | ------ |
| Decision Tree   | 78.79%   | 86.36%    | 82.61%  | 84.44%  | 87.58% |
| Naive Bayes     | 85.86%   | 85.71%    | 95.65%  | 90.41%  | 81.64% |
| ANN             | ~83.84%  | ~85%      | ~93%    | ~89%    | ~84%   |

> **Best Model by Accuracy:** Naive Bayes (85.86%)

### Metrics Explained

| Metric        | What It Measures                                                                |
| ------------- | ------------------------------------------------------------------------------- |
| **Accuracy**  | Out of all predictions, how many were correct?                                  |
| **Precision** | When the model says "Approved", how often is it actually correct?               |
| **Recall**    | Out of all actually approved loans, how many did the model catch?               |
| **F1-Score**  | Harmonic mean of Precision and Recall — balanced measure                        |
| **AUC**       | Area under ROC curve — how well the model distinguishes classes (1.0 = perfect) |

---

## 📐 Dataset Information

| Column              | Description                                        | Type        |
| ------------------- | -------------------------------------------------- | ----------- |
| `Loan_ID`           | Unique identifier (dropped before training)        | Text        |
| `Gender`            | Male / Female                                      | Categorical |
| `Married`           | Yes / No                                           | Categorical |
| `Dependents`        | Number of dependents (0, 1, 2, 3+)                 | Categorical |
| `Education`         | Graduate / Not Graduate                            | Categorical |
| `Self_Employed`     | Yes / No                                           | Categorical |
| `ApplicantIncome`   | Monthly income of the applicant                    | Numerical   |
| `CoapplicantIncome` | Monthly income of the co-applicant                 | Numerical   |
| `LoanAmount`        | Loan amount (in thousands)                         | Numerical   |
| `Loan_Amount_Term`  | Repayment period (in months)                       | Numerical   |
| `Credit_History`    | 1 = good track record, 0 = bad                     | Numerical   |
| `Property_Area`     | Urban / Semiurban / Rural                          | Categorical |
| `Loan_Status`       | **Target**: 1 = Approved, 0 = Rejected             | Binary      |

- **Total Samples:** 491
- **Features:** 11 (after dropping Loan_ID and index columns)
- **Approval Rate:** ~69.9%
- **Train/Test Split:** 80/20 with stratification

---

## 🔧 Preprocessing Pipeline

1. **Drop Index Column** — Remove `Unnamed: 0` (CSV artifact) and `Loan_ID`
2. **Handle Missing Values** — Categorical → mode (most frequent), Numerical → median
3. **Label Encoding** — Convert text categories to numbers (e.g., Male → 1, Female → 0)
4. **Feature Scaling** — StandardScaler (mean=0, std=1) to normalize all features
5. **Train/Test Split** — 80% training, 20% testing (stratified to maintain class balance)

---

## 📝 Code Files Overview

| File                     | Lines | Purpose                                                         |
| ------------------------ | ----- | --------------------------------------------------------------- |
| `train_models.py`        | ~271  | Complete ML pipeline — download, preprocess, train, save        |
| `dashboard.py`           | ~927  | Streamlit dashboard — EDA, insights, prediction, source code    |
| `notebook_code.py`       | ~540  | All notebook code as standalone Python script (all errors fixed)|
| `Loan_Approval_Prediction.ipynb` | —  | Original Jupyter Notebook with outputs                    |

---

## 📋 Complete Code Section

Below is a summary of every major code component. Full source code is available in the dashboard's **📝 Source Code** page or in the individual `.py` files.

### Data Loading & Preprocessing

```python
# Load dataset
df = pd.read_csv('loan_data.csv')

# Handle missing values
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='median')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
X[numerical_cols]   = num_imputer.fit_transform(X[numerical_cols])

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Feature scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

### Decision Tree Model

```python
dt_model = DecisionTreeClassifier(
    max_depth=5, min_samples_split=10,
    min_samples_leaf=5, random_state=42, criterion='gini'
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]
```

### Naive Bayes Model

```python
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_prob = nb_model.predict_proba(X_test)[:, 1]
```

### ANN Model (TensorFlow/Keras)

```python
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = ann_model.fit(
    X_train, y_train,
    epochs=100, batch_size=16,
    validation_split=0.2, verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True
        )
    ]
)
```

### Evaluation Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

metrics = {
    'Accuracy':  accuracy_score(y_test, predictions),
    'Precision': precision_score(y_test, predictions),
    'Recall':    recall_score(y_test, predictions),
    'F1-Score':  f1_score(y_test, predictions),
    'AUC':       roc_auc_score(y_test, probabilities),
}
```

### Prediction Function

```python
def predict_loan_approval(model, input_data, model_name='Model'):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[X.columns]
    input_scaled = scaler.transform(input_df)

    if model_name == 'ANN':
        prob = model.predict(input_scaled, verbose=0).flatten()[0]
        pred = 1 if prob >= 0.5 else 0
    else:
        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

    result = 'Approved' if pred == 1 else 'Not Approved'
    return result, prob
```

---

## ❓ Troubleshooting

| Problem                          | Solution                                                        |
| -------------------------------- | --------------------------------------------------------------- |
| `ModuleNotFoundError`            | Run: `pip install <package-name>`                               |
| `Model artefacts not found`      | Run: `python train_models.py` first                             |
| ANN shows as "Unavailable"       | Run: `pip install tensorflow` then retrain                      |
| Dashboard not opening            | Open `http://localhost:8501` manually in browser                |
| Port 8501 already in use         | Run: `streamlit run dashboard.py --server.port 8502`            |
| `!pip install` error in .py file | This is Jupyter magic syntax — use `pip install` in terminal    |
| `display()` not defined          | This is Jupyter-only — use `print()` in .py scripts             |

---

## Live Demo
https://loan-approval-prediction-s.streamlit.app/

## 👥 Credits

- **Dataset:** [DPhi Loan Data](https://github.com/dphi-official/Datasets/blob/master/Loan_Data/loan_train.csv)
- **Frameworks:** Scikit-Learn, TensorFlow/Keras, Streamlit, Matplotlib, Seaborn
- **Built as part of an ML Lab course project**

---

*This project demonstrates the complete ML pipeline — from data analysis and model training to interactive deployment via a web dashboard.*
