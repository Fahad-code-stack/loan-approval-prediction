"""
dashboard.py  --  Loan Approval Professional Dashboard
======================================================
Run:  streamlit run dashboard.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

warnings.filterwarnings("ignore")

def preprocess_and_split(df):
    df_proc = df.copy()
    if "Unnamed: 0" in df_proc.columns:
        df_proc = df_proc.drop("Unnamed: 0", axis=1)
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
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders


# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── colour palette ─────────────────────────────────────────────────────────────
C = {
    "bg":         "#0B0F14",
    "sidebar":    "#111827",
    "card":       "#1F2937",
    "teal":       "#14B8A6",
    "teal_light": "#2DD4BF",
    "green":      "#22C55E",
    "red":        "#EF4444",
    "amber":      "#F59E0B",
    "text":       "#F8FAFC",
    "muted":      "#94A3B8",
}

CHART_BG   = C["card"]
CHART_TEXT = C["muted"]

MODEL_COLORS = {
    "Decision Tree": "#14B8A6",
    "Naive Bayes":   "#F59E0B",
    "ANN":           "#818CF8",
}

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSight - ML Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ─────────────────────────────────────────────────────────────────
bg         = C["bg"]
sidebar    = C["sidebar"]
card       = C["card"]
teal       = C["teal"]
teal_light = C["teal_light"]
green      = C["green"]
red        = C["red"]
amber      = C["amber"]
text       = C["text"]
muted      = C["muted"]

st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {bg} !important;
    color: {text};
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}
[data-testid="stSidebar"] {{
    background-color: {sidebar} !important;
}}
[data-testid="stSidebar"] * {{ color: {text} !important; }}

[data-testid="metric-container"] {{
    background: {card};
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}}
[data-testid="metric-container"] label {{
    color: {muted} !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: .05em;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {teal} !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}}

h1 {{ color: {text} !important; font-weight: 800; letter-spacing:-0.03em; }}
h2 {{ color: {text} !important; font-weight: 700; }}
h3 {{ color: {teal} !important; font-weight: 600; }}
label {{ color: {muted} !important; font-size:0.82rem; }}
hr {{ border-color: #2D3748; margin: 1.5rem 0; }}

.card {{
    background: {card};
    border: 1px solid #2D3748;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}}

.badge {{
    display:inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
}}
.badge-green  {{ background:#052e16; color:{green}; border:1px solid {green}; }}
.badge-red    {{ background:#450a0a; color:{red};   border:1px solid {red};   }}
.badge-teal   {{ background:#042f2e; color:{teal};  border:1px solid {teal};  }}
.badge-amber  {{ background:#422006; color:{amber}; border:1px solid {amber}; }}

.result-approved {{
    background: linear-gradient(135deg,#052e16,#064e3b);
    border: 2px solid {green};
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}}
.result-rejected {{
    background: linear-gradient(135deg,#450a0a,#7f1d1d);
    border: 2px solid {red};
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}}
.result-title {{ font-size: 2rem; font-weight: 800; margin: 0; }}

thead tr th {{
    background: {card} !important;
    color: {teal} !important;
}}
tbody tr td {{ color: {text} !important; }}
.stProgress > div > div > div > div {{ background-color: {teal} !important; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA / MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models...")
def load_artefacts():

    results_path = os.path.join(MODEL_DIR, "results.json")

    if not os.path.exists(results_path):
        return None

    with open(results_path) as f:
        results = json.load(f)

    artefacts = {
        "results": results,
        "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        "label_encoders": joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl")),
        "cat_imputer": joblib.load(os.path.join(MODEL_DIR, "cat_imputer.pkl")),
        "num_imputer": joblib.load(os.path.join(MODEL_DIR, "num_imputer.pkl")),
        "dt_model": joblib.load(os.path.join(MODEL_DIR, "dt_model.pkl")),
        "nb_model": joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl")),
        "feature_importance": pd.read_csv(
            os.path.join(MODEL_DIR, "feature_importance.csv")
        ),
        "df": pd.read_csv(os.path.join(MODEL_DIR, "dataset.csv")),
    }

    # Load ANN model if available
    try:
        from tensorflow.keras.models import load_model

        ann_keras = os.path.join(MODEL_DIR, "ann_model.keras")
        ann_h5 = os.path.join(MODEL_DIR, "ann_model.h5")

        if os.path.exists(ann_keras):
            artefacts["ann_model"] = load_model(ann_keras)

        elif os.path.exists(ann_h5):
            artefacts["ann_model"] = load_model(ann_h5)

        else:
            artefacts["ann_model"] = None

    except Exception as e:
        print("ANN loading error:", e)
        artefacts["ann_model"] = None

    return artefacts

  


# ════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB HELPERS
# ════════════════════════════════════════════════════════════════════════════

def styled_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2D3748")
    ax.tick_params(colors=CHART_TEXT, labelsize=9)
    ax.xaxis.label.set_color(CHART_TEXT)
    ax.yaxis.label.set_color(CHART_TEXT)
    ax.title.set_color(C["text"])
    return fig, ax


def styled_multi(rows, cols, w=14, h=8):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(CHART_BG)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(CHART_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2D3748")
        ax.tick_params(colors=CHART_TEXT, labelsize=8)
        ax.xaxis.label.set_color(CHART_TEXT)
        ax.yaxis.label.set_color(CHART_TEXT)
        ax.title.set_color(C["text"])
    return fig, axes


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
            <div style="font-size:2.4rem;">💳</div>
            <div style="font-size:1.25rem;font-weight:800;
                        background:linear-gradient(90deg,#14B8A6,#2DD4BF);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                LoanSight
            </div>
            <div style="font-size:0.72rem;color:#94A3B8;margin-top:2px;">
                ML Approval Dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        pages = {
            "📊  Data Explorer":  "explorer",
            "🤖  Model Insights": "insights",
            "⚙️  Model Tuning":    "tuning",
            "🔮  Predict Loan":   "predict",
            "📝  Source Code":    "code",
        }

        if "page" not in st.session_state:
            st.session_state.page = "explorer"

        for label, key in pages.items():
            if st.button(label, key="nav_" + key, use_container_width=True):
                st.session_state.page = key

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.7rem;color:#94A3B8;text-align:center;
                    line-height:1.8;padding-top:0.5rem;">
            Decision Tree &nbsp;·&nbsp; Naive Bayes<br>
            Artificial Neural Network<br><br>
            <span style="color:#14B8A6">Loan Approval Prediction</span>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 -- DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════

def page_explorer(art):
    df = art["df"]

    st.markdown("## 📊 Data Explorer")
    st.markdown(
        "<span style='color:#94A3B8;font-size:0.9rem;'>"
        "Exploratory analysis of the loan applicant dataset</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # KPI row
    total    = len(df)
    import pandas.api.types as ptypes
    if ptypes.is_numeric_dtype(df["Loan_Status"]):
        approved = int(df["Loan_Status"].sum())
    else:
        approved = int((df["Loan_Status"].astype(str).str.upper().isin(["Y", "YES", "1", "TRUE"])).sum())
    rejected = total - approved
    app_rate = approved / total * 100
    missing  = int(df.isnull().sum().sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Applications",  "{:,}".format(total))
    k2.metric("Approved",            "{:,}".format(approved),
              delta="{:.1f}% rate".format(app_rate), delta_color="normal")
    k3.metric("Rejected",            "{:,}".format(rejected))
    k4.metric("Missing Values (raw)", "{:,}".format(missing))

    st.markdown("---")

    # Distribution charts
    st.markdown("### Distribution Analysis")
    fig, axes = styled_multi(2, 3, w=16, h=10)

    status_counts = df["Loan_Status"].value_counts()
    xlabels = ["Approved", "Rejected"] if status_counts.index[0] in [1, "Y"] \
              else list(status_counts.index)
    bars = axes[0, 0].bar(
        xlabels, status_counts.values,
        color=[C["green"], C["red"]], edgecolor="#0B0F14", linewidth=1.5, width=0.5,
    )
    for b in bars:
        axes[0, 0].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 2,
            str(int(b.get_height())), ha="center",
            color=C["text"], fontsize=10, fontweight="bold",
        )
    axes[0, 0].set_title("Loan Status Distribution", fontweight="bold")
    axes[0, 0].set_ylabel("Count")

    if "Gender" in df.columns:
        sns.countplot(data=df, x="Gender", hue="Loan_Status", ax=axes[0, 1],
                      palette=[C["teal"], C["red"]])
        axes[0, 1].set_title("Gender vs Loan Status", fontweight="bold")
        axes[0, 1].legend(title="Status", labels=["Approved", "Rejected"],
                          facecolor=CHART_BG, labelcolor=C["text"])

    if "Education" in df.columns:
        sns.countplot(data=df, x="Education", hue="Loan_Status", ax=axes[0, 2],
                      palette=[C["teal"], C["red"]])
        axes[0, 2].set_title("Education vs Loan Status", fontweight="bold")
        axes[0, 2].legend(title="Status", labels=["Approved", "Rejected"],
                          facecolor=CHART_BG, labelcolor=C["text"])
        axes[0, 2].tick_params(axis="x", rotation=15)

    if "ApplicantIncome" in df.columns:
        mask_app = df["Loan_Status"] == 1 if 1 in df["Loan_Status"].values \
                   else df["Loan_Status"] == "Y"
        axes[1, 0].hist(df[mask_app]["ApplicantIncome"],   bins=40, color=C["green"], alpha=0.7, label="Approved")
        axes[1, 0].hist(df[~mask_app]["ApplicantIncome"],  bins=40, color=C["red"],   alpha=0.7, label="Rejected")
        axes[1, 0].set_xlim(0, 20000)
        axes[1, 0].set_title("Applicant Income Distribution", fontweight="bold")
        axes[1, 0].legend(facecolor=CHART_BG, labelcolor=C["text"])

    if "Credit_History" in df.columns:
        sns.countplot(data=df, x="Credit_History", hue="Loan_Status", ax=axes[1, 1],
                      palette=[C["teal"], C["red"]])
        axes[1, 1].set_title("Credit History vs Loan Status", fontweight="bold")
        axes[1, 1].set_xlabel("Credit History (1=Good, 0=Bad)")
        axes[1, 1].legend(title="Status", labels=["Approved", "Rejected"],
                          facecolor=CHART_BG, labelcolor=C["text"])

    if "Property_Area" in df.columns:
        sns.countplot(data=df, x="Property_Area", hue="Loan_Status", ax=axes[1, 2],
                      palette=[C["teal"], C["red"]])
        axes[1, 2].set_title("Property Area vs Loan Status", fontweight="bold")
        axes[1, 2].legend(title="Status", labels=["Approved", "Rejected"],
                          facecolor=CHART_BG, labelcolor=C["text"])

    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    num_df = df.select_dtypes(include=[np.number])
    fig2, ax2 = styled_fig(10, 5)
    sns.heatmap(
        num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax2, linewidths=0.5, linecolor="#0B0F14",
        annot_kws={"size": 9, "color": C["text"]},
        cbar_kws={"shrink": 0.8},
    )
    ax2.set_title("Feature Correlation Matrix", fontweight="bold", pad=12)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    st.markdown("---")

    with st.expander("📋  Raw Dataset Preview", expanded=False):
        st.dataframe(
            df.head(50).style.set_properties(**{
                "background-color": C["card"],
                "color":            C["text"],
                "border":           "1px solid #2D3748",
            }),
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 -- MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

def page_insights(art):
    results = art["results"]
    models  = results["models"]

    st.markdown("## 🤖 Model Performance Insights")
    st.markdown(
        "<span style='color:#94A3B8;font-size:0.9rem;'>"
        "Comparative analysis of Decision Tree · Naive Bayes · ANN</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Metrics table
    st.markdown("### 📈 Metrics Comparison")
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    rows = []
    for m_name, m_data in models.items():
        row = {"Model": m_name}
        for k in metric_keys:
            row[k] = round(m_data["metrics"].get(k, 0), 4)
        row["Train Acc"] = round(m_data["metrics"].get("Train_Accuracy", 0), 4)
        rows.append(row)
    comp_df = pd.DataFrame(rows).set_index("Model")
    best_model = comp_df["Accuracy"].idxmax()

    def highlight_best(s):
        return [
            "background-color: #042f2e; color: #2DD4BF; font-weight:bold"
            if v == s.max() else ""
            for v in s
        ]

    styled = (
        comp_df.style
        .apply(highlight_best, subset=metric_keys)
        .format("{:.4f}")
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", C["card"]),
                ("color", C["teal"]),
                ("font-size", "0.8rem"),
                ("text-transform", "uppercase"),
            ]},
            {"selector": "td", "props": [
                ("background-color", C["card"]),
                ("color", C["text"]),
                ("border", "1px solid #2D3748"),
            ]},
        ])
    )
    st.dataframe(styled, use_container_width=True)

    best_acc = "{:.4f}".format(comp_df.loc[best_model, "Accuracy"])
    best_auc = "{:.4f}".format(comp_df.loc[best_model, "AUC"])
    st.markdown(
        "<div style='margin-top:-0.5rem;margin-bottom:1rem;'>"
        "<span class='badge badge-teal'>🏆 Best Model: " + best_model + "</span>"
        "<span class='badge badge-green'>Accuracy: " + best_acc + "</span>"
        "<span class='badge badge-amber'>AUC: " + best_auc + "</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Bar charts
    st.markdown("### 📊 Visual Metric Breakdown")
    fig, axes = styled_multi(2, 3, w=16, h=10)
    all_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    model_names = list(models.keys())
    colors      = [MODEL_COLORS.get(m, C["teal"]) for m in model_names]

    for idx, metric in enumerate(all_metrics):
        r, c = divmod(idx, 3)
        vals = [models[m]["metrics"].get(metric, 0) for m in model_names]
        bars = axes[r, c].bar(model_names, vals, color=colors,
                              edgecolor="#0B0F14", width=0.5, linewidth=1.2)
        axes[r, c].set_title(metric, fontweight="bold")
        axes[r, c].set_ylim(0, 1.12)
        axes[r, c].yaxis.grid(True, alpha=0.15, linestyle="--")
        axes[r, c].set_axisbelow(True)
        for b in bars:
            axes[r, c].text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                "{:.3f}".format(b.get_height()),
                ha="center", va="bottom", color=C["text"], fontsize=9, fontweight="bold",
            )

    axes[1, 2].axis("off")
    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # ROC curves
    st.markdown("### 📉 ROC Curve Comparison")
    fig3, ax3 = styled_fig(9, 5)
    for m_name, m_data in models.items():
        fpr     = m_data.get("fpr", [])
        tpr_    = m_data.get("tpr", [])
        auc_val = m_data["metrics"].get("AUC", 0)
        ax3.plot(fpr, tpr_, linewidth=2.5,
                 color=MODEL_COLORS.get(m_name, C["teal"]),
                 label=m_name + " (AUC=" + "{:.3f}".format(auc_val) + ")")
    ax3.plot([0, 1], [0, 1], "--", color=C["muted"], linewidth=1, label="Random")
    ax3.fill_between([0, 1], [0, 1], alpha=0.04, color=C["muted"])
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve Comparison", fontweight="bold")
    ax3.legend(facecolor=CHART_BG, labelcolor=C["text"], fontsize=10)
    ax3.yaxis.grid(True, alpha=0.1, linestyle="--")
    ax3.xaxis.grid(True, alpha=0.1, linestyle="--")
    st.pyplot(fig3, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Confusion matrices
    st.markdown("### 🔢 Confusion Matrices")
    cm_cols  = st.columns(len(models))
    cm_cmaps = {"Decision Tree": "Blues", "Naive Bayes": "Greens", "ANN": "Purples"}

    for col, (m_name, m_data) in zip(cm_cols, models.items()):
        with col:
            model_col = MODEL_COLORS.get(m_name, C["teal"])
            st.markdown(
                "<h4 style='text-align:center;color:" + model_col + ";'>"
                + m_name + "</h4>",
                unsafe_allow_html=True,
            )
            cm = np.array(m_data["cm"])
            fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
            fig_cm.patch.set_facecolor(CHART_BG)
            ax_cm.set_facecolor(CHART_BG)
            sns.heatmap(
                cm, annot=True, fmt="d",
                cmap=cm_cmaps.get(m_name, "Blues"),
                ax=ax_cm, linewidths=1, linecolor="#0B0F14",
                xticklabels=["Not Approved", "Approved"],
                yticklabels=["Not Approved", "Approved"],
                annot_kws={"size": 13, "weight": "bold", "color": C["text"]},
            )
            ax_cm.set_xlabel("Predicted", color=CHART_TEXT, fontsize=9)
            ax_cm.set_ylabel("Actual",    color=CHART_TEXT, fontsize=9)
            ax_cm.tick_params(colors=CHART_TEXT, labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close()

    st.markdown("---")

    # ANN history
    if results.get("ann_available") and results.get("ann_history"):
        st.markdown("### 🧠 ANN Training History")
        hist = results["ann_history"]
        fig4, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 4))
        fig4.patch.set_facecolor(CHART_BG)
        for ax in [ax_l, ax_a]:
            ax.set_facecolor(CHART_BG)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2D3748")
            ax.tick_params(colors=CHART_TEXT)
            ax.xaxis.label.set_color(CHART_TEXT)
            ax.yaxis.label.set_color(CHART_TEXT)
            ax.title.set_color(C["text"])
            ax.yaxis.grid(True, alpha=0.1, linestyle="--")

        ax_l.plot(hist["loss"],     color=C["teal"],  linewidth=2, label="Train Loss")
        ax_l.plot(hist["val_loss"], color=C["amber"], linewidth=2, label="Val Loss")
        ax_l.set_title("Loss Over Epochs", fontweight="bold")
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("Loss")
        ax_l.legend(facecolor=CHART_BG, labelcolor=C["text"])

        ax_a.plot(hist["accuracy"],     color=C["green"], linewidth=2, label="Train Acc")
        ax_a.plot(hist["val_accuracy"], color=C["red"],   linewidth=2, label="Val Acc")
        ax_a.set_title("Accuracy Over Epochs", fontweight="bold")
        ax_a.set_xlabel("Epoch")
        ax_a.set_ylabel("Accuracy")
        ax_a.legend(facecolor=CHART_BG, labelcolor=C["text"])

        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()
        st.markdown("---")

    # Feature importance
    st.markdown("### 🎯 Feature Importance (Decision Tree)")
    fi = art["feature_importance"]
    fig5, ax5 = styled_fig(9, 4.5)
    colors_fi = [C["teal"] if i < 3 else C["muted"] for i in range(len(fi))]
    bars5 = ax5.barh(
        fi["Feature"][::-1], fi["Importance"][::-1],
        color=list(reversed(colors_fi)), edgecolor="#0B0F14", height=0.6,
    )
    for b in bars5:
        ax5.text(b.get_width() + 0.002, b.get_y() + b.get_height() / 2,
                 "{:.3f}".format(b.get_width()),
                 va="center", color=C["text"], fontsize=9)
    ax5.set_title("Feature Importance Score", fontweight="bold")
    ax5.set_xlabel("Importance")
    ax5.xaxis.grid(True, alpha=0.12, linestyle="--")
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 -- LOAN PREDICTION TOOL
# ════════════════════════════════════════════════════════════════════════════

def predict_with_model(model_name, input_dict, art):
    results      = art["results"]
    le           = art["label_encoders"]
    scl          = art["scaler"]
    feature_cols = results["feature_cols"]

    row = {}
    for col in feature_cols:
        val = input_dict.get(col, 0)
        if col in le:
            try:
                val = int(le[col].transform([str(val)])[0])
            except Exception:
                val = 0
        row[col] = val

    df_input = pd.DataFrame([row])[feature_cols]
    df_input = df_input.apply(pd.to_numeric, errors="coerce").fillna(0)
    df_scaled = scl.transform(df_input)

    if model_name == "Decision Tree":
        prob = art["dt_model"].predict_proba(df_scaled)[0][1]
    elif model_name == "Naive Bayes":
        prob = art["nb_model"].predict_proba(df_scaled)[0][1]
    elif model_name == "ANN" and art["ann_model"] is not None:
        prob = float(art["ann_model"].predict(df_scaled, verbose=0).flatten()[0])
    else:
        prob = 0.5

    return prob >= 0.5, prob


def page_predict(art):
    results = art["results"]
    le      = art["label_encoders"]

    st.markdown("## 🔮 Loan Prediction Tool")
    st.markdown(
        "<span style='color:#94A3B8;font-size:0.9rem;'>"
        "Fill in the applicant details and get an instant prediction</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("### 📝 Applicant Information")
        col1, col2 = st.columns(2)

        with col1:
            gender_opts   = list(le["Gender"].classes_)       if "Gender"       in le else ["Male", "Female"]
            married_opts  = list(le["Married"].classes_)      if "Married"      in le else ["Yes", "No"]
            dep_opts      = list(le["Dependents"].classes_)   if "Dependents"   in le else ["0", "1", "2", "3+"]
            edu_opts      = list(le["Education"].classes_)    if "Education"    in le else ["Graduate", "Not Graduate"]
            se_opts       = list(le["Self_Employed"].classes_) if "Self_Employed" in le else ["Yes", "No"]
            area_opts     = list(le["Property_Area"].classes_) if "Property_Area" in le else ["Urban", "Semiurban", "Rural"]

            gender       = st.selectbox("Gender",        gender_opts,  key="p_gender")
            married      = st.selectbox("Married",       married_opts, key="p_married")
            dependents   = st.selectbox("Dependents",    dep_opts,     key="p_dep")
            education    = st.selectbox("Education",     edu_opts,     key="p_edu")
            self_emp     = st.selectbox("Self Employed", se_opts,      key="p_se")
            prop_area    = st.selectbox("Property Area", area_opts,    key="p_area")

        with col2:
            app_income  = st.number_input("Applicant Income ($)",    0,  100000, 5000,  500, key="p_income")
            coapp_inc   = st.number_input("Co-applicant Income ($)", 0,   50000, 1500,  250, key="p_coincome")
            loan_amt    = st.number_input("Loan Amount ($K)",        1,     700,  150,   10, key="p_loan")
            loan_term   = st.selectbox("Loan Term (months)",
                                       [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                                       index=9, key="p_term")
            credit_hist = st.radio("Credit History", ["Good (1)", "Bad (0)"], key="p_credit")

        st.markdown("---")
        model_choices = [
    "Decision Tree",
    "Naive Bayes",
    "ANN"
]
        model_sel   = st.selectbox("Model to Use", model_choices, key="p_model")
        predict_btn = st.button("🔍  Predict Loan Status", use_container_width=True, type="primary")

    with right:
        st.markdown("### 📊 Prediction Result")

        if predict_btn:
            credit_val = 1 if "Good" in credit_hist else 0
            input_dict = {
                "Gender":            gender,
                "Married":           married,
                "Dependents":        dependents,
                "Education":         education,
                "Self_Employed":     self_emp,
                "ApplicantIncome":   app_income,
                "CoapplicantIncome": coapp_inc,
                "LoanAmount":        loan_amt,
                "Loan_Amount_Term":  loan_term,
                "Credit_History":    credit_val,
                "Property_Area":     prop_area,
            }

            approved, prob = predict_with_model(model_sel, input_dict, art)
            pct = prob * 100

            if approved:
                st.markdown(
                    "<div class='result-approved'>"
                    "<div class='result-title' style='color:#22C55E'>✅ APPROVED</div>"
                    "<div style='font-size:1rem;color:#94A3B8;margin-top:0.5rem;'>"
                    "Loan likely to be approved</div>"
                    "<div style='font-size:2.5rem;font-weight:800;color:#22C55E;margin:0.8rem 0;'>"
                    + "{:.1f}%".format(pct) +
                    "</div>"
                    "<div style='color:#94A3B8;font-size:0.85rem;'>approval confidence</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                reject_pct = "{:.1f}%".format(100 - pct)
                st.markdown(
                    "<div class='result-rejected'>"
                    "<div class='result-title' style='color:#EF4444'>❌ REJECTED</div>"
                    "<div style='font-size:1rem;color:#94A3B8;margin-top:0.5rem;'>"
                    "Loan likely to be rejected</div>"
                    "<div style='font-size:2.5rem;font-weight:800;color:#EF4444;margin:0.8rem 0;'>"
                    + reject_pct +
                    "</div>"
                    "<div style='color:#94A3B8;font-size:0.85rem;'>rejection confidence</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Approval Probability**")
            st.progress(float(prob))

            st.markdown("<br>", unsafe_allow_html=True)
            if pct >= 75:
                risk_label = "Low Risk"
                risk_color = C["green"]
                risk_badge = "badge-green"
            elif pct >= 50:
                risk_label = "Moderate Risk"
                risk_color = C["amber"]
                risk_badge = "badge-amber"
            else:
                risk_label = "High Risk"
                risk_color = C["red"]
                risk_badge = "badge-red"

            credit_label = "Good" if credit_val == 1 else "Bad"
            conf_str     = "{:.2f}%".format(pct)
            loan_str     = "${:,}K".format(loan_amt)

            st.markdown(
                "<div class='card'>"
                "<div style='font-size:0.78rem;color:#94A3B8;text-transform:uppercase;"
                "letter-spacing:.06em;margin-bottom:0.7rem;'>Risk Assessment</div>"
                "<span class='badge " + risk_badge + "'>" + risk_label + "</span>"
                "<div style='margin-top:1rem;display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;'>"
                "<div><div style='font-size:0.72rem;color:#94A3B8;'>Model Used</div>"
                "<div style='font-weight:700;color:#14B8A6;margin-top:2px;'>" + model_sel + "</div></div>"
                "<div><div style='font-size:0.72rem;color:#94A3B8;'>Confidence</div>"
                "<div style='font-weight:700;color:" + risk_color + ";margin-top:2px;'>" + conf_str + "</div></div>"
                "<div><div style='font-size:0.72rem;color:#94A3B8;'>Credit History</div>"
                "<div style='font-weight:700;color:#F8FAFC;margin-top:2px;'>" + credit_label + "</div></div>"
                "<div><div style='font-size:0.72rem;color:#94A3B8;'>Loan Amount</div>"
                "<div style='font-weight:700;color:#F8FAFC;margin-top:2px;'>" + loan_str + "</div></div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

            credit_icon = "🟢" if credit_val == 1 else "🔴"
            income_icon = "🟢" if app_income > 3000 else "🔴"
            loan_icon   = "🟢" if loan_amt < 200 else "🟡"
            edu_icon    = "🟢" if education == "Graduate" else "🟡"

            st.markdown(
                "<div class='card' style='margin-top:0.5rem;'>"
                "<div style='font-size:0.78rem;color:#94A3B8;text-transform:uppercase;"
                "letter-spacing:.06em;margin-bottom:0.7rem;'>Key Influencing Factors</div>"
                "<div style='font-size:0.85rem;line-height:2;'>"
                + credit_icon + " Credit History &nbsp;·&nbsp; "
                + income_icon + " Income Level &nbsp;·&nbsp; "
                + loan_icon   + " Loan Amount &nbsp;·&nbsp; "
                + edu_icon    + " Education"
                "</div></div>",
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                "<div style='text-align:center;padding:3rem 1rem;"
                "border:2px dashed #2D3748;border-radius:16px;margin-top:1rem;'>"
                "<div style='font-size:3rem;margin-bottom:1rem;'>🔮</div>"
                "<div style='font-size:1.1rem;font-weight:600;color:#94A3B8;'>"
                "Fill in the form and click<br>"
                "<span style='color:#14B8A6;'>Predict Loan Status</span></div>"
                "<div style='font-size:0.8rem;color:#4B5563;margin-top:0.8rem;'>"
                "Results appear here instantly</div></div>",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Available Models**")
            model_info = {
                "Decision Tree": ("🌳", "Fast · Interpretable",   "#14B8A6"),
                "Naive Bayes":   ("📊", "Probabilistic · Robust", "#F59E0B"),
                "ANN":           ("🧠", "Deep Learning · Precise", "#818CF8"),
            }
            for name, (icon, desc, col_color) in model_info.items():
                avail   = name != "ANN" or art["ann_model"] is not None
                opacity = "1" if avail else "0.4"
                unavail = "" if avail else "<span style='color:#EF4444;font-size:0.7rem;'> · Unavailable</span>"
                st.markdown(
                    "<div class='card' style='margin-bottom:0.5rem;opacity:" + opacity + ";'>"
                    "<span style='font-size:1.3rem;'>" + icon + "</span>"
                    "<span style='font-weight:700;color:" + col_color + ";margin-left:0.5rem;'>"
                    + name + "</span><br>"
                    "<span style='font-size:0.78rem;color:#94A3B8;'>" + desc + "</span>"
                    + unavail + "</div>",
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3.5 -- MODEL TUNING
# ════════════════════════════════════════════════════════════════════════════

def page_tuning(art):
    st.markdown("## ⚙️ Model Tuning & Live Training")
    st.markdown(
        "<span style='color:#94A3B8;font-size:0.9rem;'>"
        "Configure hyperparameters for Decision Tree, Naive Bayes, and ANN, and retrain them live on the dataset.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    is_custom = st.session_state.get("is_custom", False)
    if is_custom:
        st.markdown(
            "<div class='card' style='border: 1px solid #14B8A6; background-color: #042f2e;'>"
            "<span style='color:#2DD4BF;font-weight:700;'>🟢 Active: Custom Trained Models</span><br>"
            "<span style='font-size:0.82rem;color:#94A3B8;'>The metrics, charts, and prediction tool are currently using your customized models.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Reset to Pre-trained Defaults", type="secondary"):
            st.session_state.active_artefacts = load_artefacts()
            st.session_state.is_custom = False
            st.success("Restored pre-trained defaults!")
            st.rerun()
    else:
        st.markdown(
            "<div class='card' style='border: 1px solid #2D3748; background-color: #1F2937;'>"
            "<span style='color:#F8FAFC;font-weight:700;'>🔵 Active: Pre-trained Default Models</span><br>"
            "<span style='font-size:0.82rem;color:#94A3B8;'>Using the baseline models trained in `train_models.py`. Adjust settings below to customize.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### 🎛️ Hyperparameters Configuration")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🌳 Decision Tree")
        dt_crit = st.selectbox("Split Criterion", ["gini", "entropy"], key="tune_dt_crit")
        
        dt_depth_limit = st.checkbox("Limit Max Depth", value=True, key="tune_dt_depth_limit")
        if dt_depth_limit:
            dt_depth = st.slider("Max Depth", 1, 25, 5, key="tune_dt_depth")
        else:
            dt_depth = None
            
        dt_split = st.slider("Min Samples Split", 2, 50, 10, key="tune_dt_split")
        dt_leaf = st.slider("Min Samples Leaf", 1, 50, 5, key="tune_dt_leaf")

    with col2:
        st.markdown("#### 📊 Naive Bayes")
        nb_smooth = st.selectbox(
            "Variance Smoothing",
            [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            index=2,
            format_func=lambda x: f"{x:.0e}",
            key="tune_nb_smooth"
        )
        st.markdown(
            "<div style='font-size:0.8rem;color:#94A3B8;margin-top:1rem;line-height:1.4;'>"
            "Naive Bayes is a stable statistical classifier. The variance smoothing parameter "
            "adds a portion of the largest variance of all features to the variances to stabilize calculations."
            "</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown("#### 🧠 Neural Network (ANN)")
        ann_lr = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01, 0.1], index=1, key="tune_ann_lr")
        ann_epochs = st.slider("Training Epochs", 10, 150, 80, key="tune_ann_epochs")
        ann_batch = st.selectbox("Batch Size", [8, 16, 32, 64], index=1, key="tune_ann_batch")
        ann_neurons1 = st.slider("Layer 1 Neurons", 8, 128, 64, key="tune_ann_neurons1")
        ann_neurons2 = st.slider("Layer 2 Neurons", 4, 64, 32, key="tune_ann_neurons2")
        ann_dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3, key="tune_ann_dropout")

    st.markdown("---")
    
    if st.button("🚀 Train & Update Models", type="primary", use_container_width=True):
        with st.spinner("Training models live on the dataset..."):
            try:
                # 1. Fetch raw data from active artefacts
                raw_df = art["df"]
                
                # 2. Process and split
                X_train, X_test, y_train, y_test, new_scaler, new_encoders = preprocess_and_split(raw_df)
                
                # 3. Train Decision Tree
                dt_clf = DecisionTreeClassifier(
                    criterion=dt_crit,
                    max_depth=dt_depth,
                    min_samples_split=dt_split,
                    min_samples_leaf=dt_leaf,
                    random_state=42
                )
                dt_clf.fit(X_train, y_train)
                dt_pred = dt_clf.predict(X_test)
                dt_prob = dt_clf.predict_proba(X_test)[:, 1]
                dt_train_pred = dt_clf.predict(X_train)
                
                dt_metrics = {
                    "Accuracy": accuracy_score(y_test, dt_pred),
                    "Precision": precision_score(y_test, dt_pred),
                    "Recall": recall_score(y_test, dt_pred),
                    "F1-Score": f1_score(y_test, dt_pred),
                    "AUC": roc_auc_score(y_test, dt_prob),
                    "Train_Accuracy": accuracy_score(y_train, dt_train_pred),
                }
                fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
                cm_dt = confusion_matrix(y_test, dt_pred).tolist()
                
                # Feature importance
                new_fi = pd.DataFrame({
                    "Feature": X_train.columns.tolist(),
                    "Importance": dt_clf.feature_importances_
                }).sort_values("Importance", ascending=False).reset_index(drop=True)
                
                # 4. Train Naive Bayes
                nb_clf = GaussianNB(var_smoothing=nb_smooth)
                nb_clf.fit(X_train, y_train)
                nb_pred = nb_clf.predict(X_test)
                nb_prob = nb_clf.predict_proba(X_test)[:, 1]
                nb_train_pred = nb_clf.predict(X_train)
                
                nb_metrics = {
                    "Accuracy": accuracy_score(y_test, nb_pred),
                    "Precision": precision_score(y_test, nb_pred),
                    "Recall": recall_score(y_test, nb_pred),
                    "F1-Score": f1_score(y_test, nb_pred),
                    "AUC": roc_auc_score(y_test, nb_prob),
                    "Train_Accuracy": accuracy_score(y_train, nb_train_pred),
                }
                fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_prob)
                cm_nb = confusion_matrix(y_test, nb_pred).tolist()
                
                # 5. Train ANN
                ann_available = False
                ann_clf = None
                ann_metrics = {}
                fpr_ann, tpr_ann = [], []
                cm_ann = []
                ann_history = {}
                
                try:
                    import tensorflow as tf
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Dense, Dropout
                    from tensorflow.keras.optimizers import Adam
                    
                    ann_clf = Sequential([
                        Dense(ann_neurons1, activation="relu", input_shape=(X_train.shape[1],)),
                        Dropout(ann_dropout),
                        Dense(ann_neurons2, activation="relu"),
                        Dropout(ann_dropout / 1.5),
                        Dense(16, activation="relu"),
                        Dense(1, activation="sigmoid"),
                    ])
                    
                    ann_clf.compile(
                        optimizer=Adam(learning_rate=ann_lr),
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                    
                    history = ann_clf.fit(
                        X_train, y_train,
                        epochs=ann_epochs,
                        batch_size=ann_batch,
                        validation_split=0.2,
                        verbose=0,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor="val_loss", patience=15, restore_best_weights=True
                            )
                        ],
                    )
                    
                    ann_prob_test = ann_clf.predict(X_test, verbose=0).flatten()
                    ann_pred_test = (ann_prob_test >= 0.5).astype(int)
                    ann_prob_train = ann_clf.predict(X_train, verbose=0).flatten()
                    ann_pred_train = (ann_prob_train >= 0.5).astype(int)
                    
                    ann_metrics = {
                        "Accuracy": accuracy_score(y_test, ann_pred_test),
                        "Precision": precision_score(y_test, ann_pred_test),
                        "Recall": recall_score(y_test, ann_pred_test),
                        "F1-Score": f1_score(y_test, ann_pred_test),
                        "AUC": roc_auc_score(y_test, ann_prob_test),
                        "Train_Accuracy": accuracy_score(y_train, ann_pred_train),
                    }
                    fpr_ann, tpr_ann, _ = roc_curve(y_test, ann_prob_test)
                    cm_ann = confusion_matrix(y_test, ann_pred_test).tolist()
                    
                    ann_history = {
                        "loss": [float(l) for l in history.history["loss"]],
                        "val_loss": [float(l) for l in history.history["val_loss"]],
                        "accuracy": [float(a) for a in history.history["accuracy"]],
                        "val_accuracy": [float(a) for a in history.history["val_accuracy"]],
                    }
                    ann_available = True
                except Exception as ann_err:
                    st.warning(f"Failed to train Keras ANN: {ann_err}. Decision Tree and Naive Bayes will still be updated.")
                    
                # 6. Update results dict
                new_results = {
                    "feature_cols": X_train.columns.tolist(),
                    "categorical_cols": [c for c in raw_df.columns if raw_df[c].dtype == "object" and c != "Loan_Status" and c != "Loan_ID"],
                    "numerical_cols": [c for c in raw_df.columns if raw_df[c].dtype != "object" and c != "Loan_Status" and c != "Loan_ID" and c != "Unnamed: 0"],
                    "ann_available": ann_available,
                    "dataset_shape": list(raw_df.shape),
                    "approval_rate": float((raw_df["Loan_Status"].isin([1, "Y", "Yes", "YES", "1", True])).mean()),
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
                    new_results["models"]["ANN"] = {
                        "metrics": ann_metrics,
                        "cm": cm_ann,
                        "fpr": fpr_ann.tolist(),
                        "tpr": tpr_ann.tolist(),
                    }
                    
                # 7. Store updated artefacts in session state
                st.session_state.active_artefacts = {
                    "results": new_results,
                    "scaler": new_scaler,
                    "label_encoders": new_encoders,
                    "cat_imputer": art["cat_imputer"],
                    "num_imputer": art["num_imputer"],
                    "dt_model": dt_clf,
                    "nb_model": nb_clf,
                    "ann_model": ann_clf,
                    "feature_importance": new_fi,
                    "df": raw_df,
                }
                st.session_state.is_custom = True
                st.success("🎉 Models successfully trained and updated!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during training: {e}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 -- SOURCE CODE VIEWER
# ════════════════════════════════════════════════════════════════════════════

def page_code(art):
    st.markdown("## 📝 Complete Source Code")
    st.markdown(
        "<span style='color:#94A3B8;font-size:0.9rem;'>"
        "Review the full implementation of every component in one place</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    code_files = [
        ("🏋️  Model Training — train_models.py",
         os.path.join(BASE_DIR, "train_models.py"),
         "Downloads the dataset, preprocesses it, trains Decision Tree, "
         "Naive Bayes & ANN models, and serializes all artefacts to disk."),
        ("🖥️  Streamlit Dashboard — dashboard.py",
         os.path.join(BASE_DIR, "dashboard.py"),
         "The interactive web dashboard with EDA, model comparison, "
         "live prediction, and this source code viewer."),
        ("📓  Notebook Code — notebook_code.py",
         os.path.join(BASE_DIR, "notebook_code.py"),
         "The Python code corresponding to the Jupyter Notebook cells "
         "used for initial research, plotting, and experimentation."),
    ]

    for title, path, description in code_files:
        with st.expander(title, expanded=False):
            st.markdown(
                f"<div style='color:#94A3B8;font-size:0.82rem;margin-bottom:0.8rem;'>"
                f"{description}</div>",
                unsafe_allow_html=True,
            )
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
                st.markdown(
                    f"<div style='color:#94A3B8;font-size:0.72rem;margin-bottom:0.3rem;'>"
                    f"📏 {len(code.splitlines())} lines  ·  "
                    f"{len(code):,} characters</div>",
                    unsafe_allow_html=True,
                )
                st.code(code, language="python")
            else:
                st.warning(f"File not found: {os.path.basename(path)}")

    st.markdown("---")
    st.markdown(
        "<div class='card' style='text-align:center;'>"
        "<div style='font-size:0.78rem;color:#94A3B8;text-transform:uppercase;"
        "letter-spacing:.06em;margin-bottom:0.5rem;'>Project Info</div>"
        "<div style='font-size:0.85rem;color:#F8FAFC;line-height:1.8;'>"
        "💳 <span style='color:#14B8A6;font-weight:700;'>LoanSight</span> — "
        "ML Loan Approval Dashboard<br>"
        "Built with Python · Scikit-Learn · TensorFlow · Streamlit"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    render_sidebar()
    
    if "active_artefacts" not in st.session_state:
        st.session_state.active_artefacts = load_artefacts()
        st.session_state.is_custom = False

    art = st.session_state.active_artefacts

    if art is None:
        st.error("Model artefacts not found. Please run:  python train_models.py")
        st.code("python train_models.py", language="bash")
        st.stop()

    page = st.session_state.get("page", "explorer")

    if page == "explorer":
        page_explorer(art)
    elif page == "insights":
        page_insights(art)
    elif page == "tuning":
        page_tuning(art)
    elif page == "predict":
        page_predict(art)
    elif page == "code":
        page_code(art)


if __name__ == "__main__":
    main()
