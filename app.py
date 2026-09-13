"""
Credit Card Fraud Detection - Streamlit Application
Dataset: creditcard.csv (284,807 transactions, 492 fraudulent)
Features: Time, V1-V28 (PCA components), Amount, Class
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score, precision_score, recall_score
)
from sklearn.utils import resample

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root{
        --bg:#f6f8fb;
        --card1:#667eea;
        --card2:#764ba2;
        --accent:#f5576c;
        --muted:#6b7280;
        --heading:#eaf2ff;
        --heading-soft:#b7cae8;
        --heading-accent:#3db6ff;
    }
    body { background: var(--bg); }
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: var(--heading);
        text-align: center;
        padding: 0.8rem 0;
        margin-bottom: 0.2rem;
        letter-spacing: 0.02em;
        text-shadow: 0 4px 18px rgba(61, 182, 255, 0.28);
    }
    .subtitle {
        text-align: center;
        color: var(--heading-soft);
        font-size: 1.05rem;
        margin-bottom: 1.4rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 18px rgba(15,23,42,0.06);
    }
    .card-title { font-size:0.95rem; font-weight:700; }
    .card-value { font-size:1.5rem; margin-top:6px; font-weight:800; }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #dce9ff;
        border-left: 4px solid var(--heading-accent);
        padding: 8px 12px;
        border-radius: 8px;
        background: linear-gradient(90deg, rgba(25, 44, 82, 0.55), rgba(25, 44, 82, 0.08));
        margin: 1.2rem 0 0.8rem 0;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1226 0%, #14213e 100%);
        color: #e6eef8;
    }
    div[data-testid="stSidebar"] .stMarkdown, 
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label {
        color: #e6eef8 !important;
    }
    .btn-primary {
        background: linear-gradient(90deg, var(--card1), var(--card2));
        color: white; padding: 0.6rem 1rem; border-radius:10px; font-weight:700;
    }
    .small-muted { color:var(--muted); font-size:0.9rem }
    .model-loaded { font-size:2.2rem; font-weight:800; color: #ffffff; opacity:0.95; margin-top:1.4rem; text-shadow: 0 6px 20px rgba(0,0,0,0.6); }
    .pill { display:inline-block; padding:6px 12px; border-radius:10px; background:linear-gradient(90deg,#064e2a,#087f44); color:#dcfce7; font-weight:800; font-family:inherit; }
    .table-wrapper { border-radius:10px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# Consistent color palette for charts and accents
COLORS = {
    'legit': '#4facfe',
    'fraud': '#f5576c',
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#3db6ff',
    'bg_light': '#f8f9fa',
}

# UI helper: render an HTML metric card (used in several places)
def render_metric_card(title, value, subtitle='', color1="#cbcfe2", color2="#f4f1f8"):
    html = f"""
    <div class="metric-card" style="background: linear-gradient(135deg, {color1} 0%, {color2} 100%);">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      <div class="small-muted">{subtitle}</div>
    </div>
    """
    return st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 Fraud Detection")
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")

    uploaded_file = st.file_uploader("📂 Upload creditcard.csv", type=["csv"])

    model_choice = st.selectbox(
        "🤖 Select Model",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )

    balance_method = st.selectbox(
        "⚖️ Handle Class Imbalance",
        ["SMOTE-like Oversampling", "Undersampling", "None (class_weight)"]
    )

    test_size = st.slider("📊 Test Split Size", 0.1, 0.4, 0.2, 0.05)

    st.markdown("---")
    st.markdown("### ℹ️ Dataset Info")
    st.info("284,807 transactions\n\n492 fraudulent (0.17%)\n\n28 PCA features + Time + Amount")

    st.markdown("---")
    st.markdown("### 📌 Navigation")
    page = st.radio("Go to", [
        "🏠 Overview",
        "📊 Data Analysis",
        "🤖 Train & Evaluate",
        "🔍 Predict Transaction"
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        return df
    # Try common local paths automatically
    import os
    candidate_paths = [
        r"C:\Users\User\Downloads\creditcard.csv",
        r"C:\Users\User\Desktop\creditcard.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "creditcard.csv"),
        os.path.join(os.path.expanduser("~"), "Downloads", "creditcard.csv"),
        os.path.join(os.path.expanduser("~"), "Desktop", "creditcard.csv"),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔐 Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning · Streamlit · IEEE Project | Built with Python & Scikit-learn</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset..."):
    df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ Please upload `creditcard.csv` using the sidebar uploader.")
    st.markdown("""
    **Expected columns:** `Time`, `V1` to `V28`, `Amount`, `Class`
    
    - `Class = 0` → Legitimate transaction  
    - `Class = 1` → Fraudulent transaction
    """)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="section-header">📋 Dataset Summary</div>', unsafe_allow_html=True)

    total = len(df)
    fraud = df['Class'].sum()
    legit = total - fraud
    fraud_pct = (fraud / total) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Transactions", f"{total:,}", subtitle=f"{total:,} transactions", color1=COLORS['primary'], color2=COLORS['secondary'])
    with col2:
        render_metric_card("Legitimate", f"{legit:,}", subtitle=f"{100-fraud_pct:.2f}% of total", color1=COLORS['legit'], color2=COLORS['accent'])
    with col3:
        render_metric_card("Fraudulent", f"{fraud:,}", subtitle=f"{fraud_pct:.4f}% of total", color1=COLORS['fraud'], color2=COLORS['secondary'])
    with col4:
        render_metric_card("Features", "30 (V1-V28 + Time + Amount)", subtitle='PCA + Time + Amount', color1=COLORS['primary'], color2=COLORS['accent'])

    st.markdown('<div class="section-header">📌 How It Works</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Step 1: Data Preprocessing**
        - Scale `Amount` and `Time`
        - Handle class imbalance
        - Train/test split
        """)
    with col2:
        st.markdown("""
        **Step 2: Model Training**
        - Logistic Regression
        - Random Forest
        - Gradient Boosting
        """)
    with col3:
        st.markdown("""
        **Step 3: Evaluation**
        - ROC-AUC Score
        - Precision-Recall Curve
        - Confusion Matrix
        """)

    st.markdown('<div class="section-header">👁️ Sample Data</div>', unsafe_allow_html=True)
    # Styled preview table (dark themed)
    try:
        styled = df.head(10).style.set_table_styles([
            {"selector": "th", "props": [("background-color", "#0b1226"), ("color", "#e6eef8")]},
            {"selector": "td", "props": [("background-color", "#0f172a"), ("color", "#e6eef8")]}
        ]).set_properties(**{"border-radius": "6px"})
        st.dataframe(styled, width='stretch')
    except Exception:
        st.dataframe(df.head(10), width='stretch')

    # Class distribution pie
    st.markdown('<div class="section-header">🥧 Class Distribution</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(COLORS['bg_light'])

    # Pie chart
    axes[0].pie([legit, fraud], labels=["Legitimate", "Fraud"],
                colors=[COLORS['legit'], COLORS['fraud']], autopct='%1.3f%%',
                startangle=90, shadow=True,
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title("Transaction Distribution", fontsize=14, fontweight='bold')

    # Bar chart
    bars = axes[1].bar(["Legitimate", "Fraudulent"], [legit, fraud],
                       color=[COLORS['legit'], COLORS['fraud']], edgecolor='white',
                       linewidth=1.5, width=0.5)
    axes[1].set_title("Count Comparison", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Number of Transactions")
    axes[1].set_yscale("log")
    for bar, val in zip(bars, [legit, fraud]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
                     f'{val:,}', ha='center', fontweight='bold')
    axes[1].set_facecolor(COLORS['bg_light'])

    plt.tight_layout()
    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Data Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Data Analysis":
    st.markdown('<div class="section-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Feature Distributions", "🔗 Correlation", "💰 Amount Analysis"])

    with tab1:
        st.markdown("**Distribution of Transaction Amount by Class**")
        fig, ax = plt.subplots(figsize=(10, 4))
        fraud_amounts = df[df['Class'] == 1]['Amount']
        legit_amounts = df[df['Class'] == 0]['Amount']
        ax.hist(legit_amounts.clip(upper=500), bins=60, alpha=0.6,
            color=COLORS['legit'], label='Legitimate', density=True)
        ax.hist(fraud_amounts.clip(upper=500), bins=60, alpha=0.6,
            color=COLORS['fraud'], label='Fraudulent', density=True)
        ax.set_xlabel("Amount (clipped at 500)")
        ax.set_ylabel("Density")
        ax.set_title("Amount Distribution: Legitimate vs Fraudulent")
        ax.legend()
        st.pyplot(fig)

        st.markdown("**Select PCA Feature to Visualize**")
        feature = st.selectbox("Feature", [f"V{i}" for i in range(1, 15)])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df[df['Class'] == 0][feature], bins=80, alpha=0.6,
            color=COLORS['legit'], label='Legitimate', density=True)
        ax.hist(df[df['Class'] == 1][feature], bins=80, alpha=0.6,
            color=COLORS['fraud'], label='Fraudulent', density=True)
        ax.set_title(f"Distribution of {feature}")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.markdown("**Correlation Heatmap (sample of key features)**")
        key_cols = ['V1','V2','V3','V4','V9','V10','V11','V12','V14','V17','Amount','Class']
        corr = df[key_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, linewidths=0.5,
                    annot_kws={'size': 8})
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        # Top correlations with Class
        st.markdown("**Top Features Correlated with Fraud (Class)**")
        class_corr = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        class_corr.head(15).plot(kind='bar', ax=ax,
                      color=[COLORS['fraud'] if x > 0.2 else COLORS['primary'] for x in class_corr.head(15)],
                      edgecolor='white')
        ax.set_title("Absolute Correlation with Fraud Label (Top 15)")
        ax.set_ylabel("|Correlation|")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Fraudulent Transaction Statistics**")
            fraud_df = df[df['Class'] == 1]
            stats = fraud_df['Amount'].describe()
            st.dataframe(stats.to_frame("Fraud Amount Stats"), width='stretch')

        with col2:
            st.markdown("**Legitimate Transaction Statistics**")
            legit_df = df[df['Class'] == 0]
            stats2 = legit_df['Amount'].describe()
            st.dataframe(stats2.to_frame("Legit Amount Stats"), width='stretch')

        # Box plot
        fig, ax = plt.subplots(figsize=(8, 5))
        data_to_plot = [
            df[df['Class'] == 0]['Amount'].clip(upper=1000).values,
            df[df['Class'] == 1]['Amount'].clip(upper=1000).values
        ]
        bp = ax.boxplot(data_to_plot, patch_artist=True,
                labels=['Legitimate', 'Fraudulent'])
        bp['boxes'][0].set_facecolor(COLORS['legit'])
        bp['boxes'][1].set_facecolor(COLORS['fraud'])
        ax.set_title("Amount Distribution Boxplot (clipped at 1000)")
        ax.set_ylabel("Amount ($)")
        st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Train & Evaluate":
    st.markdown('<div class="section-header">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)

    if st.button("🚀 Train Model", use_container_width=True, type="primary"):
        # ── Preprocessing ──────────────────────────────────────────────────
        with st.spinner("Preprocessing data..."):
            df_model = df.copy()
            scaler = StandardScaler()
            # Fit scaler on both Amount and Time together so feature names match on transform
            df_model[['Amount', 'Time']] = scaler.fit_transform(df_model[['Amount', 'Time']])

            X = df_model.drop('Class', axis=1)
            y = df_model['Class']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        with st.spinner("Handling class imbalance..."):
            if balance_method == "SMOTE-like Oversampling":
                # Manual oversampling (no imbalanced-learn dependency)
                X_train_df = pd.DataFrame(X_train)
                y_train_s = pd.Series(y_train.values)
                fraud_idx = y_train_s[y_train_s == 1].index
                legit_idx = y_train_s[y_train_s == 0].index

                fraud_X = X_train_df.iloc[fraud_idx]
                legit_X = X_train_df.iloc[legit_idx]

                fraud_upsampled_X = resample(fraud_X, replace=True,
                                             n_samples=len(legit_X) // 2,
                                             random_state=42)
                fraud_upsampled_y = pd.Series([1] * len(fraud_upsampled_X))

                X_train_bal = pd.concat([legit_X, fraud_upsampled_X])
                y_train_bal = pd.concat([pd.Series([0] * len(legit_X)), fraud_upsampled_y])

                X_train, y_train = X_train_bal.values, y_train_bal.values

            elif balance_method == "Undersampling":
                X_train_df = pd.DataFrame(X_train)
                y_train_s = pd.Series(y_train.values)
                fraud_idx = y_train_s[y_train_s == 1].index
                legit_idx = y_train_s[y_train_s == 0].index

                legit_downsampled_idx = resample(legit_idx, replace=False,
                                                  n_samples=len(fraud_idx) * 10,
                                                  random_state=42)
                X_train = pd.concat([X_train_df.iloc[fraud_idx],
                                     X_train_df.iloc[legit_downsampled_idx]]).values
                y_train = np.concatenate([np.ones(len(fraud_idx)),
                                          np.zeros(len(legit_downsampled_idx))])

        # ── Train ───────────────────────────────────────────────────────────
        with st.spinner(f"Training {model_choice}..."):
            t0 = time.time()

            if model_choice == "Logistic Regression":
                model = LogisticRegression(
                    max_iter=1000, class_weight='balanced', random_state=42
                )
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=8,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4,
                    learning_rate=0.1, random_state=42
                )

            model.fit(X_train, y_train)
            train_time = time.time() - t0

        # ── Predict ─────────────────────────────────────────────────────────
        with st.spinner("Evaluating..."):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc  = average_precision_score(y_test, y_proba)
            f1      = f1_score(y_test, y_pred)
            prec    = precision_score(y_test, y_pred)
            rec     = recall_score(y_test, y_pred)
            cm      = confusion_matrix(y_test, y_pred)

        # ── Display Metrics ─────────────────────────────────────────────────
        st.success(f"✅ Model trained in {train_time:.2f}s")

        # Prominent 'Model is loaded' header and ROC pill (UI/UX)
        st.markdown('<div class="model-loaded">Model is loaded</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top:8px; display:flex; align-items:center; gap:12px">
          <div style="color:#6b7280; font-weight:700">ROC AUC:</div>
          <div class="pill">%s</div>
        </div>
        """ % f"{roc_auc:.12f}", unsafe_allow_html=True)

        # ── Plots ───────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"{model_choice} Evaluation Results", fontsize=15, fontweight='bold')

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Legit', 'Fraud'],
                    yticklabels=['Legit', 'Fraud'],
                    linewidths=2, linecolor='white',
                    annot_kws={'size': 16, 'weight': 'bold'})
        axes[0].set_title("Confusion Matrix", fontweight='bold')
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[1].plot(fpr, tpr, color=COLORS['primary'], lw=2,
                 label=f'ROC AUC = {roc_auc:.4f}')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].fill_between(fpr, tpr, alpha=0.1, color=COLORS['primary'])
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve", fontweight='bold')
        axes[1].legend()
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.02])

        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        axes[2].plot(recall_vals, precision_vals, color=COLORS['fraud'], lw=2,
                 label=f'PR AUC = {pr_auc:.4f}')
        axes[2].fill_between(recall_vals, precision_vals, alpha=0.1, color=COLORS['fraud'])
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title("Precision-Recall Curve", fontweight='bold')
        axes[2].legend()
        axes[2].set_xlim([0, 1])
        axes[2].set_ylim([0, 1.02])

        plt.tight_layout()
        st.pyplot(fig)

        # Classification Report
        st.markdown('<div class="section-header">📋 Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred,
                                       target_names=['Legitimate', 'Fraudulent'],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues',
                                  subset=['precision', 'recall', 'f1-score']),
                 width='stretch')

        # Feature Importance (if applicable)
        if hasattr(model, 'feature_importances_'):
            st.markdown('<div class="section-header">🔍 Feature Importance</div>', unsafe_allow_html=True)
            importances = model.feature_importances_
            feat_names = X.columns.tolist()
            feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = [COLORS['fraud'] if i < 5 else COLORS['primary'] for i in range(len(feat_imp[:15]))]
            feat_imp[:15].plot(kind='bar', ax=ax, color=colors, edgecolor='white')
            ax.set_title("Top 15 Feature Importances", fontweight='bold')
            ax.set_ylabel("Importance Score")
            ax.tick_params(axis='x', rotation=45)
            red_patch = mpatches.Patch(color=COLORS['fraud'], label='Top 5 Features')
            blue_patch = mpatches.Patch(color=COLORS['primary'], label='Other Features')
            ax.legend(handles=[red_patch, blue_patch])
            plt.tight_layout()
            st.pyplot(fig)

        elif hasattr(model, 'coef_'):
            st.markdown('<div class="section-header">🔍 Feature Coefficients</div>', unsafe_allow_html=True)
            coefs = pd.Series(model.coef_[0], index=X.columns).abs().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            coefs[:15].plot(kind='bar', ax=ax, color=COLORS['primary'], edgecolor='white')
            ax.set_title("Top 15 Feature Coefficients (Absolute)", fontweight='bold')
            ax.set_ylabel("|Coefficient|")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        # Save model to session state for prediction tab
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['feature_cols'] = X.columns.tolist()
        st.session_state['model_name'] = model_choice
        st.session_state['roc_auc'] = roc_auc

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Predict Transaction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Predict Transaction":
    st.markdown('<div class="section-header">🔍 Predict a Transaction</div>', unsafe_allow_html=True)

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first on the **Train & Evaluate** page.")
    else:
        st.info(f"Using: **{st.session_state['model_name']}** | ROC-AUC: **{st.session_state['roc_auc']:.4f}**")

        st.markdown("#### Option 1: Use a random transaction from dataset")
        col1, col2 = st.columns(2)
        with col1:
            pick_type = st.radio("Pick sample type", ["Random", "Random Fraudulent", "Random Legitimate"])
        with col2:
            if st.button("🎲 Pick Sample", use_container_width=True):
                if pick_type == "Random Fraudulent":
                    sample = df[df['Class'] == 1].sample(1).iloc[0]
                elif pick_type == "Random Legitimate":
                    sample = df[df['Class'] == 0].sample(1).iloc[0]
                else:
                    sample = df.sample(1).iloc[0]
                st.session_state['sample'] = sample

        if 'sample' in st.session_state:
            sample = st.session_state['sample']
            true_label = int(sample['Class'])

            # Predict
            model = st.session_state['model']
            scaler = st.session_state['scaler']

            input_df = pd.DataFrame([sample.drop('Class')])
            # Transform both Amount and Time using the trained scaler
            input_df[['Amount', 'Time']] = scaler.transform(input_df[['Amount', 'Time']])

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            # Display
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", "🚨 FRAUD" if pred == 1 else "✅ LEGITIMATE")
            col2.metric("Fraud Probability", f"{prob:.4f} ({prob*100:.2f}%)")
            col3.metric("True Label", "Fraud" if true_label == 1 else "Legitimate")

            if pred == 1 and true_label == 1:
                st.error("🚨 Correct! This transaction IS fraudulent and was detected.")
            elif pred == 0 and true_label == 0:
                st.success("✅ Correct! This is a legitimate transaction.")
            elif pred == 1 and true_label == 0:
                st.warning("⚠️ False Positive: Predicted FRAUD but it's actually legitimate.")
            else:
                st.error("❌ False Negative: Predicted LEGITIMATE but it's actually FRAUD!")

            # Confidence gauge
                fig, ax = plt.subplots(figsize=(7, 3))
                bar_color = COLORS['fraud'] if prob > 0.5 else COLORS['legit']
                ax.barh(['Fraud Probability'], [prob], color=bar_color, height=0.4)
                ax.barh(['Fraud Probability'], [1 - prob], left=[prob],
                    color='#e0e0e0', height=0.4)
                ax.axvline(x=0.5, color='#333', linestyle='--', lw=1.5, label='Decision Boundary (0.5)')
            ax.set_xlim(0, 1)
            ax.set_title(f"Fraud Probability: {prob:.4f}", fontweight='bold')
            ax.legend()
            ax.set_xlabel("Probability")
            st.pyplot(fig)

            with st.expander("📋 View Transaction Features"):
                feat_df = pd.DataFrame(sample).T
                st.dataframe(feat_df, width='stretch')

        st.markdown("---")
        st.markdown("#### Option 2: Enter transaction details manually")
        with st.expander("✏️ Manual Input"):
            st.markdown("Enter the feature values for a custom transaction:")
            cols = st.columns(4)
            manual_vals = {}
            all_features = [c for c in df.columns if c != 'Class']
            for i, feat in enumerate(all_features):
                with cols[i % 4]:
                    default_val = float(df[feat].mean())
                    manual_vals[feat] = st.number_input(feat, value=default_val,
                                                         format="%.4f", key=f"m_{feat}")

            if st.button("🔮 Predict Manually", use_container_width=True):
                model = st.session_state['model']
                scaler = st.session_state['scaler']
                manual_df = pd.DataFrame([manual_vals])
                # Transform both Amount and Time using the trained scaler
                manual_df[['Amount', 'Time']] = scaler.transform(manual_df[['Amount', 'Time']])

                pred = model.predict(manual_df)[0]
                prob = model.predict_proba(manual_df)[0][1]

                if pred == 1:
                    st.error(f"🚨 **FRAUD DETECTED** — Probability: {prob:.4f}")
                else:
                    st.success(f"✅ **LEGITIMATE** — Fraud Probability: {prob:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Credit Card Fraud Detection | Machine Learning Project | IEEE Format Report"
    "</div>",
    unsafe_allow_html=True
)
