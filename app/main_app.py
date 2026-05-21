
import streamlit as st


st.set_page_config(page_title="Strategic Model Dashboard", layout="wide", page_icon="🏦")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from sklearn.metrics import roc_curve, auc, precision_score, recall_score

sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep import load_and_split_data, clean_data
from src.feature_engineering import engineer_features
from src.model_training import prepare_baseline_data, prepare_woe_data, train_and_evaluate
from src.scorecard import generate_scorecard
from src.business_logic import calculate_portfolio_profit, audit_fairness, discover_interactions, calculate_psi

@st.cache_resource(show_spinner="Training ML Models (CV Tuning active)... This may take a minute.")
def load_and_train_pipeline():
    train, test = load_and_split_data("data/raw/loan_book.csv")
    train_raw = train.copy() 
    
    train_clean, imputers = clean_data(train, is_train=True)
    test_clean, _ = clean_data(test, is_train=False, imputers=imputers)

    X_train_base, y_train_base, X_test_base, y_test_base = prepare_baseline_data(train_clean, test_clean)
    base_model, base_probs = train_and_evaluate(X_train_base, y_train_base, X_test_base, y_test_base, "Baseline Model")

    train_woe, test_woe, woe_engine = engineer_features(train_clean, test_clean)
    X_train_woe, y_train_woe, X_test_woe, y_test_woe = prepare_woe_data(train_woe, test_woe)

    # ADV FEAT: Hyperparameter Tuning via Cross Validation
    adv_model, adv_probs = train_and_evaluate(X_train_woe, y_train_woe, X_test_woe, y_test_woe, "Advanced Model", use_cv=True)
    scorecard_df = generate_scorecard(adv_model, woe_engine, X_train_woe.columns)

    shadow_insights = discover_interactions(X_train_woe, y_train_woe)

    woe_cols = [c for c in X_test_woe.columns]
    test_scores = _score_portfolio(X_test_woe, scorecard_df, woe_engine, woe_cols)
    train_scores = _score_portfolio(X_train_woe, scorecard_df, woe_engine, woe_cols) # Added for PSI

    test_clean = test_clean.copy()
    test_clean['predicted_default_prob'] = adv_probs
    test_clean['baseline_default_prob'] = base_probs
    test_clean['credit_score'] = test_scores
    test_clean['actual_default'] = y_test_woe.values

    # ADV FEAT: Calculate PSI
    psi_value = calculate_psi(train_scores, test_scores)

    return train_raw, train_clean, test_clean, scorecard_df, shadow_insights, woe_engine, psi_value, adv_model.C_[0]


def _score_portfolio(X_woe, scorecard_df, woe_engine, woe_cols):
    scores = np.zeros(len(X_woe))
    for feature_col in woe_cols:
        original_feature = feature_col.replace('_woe', '')
        feature_rows = scorecard_df[scorecard_df['Feature'] == original_feature].copy()
        if feature_rows.empty: continue
        
        woe_values = feature_rows['WoE'].values
        points_values = feature_rows['Points'].values
        applicant_woes = X_woe[feature_col].values
        
        for i, aw in enumerate(applicant_woes):
            idx = np.argmin(np.abs(woe_values - aw))
            scores[i] += points_values[idx]
    return scores.astype(int)

# Load data
train_raw, df_train, df_test, scorecard_df, shadow_insights, woe_engine, psi_value, optimal_c = load_and_train_pipeline()

st.title("Retail Lending: Strategic Model Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Interactive EDA", 
    "2. Model Comparison & PSI", 
    "3. P&L & Stress Testing", 
    "4. Fairness Audit"
])

# --- TAB 1: INTERACTIVE EDA ---
with tab1:
    st.header("Exploratory Data Analysis")
    eda_sub1, eda_sub2 = st.tabs(["Univariate & WoE", "Bivariate Explorer"])
    
    with eda_sub1:
        col_f1, col_f2 = st.columns([1, 2])
        feature = col_f1.selectbox("Select Feature", df_train.drop(columns=['default_flag']).columns)
        plot_data = df_train.sample(n=min(3000, len(df_train)), random_state=42)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data=plot_data, x=feature, hue='default_flag', multiple='stack', palette='coolwarm', ax=ax)
            ax.legend(title='Default Flag', loc='upper right')
            st.pyplot(fig)
        with col_p2:
            if feature in woe_engine.woe_dicts:
                iv_val = woe_engine.iv_scores.get(feature, 0)
                st.write(f"**Weight of Evidence (WoE)** | IV: {iv_val:.3f}")
                woe_df = pd.DataFrame(list(woe_engine.woe_dicts[feature].items()), columns=['Bin', 'WoE'])
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=woe_df, x='Bin', y='WoE', palette='viridis', ax=ax2)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2)

    with eda_sub2:
        col_x, col_y = st.columns(2)
        x_feat = col_x.selectbox("X-Axis Feature", df_train.columns, index=0)
        y_feat = col_y.selectbox("Y-Axis Feature", df_train.columns, index=1)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        scatter_data = df_train.sample(n=min(1000, len(df_train)), random_state=42)
        if pd.api.types.is_numeric_dtype(df_train[x_feat]) and pd.api.types.is_numeric_dtype(df_train[y_feat]):
            sns.scatterplot(data=scatter_data, x=x_feat, y=y_feat, hue='default_flag', alpha=0.6, palette='coolwarm', ax=ax3)
        else:
            sns.boxplot(data=scatter_data, x=x_feat, y=y_feat, hue='default_flag', palette='coolwarm', ax=ax3)
        ax3.legend(loc='upper right')
        st.pyplot(fig3)

# --- TAB 2: MODEL COMPARISON & PSI ---
with tab2:
    st.header("Model Evaluation & Stability")
    
    fpr_b, tpr_b, _ = roc_curve(df_test['actual_default'], df_test['baseline_default_prob'])
    auc_b, gini_b = auc(fpr_b, tpr_b), (auc(fpr_b, tpr_b) * 2) - 1
    
    fpr_a, tpr_a, _ = roc_curve(df_test['actual_default'], df_test['predicted_default_prob'])
    auc_a, gini_a = auc(fpr_a, tpr_a), (auc(fpr_a, tpr_a) * 2) - 1
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("ROC Curve (Tuned via 5-Fold CV)")
        st.caption(f"Optimal Regularization Strength (C): **{optimal_c:.4f}**")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        ax4.plot(fpr_b, tpr_b, label=f'Baseline (AUC = {auc_b:.3f})', linestyle='--')
        ax4.plot(fpr_a, tpr_a, label=f'Advanced (AUC = {auc_a:.3f})', linewidth=2)
        ax4.plot([0, 1], [0, 1], 'k--', label='Random')
        ax4.legend(loc='lower right')
        st.pyplot(fig4)
        
    with col_m2:
        st.subheader("Population Stability Index (PSI)")
        st.metric("Portfolio PSI Score", f"{psi_value:.4f}")
        if psi_value < 0.1:
            st.success("**Stable:** The distribution of test scores matches training data perfectly.")
        elif psi_value < 0.2:
            st.warning("**Warning:** Minor shift detected in applicant population.")
        else:
            st.error("**Critical Shift:** Major data drift detected. Retraining required.")
        st.dataframe(scorecard_df, height=300)

# --- TAB 3: BUSINESS VALUE & STRESS TESTING ---
with tab3:
    st.header("Portfolio Profit & Macro-Economic Stress Testing")
    
    col_s1, col_s2 = st.columns(2)
    cutoff_prob = col_s1.slider("Max Default Probability Threshold (%)", 10, 80, 40, 1) / 100.0
    stress_multiplier = col_s2.slider("Macro Stress Multiplier (x Defaults)", 1.0, 3.0, 1.0, 0.1)

    net_profit, rev, losses, app_rate = calculate_portfolio_profit(
        y_true=df_test['actual_default'], y_prob=df_test['predicted_default_prob'],
        cutoff_prob=cutoff_prob, loan_amounts=df_test['loan_amount'], 
        macro_stress_multiplier=stress_multiplier
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approval Rate", f"{app_rate:.1f}%")
    c2.metric("Interest Revenue", f"${rev:,.0f}")
    c3.metric(f"Stressed Losses ({stress_multiplier}x)", f"-${losses:,.0f}")
    c4.metric("Net Profit", f"${net_profit:,.0f}")

    st.divider()
    st.subheader("Next Steps: Reject Inference (Selection Bias Mitigation)")
    st.info("Because the model is trained only on 'approved' applicants (the existing loan book), it suffers from selection bias. To deploy this to production, we would use **Fuzzy Augmentation** (Reject Inference) to predict outcomes on historical rejected applicants, append them to the training set, and retrain the model to capture the full market distribution. The backend utility `apply_reject_inference()` has been prepared for this phase.")

# --- TAB 4: FAIRNESS AUDIT ---
with tab4:
    st.header("Regulatory Fairness Audit")
    cutoff_score = st.slider("Credit Score Approval Cutoff", int(df_test['credit_score'].min()), int(df_test['credit_score'].max()), 600, 5)
    fairness_df = audit_fairness(df_test, 'region', 'credit_score', cutoff_score)

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=fairness_df, x='region', y='Approval_Rate', palette="viridis", ax=ax5)
    ax5.axhline(fairness_df['Approval_Rate'].mean(), color='red', linestyle='--', label='Average')
    ax5.legend(loc='lower right')
    st.pyplot(fig5)