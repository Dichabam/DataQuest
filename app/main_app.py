# app/main_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep import load_and_split_data, clean_data
from src.feature_engineering import engineer_features
from src.model_training import prepare_woe_data, train_and_evaluate
from src.scorecard import generate_scorecard
from src.business_logic import calculate_portfolio_profit, audit_fairness, discover_interactions

@st.cache_resource
def load_and_train_pipeline():
    # 1. Load & Clean
    train, test = load_and_split_data("data/raw/loan_book.csv")
    train_clean, imputers = clean_data(train, is_train=True)
    test_clean, _ = clean_data(test, is_train=False, imputers=imputers)
    
    # 2. Engineer WoE
    train_woe, test_woe, woe_engine = engineer_features(train_clean, test_clean)
    X_train_woe, y_train, X_test_woe, y_test = prepare_woe_data(train_woe, test_woe)
    
    # 3. Train Model
    model, test_probs = train_and_evaluate(X_train_woe, y_train, X_test_woe, y_test, "Advanced Model")
    scorecard = generate_scorecard(model, woe_engine, X_train_woe.columns)
    
    # 4. Generate ML Insights (Shadow Model)
    shadow_insights = discover_interactions(X_train_woe, y_train)
    
    # Attach predictions to test set for business logic
    test_clean['predicted_default_prob'] = test_probs
    # Convert probability to rough credit score (Range 300 - 850)
    test_clean['credit_score'] = 850 - (test_probs * 550) 
    test_clean['actual_default'] = y_test.values
    
    return train_clean, test_clean, scorecard, shadow_insights

df_train, df_test, scorecard_df, shadow_insights = load_and_train_pipeline()

# --- APP LAYOUT ---
st.set_page_config(page_title="Retail Lending Analytics", layout="wide", page_icon="🏦")
st.title("🏦 Retail Lending: Strategic Model Dashboard")
st.markdown("*A Multidisciplinary Framework balancing Predictive Power, Profitability, and Regulatory Compliance.*")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Risk Exploration", 
    "🧮 2. The Scorecard", 
    "💼 3. Profit Optimization (CEO View)", 
    "⚖️ 4. Fairness Audit (Regulator View)"
])

# --- TAB 1: EDA & SHADOW ML ---
with tab1:
    st.header("Risk Exploration & ML Insights")
    st.markdown("We use a complex **Random Forest Shadow Model** alongside traditional EDA to detect non-linear risk patterns.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Default Probability by Home Ownership")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df_train, x='home_ownership', y='default_flag', errorbar=None, palette="Blues_d", ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Shadow ML Feature Importance")
        st.dataframe(shadow_insights.head(5), use_container_width=True)
        st.info("💡 **Insight:** While Logistic Regression uses linear weights, our Shadow AI confirms that Income and Utilisation possess complex, non-linear predictive power.")

# --- TAB 2: SCORECARD ---
with tab2:
    st.header("Transparent Scoring Engine")
    st.markdown("This point system is derived from the Logistic Regression model. It is 100% interpretable.")
    st.dataframe(scorecard_df, use_container_width=True)

# --- TAB 3: THE BUSINESS FRONTIER ---
with tab3:
    st.header("Portfolio Profit Optimization")
    st.markdown("Move the slider to set the risk appetite. See how the model impacts the bottom line on our Test Portfolio.")
    
    # Interactive Cutoff Slider
    acceptable_risk = st.slider("Maximum Acceptable Default Probability (%)", min_value=10, max_value=80, value=40, step=1)
    cutoff_prob = acceptable_risk / 100.0
    
    net_profit, rev, losses, app_rate = calculate_portfolio_profit(
        y_true=df_test['actual_default'], 
        y_prob=df_test['predicted_default_prob'], 
        cutoff_prob=cutoff_prob, 
        loan_amounts=df_test['loan_amount']
    )
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Approval Rate", f"{app_rate:.1f}%")
    col2.metric("Interest Revenue", f"${rev:,.0f}")
    col3.metric("Default Losses", f"-${losses:,.0f}")
    col4.metric("Net Profit", f"${net_profit:,.0f}", delta="Optimized")
    
    if net_profit < 0:
        st.error("⚠️ **Warning:** The current risk threshold is too loose. Losses exceed interest revenue.")
    elif app_rate < 20:
        st.warning("⚠️ **Warning:** The threshold is too strict. We are turning away too many good customers and stunting growth.")
    else:
        st.success("✅ **Healthy Portfolio:** Balance of risk and reward achieved.")

# --- TAB 4: FAIRNESS AUDIT ---
with tab4:
    st.header("Regulatory Fairness Audit")
    st.markdown("Ensuring our Logistic Regression model does not systematically bias against specific geographic regions.")
    
    cutoff_score = 600 # Assume 600 is the hard cutoff for this test
    fairness_df = audit_fairness(df_test, 'region', 'credit_score', cutoff_score)
    
    st.write(f"**Approval Rates by Region (Assuming Cutoff Score of {cutoff_score})**")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=fairness_df, x='region', y='Approval_Rate', palette="viridis", ax=ax)
    ax.axhline(fairness_df['Approval_Rate'].mean(), color='red', linestyle='--', label='Average Approval Rate')
    ax.legend()
    st.pyplot(fig)
    
    variance = fairness_df['Approval_Rate'].max() - fairness_df['Approval_Rate'].min()
    if variance > 15:
        st.warning(f"**Audit Flag:** High variance ({variance:.1f}%) in approval rates across regions. Manual review recommended.")
    else:
        st.success(f"**Audit Passed:** Approval rates are relatively consistent across regions (Variance: {variance:.1f}%). No glaring geographical bias detected.")