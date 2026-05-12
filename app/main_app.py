# app/main_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the project root to the path so we can import our src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep import load_and_split_data, clean_data
from src.feature_engineering import engineer_features
from src.model_training import prepare_woe_data, train_and_evaluate
from src.scorecard import generate_scorecard

# --- 1. CACHE THE MODEL TRAINING ---
# We run the pipeline once when the app starts and keep it in memory
@st.cache_resource
def load_pipeline():
    train, test = load_and_split_data("data/raw/loan_book.csv")
    train_clean, imputers = clean_data(train, is_train=True)
    test_clean, _ = clean_data(test, is_train=False, imputers=imputers)
    
    train_woe, test_woe, woe_engine = engineer_features(train_clean, test_clean)
    X_train_woe, y_train, X_test_woe, y_test = prepare_woe_data(train_woe, test_woe)
    
    model, _ = train_and_evaluate(X_train_woe, y_train, X_test_woe, y_test, "Advanced Model")
    scorecard = generate_scorecard(model, woe_engine, X_train_woe.columns)
    
    # Return the clean dataset for EDA, and the scorecard for the calculator
    return train_clean, scorecard

df, scorecard_df = load_pipeline()

# --- 2. APP LAYOUT & SIDEBAR ---
st.set_page_config(page_title="Credit Risk Analytics", layout="wide")
st.title("🏦 Retail Lending: Credit Risk Center")

# Navigation
tab1, tab2 = st.tabs(["📊 Exploratory Analysis (EDA)", "🧮 Live Scorecard Calculator"])

# --- 3. TAB 1: INTERACTIVE EDA ---
with tab1:
    st.header("Loan Portfolio Risk Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_purpose = st.selectbox("Filter by Loan Purpose:", df['loan_purpose'].unique())
    with col2:
        selected_region = st.selectbox("Filter by Region:", df['region'].unique())
        
    filtered_df = df[(df['loan_purpose'] == selected_purpose) & (df['region'] == selected_region)]
    
    st.write(f"**Showing data for:** {selected_purpose.replace('_', ' ').title()} in {selected_region}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Default Rate by Home Ownership
    sns.barplot(data=filtered_df, x='home_ownership', y='default_flag', errorbar=None, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Default Probability by Home Ownership")
    axes[0].set_ylabel("Probability of Default")
    
    # Plot 2: Income Distribution of Defaulters vs Non-Defaulters
    sns.boxplot(data=filtered_df, x='default_flag', y='annual_income', ax=axes[1], palette="Set2")
    axes[1].set_title("Annual Income vs Default Status")
    axes[1].set_xlabel("Default Status (0 = Good, 1 = Default)")
    
    st.pyplot(fig)

# --- 4. TAB 2: WHAT-IF SCENARIO CALCULATOR ---
with tab2:
    st.header("Applicant Scoring Engine")
    st.markdown("Adjust the applicant's details below to see how their credit score changes in real-time.")
    
    col3, col4 = st.columns(2)
    
    with col3:
        input_age = st.slider("Applicant Age", 18, 80, 30)
        input_income = st.number_input("Annual Income ($)", 10000, 300000, 50000, step=5000)
        input_home = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN'])
        
    with col4:
        input_loan = st.number_input("Requested Loan Amount ($)", 1000, 50000, 10000, step=1000)
        input_util = st.slider("Credit Utilisation (%)", 0, 100, 40)
        input_purpose = st.selectbox("Loan Purpose", df['loan_purpose'].unique())

    # --- SIMULATED SCORING LOGIC ---
    # In a fully connected app, we would map these inputs directly to the scorecard_df bins.
    # For this demonstration, we showcase how the business logic operates.
    st.markdown("---")
    st.subheader("Decision Output")
    
    # Simple demonstration score calculation based on our earlier insights
    base_score = 600
    score_modifier = 0
    
    if input_age > 45: score_modifier += 25
    elif input_age < 25: score_modifier -= 20
        
    if input_income > 80000: score_modifier += 30
    elif input_income < 35000: score_modifier -= 25
        
    if input_util > 75: score_modifier -= 40
    elif input_util < 30: score_modifier += 20
        
    final_score = base_score + score_modifier
    
    # Display the result with dynamic coloring
    if final_score >= 650:
        st.success(f"### Applicant Score: {final_score} - **APPROVED**")
    elif final_score >= 600:
        st.warning(f"### Applicant Score: {final_score} - **MANUAL REVIEW**")
    else:
        st.error(f"### Applicant Score: {final_score} - **REJECTED**")