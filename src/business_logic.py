# src/business_logic.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def discover_interactions(X_train_woe, y_train):
    """
    The 'Shadow Model' Approach:
    Trains a fast, complex ML model to find non-linear patterns that 
    Logistic Regression might miss. We use this to generate insights for the business.
    """
    # Train a non-linear Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train_woe, y_train)
    
    # Extract feature importance
    importances = pd.DataFrame({
        'Feature': X_train_woe.columns,
        'RF_Importance': rf.feature_importances_
    }).sort_values(by='RF_Importance', ascending=False)
    
    return importances

def calculate_portfolio_profit(y_true, y_prob, cutoff_prob, loan_amounts, interest_rate=0.15, loss_given_default=0.80):
    """
    Translates model accuracy into Dollars. 
    If probability of default < cutoff, we approve the loan.
    """
    # Approve if their risk is BELOW our acceptable threshold
    approved = y_prob < cutoff_prob
    
    results = pd.DataFrame({
        'actual_default': y_true,
        'approved': approved,
        'loan_amount': loan_amounts
    })
    
    approved_loans = results[results['approved'] == True]
    
    # Revenue: Good loans pay back the principal + interest
    good_loans = approved_loans[approved_loans['actual_default'] == 0]
    revenue = good_loans['loan_amount'].sum() * interest_rate
    
    # Loss: Bad loans default. We lose the principal * Loss Given Default (LGD)
    bad_loans = approved_loans[approved_loans['actual_default'] == 1]
    losses = bad_loans['loan_amount'].sum() * loss_given_default
    
    net_profit = revenue - losses
    approval_rate = approved.mean() * 100
    
    return net_profit, revenue, losses, approval_rate

def audit_fairness(df, region_col, score_col, cutoff_score):
    """
    Checks if the model unfairly penalizes certain demographics/regions.
    """
    df_audit = df.copy()
    df_audit['Approved'] = df_audit[score_col] >= cutoff_score
    
    fairness_report = df_audit.groupby(region_col).agg(
        Total_Applicants=('Approved', 'count'),
        Approval_Rate=('Approved', lambda x: x.mean() * 100)
    ).reset_index()
    
    return fairness_report