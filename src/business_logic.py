import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def discover_interactions(X_train_woe, y_train):
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train_woe, y_train)
    return pd.DataFrame({
        'Feature': X_train_woe.columns,
        'RF_Importance': rf.feature_importances_,
    }).sort_values(by='RF_Importance', ascending=False)


def calculate_portfolio_profit(y_true, y_prob, cutoff_prob, loan_amounts, macro_stress_multiplier=1.0):
    """
    Translates model accuracy into estimated P&L.
    Includes a macro_stress_multiplier to simulate economic downturns where 
    default rates spike across the approved portfolio.
    """
    interest_rate, loss_given_default, loan_term_years = 0.15, 0.80, 3
    approved = y_prob < cutoff_prob

    results = pd.DataFrame({
        'actual_default': y_true.values,
        'approved': approved.values,
        'loan_amount': loan_amounts.values,
    })

    approved_loans = results[results['approved']]
    
    # Revenue from loans that didn't default
    good_loans = approved_loans[approved_loans['actual_default'] == 0]
    revenue = good_loans['loan_amount'].sum() * interest_rate * loan_term_years

    # Losses from defaulted loans (Scaled by the Macro Stress Multiplier)
    bad_loans = approved_loans[approved_loans['actual_default'] == 1]
    base_losses = bad_loans['loan_amount'].sum() * loss_given_default
    
    # Simulate stress: If stress is 1.5x, losses increase by 50%
    stressed_losses = base_losses * macro_stress_multiplier

    net_profit = revenue - stressed_losses
    approval_rate = approved.mean() * 100

    return net_profit, revenue, stressed_losses, approval_rate


def audit_fairness(df, region_col, score_col, cutoff_score):
    df_audit = df.copy()
    df_audit['Approved'] = df_audit[score_col] >= cutoff_score
    return df_audit.groupby(region_col).agg(
        Total_Applicants=('Approved', 'count'),
        Approval_Rate=('Approved', lambda x: x.mean() * 100),
    ).reset_index()


def calculate_psi(expected, actual, buckets=10):
    """
    Calculates the Population Stability Index (PSI).
    Measures if the distribution of credit scores has shifted between training and production.
    < 0.1: No shift, 0.1 - 0.2: Minor shift, > 0.2: Major shift (Model needs retraining).
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)
    
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return np.sum(psi_values)